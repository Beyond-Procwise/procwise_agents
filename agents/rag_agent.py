import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .base_agent import AgentOutput, AgentStatus, BaseAgent
from services.feedback_service import FeedbackSentiment, FeedbackService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QARecord:
    """Represents a single procurement Q&A pair."""

    question: str
    answer: str


@dataclass(frozen=True)
class TopicRecord:
    """Represents a procurement topic and its related prompts."""

    topic: str
    context_prompts: Tuple[str, ...]
    qas: Tuple[QARecord, ...]


class ResponseStructure:
    """Canonical section planning for structured answers."""

    STRUCTURES: Dict[str, Dict[str, Any]] = {
        "supplier_overview": {
            "sections": ["overview", "key_suppliers", "insights", "next_steps"],
            "style": "briefing",
        },
        "financial_analysis": {
            "sections": ["summary", "key_figures", "breakdown", "trends", "recommendations"],
            "style": "analytical",
        },
        "comparison": {
            "sections": ["overview", "comparison_table", "strengths_weaknesses", "recommendation"],
            "style": "comparative",
        },
        "policy_lookup": {
            "sections": ["policy_summary", "key_requirements", "examples", "related_policies"],
            "style": "reference",
        },
        "exploratory": {
            "sections": ["context", "findings", "details", "implications"],
            "style": "investigative",
        },
        "simple_lookup": {
            "sections": ["direct_answer", "context"],
            "style": "concise",
        },
    }


class RAGAgent(BaseAgent):
    """Retrieval agent backed by a static procurement knowledge base."""

    _DATASET_PATH = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "reference_data"
        / "procwise_mvp_chat_questions.json"
    )

    _DATASET: Optional[Tuple[TopicRecord, ...]] = None
    _EMBEDDING_CACHE: Dict[int, Tuple[np.ndarray, List[np.ndarray]]] = {}

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self._session_topics: Dict[Tuple[str, str], int] = {}
        self.feedback_service = FeedbackService(agent_nick)
        self._last_interaction: Dict[str, Any] = {}
        self._ensure_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        product_type: Optional[str] = None,
        **_: Any,
    ) -> AgentOutput:
        """Return the documented answer for ``query`` from the static dataset."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        query = query.strip()
        original_query = query

        feedback_detected = False
        feedback_id: Optional[int] = None
        feedback_sentiment: Optional[FeedbackSentiment] = None
        feedback_confidence = 0.0
        acknowledgment_text: Optional[str] = None
        ack_prefix: Optional[str] = None
        depth_mode = "standard"
        continuation_requested = False

        classification_query = original_query
        working_query = original_query

        previous_user = self._last_interaction.get("user_id")
        last_query = self._last_interaction.get("query")
        if previous_user and previous_user == user_id:
            if self.feedback_service.is_continuation_request(original_query):
                continuation_requested = True
                depth_mode = "expanded"
                base_query = last_query or original_query
                classification_query = base_query
                working_query = self._expand_query(base_query, original_query, mode="continuation")
            else:
                feedback_sentiment, feedback_confidence = self.feedback_service.detect_feedback(original_query)
                if feedback_sentiment != FeedbackSentiment.NEUTRAL and feedback_confidence >= 0.4:
                    feedback_detected = True
                    acknowledgment_text = self.feedback_service.generate_acknowledgment(
                        feedback_sentiment, original_query
                    )
                    payload_available = bool(self._last_interaction)
                    if payload_available:
                        feedback_id = self.feedback_service.store_feedback(
                            user_id=user_id,
                            session_id=session_id,
                            query=self._last_interaction.get("query", classification_query),
                            response=self._last_interaction.get("response", ""),
                            feedback_message=original_query,
                            sentiment=feedback_sentiment,
                            confidence=feedback_confidence,
                            retrieved_doc_ids=self._last_interaction.get("doc_ids", []),
                            context_metadata={
                                "doc_type": doc_type,
                                "product_type": product_type,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                    if feedback_sentiment == FeedbackSentiment.POSITIVE:
                        data = {
                            "question": self._last_interaction.get("query", classification_query),
                            "answer": acknowledgment_text,
                            "topic": self._last_interaction.get("topic", "feedback_acknowledgment"),
                            "related_prompts": [],
                            "structured": False,
                            "structure_type": "feedback_acknowledgment",
                            "sections": [],
                            "style": "acknowledgment",
                            "main_points": [],
                            "source_ids": self._last_interaction.get("doc_ids", []),
                            "original_answer": self._last_interaction.get("response", ""),
                            "feedback": {
                                "captured": True,
                                "feedback_id": feedback_id,
                                "sentiment": feedback_sentiment.value,
                                "confidence": feedback_confidence,
                                "acknowledgment": acknowledgment_text,
                            },
                        }
                        agentic_plan = "1. Recognise user feedback.\n2. Store it for training.\n3. Acknowledge the positive sentiment."
                        context_snapshot = {
                            "feedback_id": feedback_id,
                            "feedback_sentiment": feedback_sentiment.value,
                            "feedback_confidence": feedback_confidence,
                        }
                        return AgentOutput(
                            status=AgentStatus.SUCCESS,
                            data=data,
                            agentic_plan=agentic_plan,
                            context_snapshot=context_snapshot,
                            confidence=1.0,
                        )

                    ack_prefix = acknowledgment_text
                    depth_mode = "expanded"
                    base_query = last_query or original_query
                    classification_query = base_query
                    working_query = self._expand_query(base_query, None, mode="improvement")

        query = working_query

        key = self._session_key(user_id, session_id)
        query_vector = self._encode_text(query)
        topic_index, topic_score = self._select_topic(query_vector, key)
        question_index, question_score = self._select_question(query_vector, topic_index)

        topic_entry = self._dataset[topic_index]
        qa_entry = topic_entry.qas[question_index]
        related_prompts = self._build_related_prompts(topic_entry, qa_entry)

        self._session_topics[key] = topic_index

        record_id = f"static-{topic_index}-{question_index}"
        payload = {
            "record_id": record_id,
            "topic": topic_entry.topic,
            "question": qa_entry.question,
            "summary": qa_entry.answer,
            "document_type": "static_reference",
        }
        docs = [SimpleNamespace(payload=payload)]

        answer_scope = self._compose_answer_scope(topic_entry, question_index, depth_mode)
        extracted = self._extract_answer_signals(answer_scope, record_id, depth_mode)
        query_type = self._classify_query_type(classification_query)
        plan = self._plan_response_structure(classification_query, query_type, extracted)

        policy_payload: Optional[Dict[str, Any]] = None
        if query_type == "policy_lookup":
            policy_payload = self._extract_policy_payload(
                policy_name=self._derive_policy_name(classification_query, topic_entry),
                topic_entry=topic_entry,
                focus_answer=qa_entry.answer,
                depth_mode=depth_mode,
            )

        structured_answer = self._generate_structured_response(
            classification_query,
            extracted,
            docs,
            plan,
            depth_mode=depth_mode,
            policy_payload=policy_payload,
        )
        if ack_prefix:
            structured_answer = (
                f"{ack_prefix}\n\n{structured_answer}" if structured_answer else ack_prefix
            )
        followups = self._generate_contextual_followups(
            classification_query, query_type, extracted, depth_mode
        )

        response = {
            "question": qa_entry.question,
            "answer": structured_answer,
            "topic": topic_entry.topic,
            "related_prompts": followups if followups else related_prompts,
            "structured": query_type != "simple_lookup",
            "structure_type": plan["type"],
            "sections": plan["sections"],
            "style": plan["style"],
            "main_points": extracted.get("main_points", []),
            "source_ids": extracted.get("source_ids", []),
            "original_answer": qa_entry.answer,
        }

        if feedback_detected and feedback_sentiment is not None:
            response["feedback"] = {
                "captured": True,
                "feedback_id": feedback_id,
                "sentiment": feedback_sentiment.value,
                "confidence": feedback_confidence,
                "acknowledgment": acknowledgment_text,
            }

        agentic_plan = (
            "1. Match the query to the closest procurement topic.\n"
            "2. Retrieve the exact answer from the static knowledge base.\n"
            "3. Shape a structured response with contextual follow-ups."
        )

        context_snapshot = {
            "topic_similarity": float(topic_score),
            "question_similarity": float(question_score),
            "session_topic": topic_entry.topic,
            "response_depth": depth_mode,
        }

        if continuation_requested:
            context_snapshot["continuation"] = True

        if feedback_detected and feedback_sentiment is not None:
            context_snapshot.update(
                {
                    "feedback_id": feedback_id,
                    "feedback_sentiment": feedback_sentiment.value,
                    "feedback_confidence": feedback_confidence,
                }
            )

        logger.debug(
            "RAGAgent resolved query '%s' to topic '%s' question '%s'",
            query,
            topic_entry.topic,
            qa_entry.question,
        )

        record_id = payload["record_id"]
        self._last_interaction = {
            "user_id": user_id,
            "session_id": session_id,
            "query": classification_query,
            "response": structured_answer,
            "doc_ids": [record_id],
            "topic": topic_entry.topic,
            "query_type": query_type,
            "topic_index": topic_index,
            "question_index": question_index,
            "depth_mode": depth_mode,
        }

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=response,
            agentic_plan=agentic_plan,
            context_snapshot=context_snapshot,
            confidence=float(min(topic_score, question_score)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Load the dataset and build embeddings for the current embedder."""

        cls = self.__class__
        if cls._DATASET is None:
            cls._DATASET = self._load_dataset()

        embedder = getattr(self.agent_nick, "embedding_model", None)
        if embedder is None or not hasattr(embedder, "encode"):
            raise ValueError(
                "agent_nick.embedding_model must provide an 'encode' method for RAGAgent"
            )

        embedder_id = id(embedder)
        if embedder_id not in cls._EMBEDDING_CACHE:
            topic_texts: List[str] = []
            question_vectors: List[np.ndarray] = []

            for topic_entry in cls._DATASET:
                topic_text = " ".join(
                    [topic_entry.topic]
                    + [qa.question for qa in topic_entry.qas]
                    + list(topic_entry.context_prompts)
                )
                topic_texts.append(topic_text)

            topic_vectors = self._encode_texts(topic_texts)

            for topic_entry in cls._DATASET:
                questions = [qa.question for qa in topic_entry.qas]
                question_vectors.append(self._encode_texts(questions))

            cls._EMBEDDING_CACHE[embedder_id] = (topic_vectors, question_vectors)

        self._dataset = cls._DATASET
        topic_vectors, question_vectors = cls._EMBEDDING_CACHE[embedder_id]
        self._topic_vectors = topic_vectors
        self._question_vectors = question_vectors

    def _load_dataset(self) -> Tuple[TopicRecord, ...]:
        if not self._DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Static knowledge base not found at {self._DATASET_PATH}"
            )

        with self._DATASET_PATH.open("r", encoding="utf-8") as handle:
            raw_entries = json.load(handle)

        topics: List[TopicRecord] = []
        for entry in raw_entries:
            context_prompts = tuple(entry.get("context_prompts", []))
            qas = tuple(
                QARecord(question=qa["question"], answer=qa["answer"])
                for qa in entry.get("qas", [])
            )
            topics.append(
                TopicRecord(
                    topic=entry["topic"],
                    context_prompts=context_prompts,
                    qas=qas,
                )
            )

        return tuple(topics)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        embedder = self.agent_nick.embedding_model
        vectors = embedder.encode(texts)
        array = np.array(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def _encode_text(self, text: str) -> np.ndarray:
        return self._encode_texts([text])[0]

    def _session_key(self, user_id: str, session_id: Optional[str]) -> Tuple[str, str]:
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be provided for session tracking")
        return user_id, session_id or "__default__"

    def _select_topic(
        self, query_vector: np.ndarray, session_key: Tuple[str, str]
    ) -> Tuple[int, float]:
        similarities = self._topic_vectors @ query_vector
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        stored_idx = self._session_topics.get(session_key)
        if stored_idx is not None:
            stored_score = float(similarities[stored_idx])
            if best_idx != stored_idx and best_score > stored_score + 0.05:
                chosen_idx = best_idx
                chosen_score = best_score
            else:
                chosen_idx = stored_idx
                chosen_score = stored_score
        else:
            chosen_idx = best_idx
            chosen_score = best_score

        return chosen_idx, chosen_score

    def _select_question(
        self, query_vector: np.ndarray, topic_idx: int
    ) -> Tuple[int, float]:
        question_vectors = self._question_vectors[topic_idx]
        similarities = question_vectors @ query_vector
        best_idx = int(np.argmax(similarities))
        return best_idx, float(similarities[best_idx])

    def _build_related_prompts(
        self, topic: TopicRecord, qa_entry: QARecord
    ) -> List[str]:
        prompts = list(topic.context_prompts)
        if not prompts:
            prompts = [other.question for other in topic.qas if other != qa_entry]
        return prompts

    def _expand_query(
        self, base_query: str, follow_up: Optional[str], mode: str = "continuation"
    ) -> str:
        base = (base_query or "").strip()
        if not base:
            return (follow_up or "").strip()

        clause = self._build_follow_up_clause(follow_up or "", mode)
        if not clause:
            return base
        return f"{base} — {clause}"

    def _build_follow_up_clause(self, follow_up: str, mode: str) -> str:
        cleaned = follow_up.strip().rstrip("?.!")
        cleaned = re.sub(r"^(good|great|ok|okay|thanks)[,\s]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(please|kindly)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        if mode == "improvement":
            if cleaned:
                cleaned_lower = cleaned.lower()
                if any(word in cleaned_lower for word in ["wrong", "incorrect", "fix"]):
                    return "provide a corrected, comprehensive answer with the right policy guardrails."
                return f"provide a corrected, comprehensive answer and address: {cleaned}"
            return "provide a corrected, comprehensive answer with full policy detail."

        if not cleaned:
            return "provide comprehensive details and broaden the explanation."

        lowered = cleaned.lower()
        if any(phrase in lowered for phrase in ["more detail", "be more detailed", "elaborate"]):
            return "provide comprehensive details and expand each section."
        if any(keyword in lowered for keyword in ["limit", "cap", "threshold", "spending"]):
            return "provide comprehensive details on the relevant limits and guardrails."
        if any(keyword in lowered for keyword in ["example", "examples", "case"]):
            return "provide comprehensive details and add concrete examples."
        if lowered.startswith("what about"):
            remainder = cleaned[10:].strip()
            return (
                f"provide comprehensive details covering {remainder}"
                if remainder
                else "provide comprehensive details."
            )
        if lowered.startswith("how about"):
            remainder = cleaned[8:].strip()
            return (
                f"provide comprehensive details including {remainder}"
                if remainder
                else "provide comprehensive details."
            )
        return f"provide comprehensive details covering: {cleaned}"

    # ------------------------------------------------------------------
    # Structured response planning and generation
    # ------------------------------------------------------------------
    def _compose_answer_scope(
        self, topic_entry: TopicRecord, question_index: int, depth_mode: str
    ) -> str:
        primary_answer = topic_entry.qas[question_index].answer
        if depth_mode != "expanded":
            return primary_answer

        additional_answers: List[str] = []
        for idx, qa in enumerate(topic_entry.qas):
            if idx == question_index:
                continue
            additional_answers.append(qa.answer)
            if len(additional_answers) >= 3:
                break

        combined = " ".join([primary_answer] + additional_answers) if additional_answers else primary_answer
        return combined

    def _extract_answer_signals(
        self, answer: str, record_id: str, depth_mode: str = "standard"
    ) -> Dict[str, Any]:
        sentences = self._split_sentences(answer)
        limit = 10 if depth_mode == "expanded" else 6
        main_points = sentences[:limit] if sentences else [answer.strip()]

        numbers = re.findall(r"[£$]\s*[\d,.]+|\b\d+(?:\.\d+)?%", answer)
        entity_candidates = re.findall(
            r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*\b", answer
        )
        stop_words = {"The", "For", "In", "On", "And", "Procurement", "This", "That"}
        entities = [
            name
            for name in entity_candidates
            if name not in stop_words and len(name) > 2
        ]

        return {
            "main_points": main_points,
            "source_ids": [record_id],
            "entities": list(dict.fromkeys(entities)),
            "numbers": numbers,
        }

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9£])", text.strip())
        return [segment.strip() for segment in raw if segment and segment.strip()]

    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["supplier", "vendor", "who supplies", "procurement from"]
        ):
            if any(
                word in query_lower
                for word in ["compare", "vs", "versus", "difference between"]
            ):
                return "comparison"
            return "supplier_overview"

        if any(
            word in query_lower
            for word in ["spend", "cost", "price", "budget", "savings", "value", "£", "$"]
        ):
            return "financial_analysis"

        if any(
            word in query_lower
            for word in ["policy", "rule", "requirement", "compliance", "regulation"]
        ) or (
            "can i" in query_lower
            and any(term in query_lower for term in ["claim", "use", "submit", "expense"])
        ) or ("cannot" in query_lower and "claim" in query_lower):
            return "policy_lookup"

        if any(
            word in query_lower
            for word in ["compare", "vs", "versus", "better", "best", "difference"]
        ):
            return "comparison"

        if query.count(" ") < 5 and any(
            word in query_lower for word in ["what is", "who is", "when", "where"]
        ):
            return "simple_lookup"

        return "exploratory"

    def _plan_response_structure(
        self, query: str, query_type: str, extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        template = ResponseStructure.STRUCTURES.get(
            query_type, ResponseStructure.STRUCTURES["exploratory"]
        )
        headers = self._generate_section_headers(query, query_type, extracted_data)
        return {
            "type": query_type,
            "style": template["style"],
            "sections": headers,
            "use_tables": query_type in {"comparison", "financial_analysis"},
            "use_bullets": len(extracted_data.get("main_points", [])) > 2,
        }

    def _generate_section_headers(
        self, query: str, query_type: str, extracted: Dict[str, Any]
    ) -> List[str]:
        template = ResponseStructure.STRUCTURES.get(
            query_type, ResponseStructure.STRUCTURES["exploratory"]
        )
        base = template["sections"]
        mapping = {
            "overview": "Overview",
            "key_suppliers": "Primary Suppliers",
            "insights": "Key Insights",
            "next_steps": "What This Means",
            "summary": "Summary",
            "key_figures": "Key Figures",
            "breakdown": "Spending Breakdown",
            "trends": "Trends & Patterns",
            "recommendations": "Recommendations",
            "comparison_table": "Side-by-Side Comparison",
            "strengths_weaknesses": "Analysis",
            "policy_summary": "Policy Summary",
            "key_requirements": "Key Requirements",
            "examples": "Practical Examples",
            "related_policies": "Related Policies",
            "context": "Context",
            "findings": "Findings",
            "details": "Supporting Details",
            "implications": "Implications",
            "direct_answer": "Answer",
        }
        return [mapping.get(section, section.replace("_", " ").title()) for section in base]

    def _generate_structured_response(
        self,
        query: str,
        extracted_data: Dict[str, Any],
        retrieved_docs: List[Any],
        plan: Dict[str, Any],
        *,
        depth_mode: str = "standard",
        policy_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        query_type = plan["type"]

        if query_type == "policy_lookup":
            if not policy_payload:
                policy_payload = {
                    "policy_name": self._format_policy_title(query),
                    "overview": self._ensure_sentence(extracted_data.get("main_points", [""])[0])
                    if extracted_data.get("main_points")
                    else "",
                    "requirements": [],
                    "restrictions": [],
                    "spending_limits": [],
                    "approval_process": [],
                    "examples": [],
                    "exceptions": [],
                }
            return self._render_policy_response(policy_payload, depth_mode)

        opening = self._generate_opening_section(query, extracted_data, query_type)
        closing = self._generate_closing_section(query, extracted_data, query_type)

        template = ResponseStructure.STRUCTURES.get(
            query_type, ResponseStructure.STRUCTURES["exploratory"]
        )
        section_keys = template["sections"]
        headers = plan["sections"]

        section_bodies: Dict[str, str] = {}

        if query_type == "supplier_overview":
            generated = self._generate_supplier_sections(extracted_data, retrieved_docs)
            section_bodies["overview"] = opening
            section_bodies["key_suppliers"] = generated[0] if generated else ""
            section_bodies["insights"] = generated[1] if len(generated) > 1 else ""
            section_bodies["next_steps"] = closing
        elif query_type == "financial_analysis":
            generated = self._generate_financial_sections(extracted_data, retrieved_docs)
            section_bodies["summary"] = opening
            section_bodies["key_figures"] = generated[0] if generated else ""
            section_bodies["breakdown"] = generated[1] if len(generated) > 1 else ""
            section_bodies["trends"] = generated[2] if len(generated) > 2 else ""
            section_bodies["recommendations"] = closing
        elif query_type == "comparison":
            generated = self._generate_comparison_sections(extracted_data, retrieved_docs)
            section_bodies["overview"] = opening
            section_bodies["comparison_table"] = generated[0] if generated else ""
            section_bodies["strengths_weaknesses"] = (
                generated[1] if len(generated) > 1 else ""
            )
            section_bodies["recommendation"] = closing
        elif query_type == "simple_lookup":
            main_section = self._generate_simple_response(query, extracted_data, retrieved_docs)
            section_bodies["direct_answer"] = main_section
            section_bodies["context"] = closing
        else:
            generated = self._generate_exploratory_sections(extracted_data, retrieved_docs)
            section_bodies["context"] = opening
            if generated:
                section_bodies["findings"] = generated[0]
            if len(generated) > 1:
                details = generated[1]
                if len(generated) > 2:
                    details = "\n\n".join([details] + generated[2:])
                section_bodies["details"] = details
            section_bodies.setdefault("details", "")
            section_bodies["implications"] = closing

        formatted_sections: List[str] = []
        for header, key in zip_longest(headers, section_keys, fillvalue=""):
            if not header:
                continue
            body = section_bodies.get(key, "") if key else ""
            if body and body.strip():
                formatted_sections.append(f"## {header}\n{body.strip()}")

        if not formatted_sections:
            return opening if query_type != "simple_lookup" else section_bodies["direct_answer"]

        return "\n\n".join(formatted_sections)

    def _generate_opening_section(
        self, query: str, data: Dict[str, Any], query_type: str
    ) -> str:
        main_points = data.get("main_points", [])
        focus = self._derive_focus_from_query(query)

        if not main_points:
            return (
                f"I took a look at the references about {focus or 'this topic'} and will "
                "share more detail as soon as I spot something concrete."
            )

        headline = self._ensure_sentence(main_points[0])

        if query_type == "supplier_overview":
            preface = (
                f"Thanks for checking in about {focus or 'your supplier landscape'}. "
                "Here's the top-line view from the sourcing notes:"
            )
        elif query_type == "financial_analysis":
            preface = (
                f"Looking at the spend position for {focus or 'this area'}, "
                "here's what immediately stands out:"
            )
        elif query_type == "comparison":
            preface = (
                f"To help you weigh {focus or 'these options'}, "
                "here's the headline insight I found:"
            )
        elif query_type == "policy_lookup":
            preface = (
                f"You're asking about {focus or 'this policy area'}. "
                "The policy guidance boils down to this:"
            )
        else:
            preface = (
                f"I've reviewed the reference notes on {focus or 'this topic'}. "
                "Here's the quick take-away:"
            )

        return f"{preface} {headline}"

    def _derive_focus_from_query(self, query: str) -> str:
        query = query.strip().rstrip("?!. ")
        if not query:
            return "the topic"

        lowered = query.lower()
        prefixes = ["what is", "what's", "tell me about", "how do", "how does", "can i", "could i"]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                cleaned = query[len(prefix) :].strip()
                return cleaned if cleaned else "the topic"

        return query

    def _ensure_sentence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            return f"{cleaned}."
        return cleaned

    def _generate_supplier_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        supplier_data = self._extract_supplier_info(data.get("main_points", []), docs)

        if supplier_data:
            primary_lines = ["Here are the suppliers that matter most right now:"]
            for supplier in supplier_data[:3]:
                bullet_lines = [f"- **{supplier['name']}**"]
                if supplier.get("description"):
                    bullet_lines[-1] += f": {supplier['description']}"
                if supplier.get("metrics"):
                    for metric in supplier["metrics"]:
                        bullet_lines.append(f"  - {metric}")
                if supplier.get("source_id"):
                    bullet_lines.append(f"  - Source: {supplier['source_id']}")
                primary_lines.extend(bullet_lines)
            sections.append("\n".join(primary_lines).strip())

        if len(supplier_data) > 1:
            insights = ["What this mix tells us:"]
            if any(s.get("coverage", 0) >= 0.7 for s in supplier_data):
                insights.append(
                    "- Coverage is concentrated with key partners, so keep an eye on dependency risk."
                )
            if len(supplier_data) > 5:
                insights.append(
                    f"- With about {len(supplier_data)} active suppliers, you have diversification headroom."
                )
            default_highlight = data.get("main_points", [])[:2]
            for point in default_highlight:
                insights.append(f"- {self._ensure_sentence(point)}")
            sections.append("\n".join(insights).strip())

        return sections

    def _generate_financial_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        financial_data = self._extract_financial_info(data.get("main_points", []), docs)

        if financial_data.get("totals"):
            summary_lines = ["Key figures that anchor the conversation:"]
            for key, value in financial_data["totals"].items():
                summary_lines.append(f"- **{key}:** {value}")
            source = data.get("source_ids", [None])[0]
            if source:
                summary_lines.append(f"- Source: {source}")
            sections.append("\n".join(summary_lines).strip())

        breakdown_rows = financial_data.get("breakdown") or []
        if breakdown_rows:
            table_lines = [
                "Spending split by category:",
                "| Category | Amount | % of Total |",
                "|----------|--------|------------|",
            ]
            for row in breakdown_rows[:5]:
                table_lines.append(
                    f"| {row['category']} | {row['amount']} | {row['percentage']}% |"
                )
            sections.append("\n".join(table_lines))

        trends = financial_data.get("trends") or []
        if trends:
            trend_lines = ["Patterns worth noting:"]
            for trend in trends:
                trend_lines.append(f"- {self._ensure_sentence(trend)}")
            sections.append("\n".join(trend_lines).strip())

        return sections

    def _generate_comparison_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        comparison_data = self._extract_comparison_data(data.get("main_points", []), docs)

        if len(comparison_data) >= 2:
            header_names = " | ".join([c["name"] for c in comparison_data[:3]])
            table_lines = [
                "Here's how the options line up:",
                f"| Aspect | {header_names} |",
                "|--------|" + "|".join(["--------" for _ in comparison_data[:3]]) + "|",
            ]
            for metric in ["coverage", "spend", "contracts", "performance"]:
                values = [c.get(metric, "N/A") for c in comparison_data[:3]]
                if any(value != "N/A" for value in values):
                    table_lines.append(
                        f"| {metric.title()} | " + " | ".join(str(v) for v in values) + " |"
                    )
            sections.append("\n".join(table_lines))

            analysis_lines = ["What to take from this:"]
            for entity in comparison_data[:3]:
                analysis_lines.append(f"**{entity['name']}**")
                strengths = entity.get("strengths") or []
                if strengths:
                    analysis_lines.append("- Strengths:")
                    analysis_lines.extend(f"  - {self._ensure_sentence(item)}" for item in strengths)
                weaknesses = entity.get("weaknesses") or []
                if weaknesses:
                    analysis_lines.append("- Watch-outs:")
                    analysis_lines.extend(f"  - {self._ensure_sentence(item)}" for item in weaknesses)
                analysis_lines.append("")
            sections.append("\n".join(analysis_lines).strip())

        return sections

    def _derive_policy_name(self, query: str, topic_entry: TopicRecord) -> str:
        focus = self._derive_focus_from_query(query)
        candidate_focus = focus.strip() if focus else ""
        topic_candidate = topic_entry.topic.strip() if topic_entry.topic else ""

        for candidate in [candidate_focus, topic_candidate]:
            if not candidate:
                continue
            formatted = self._format_policy_title(candidate)
            if "policy" in formatted.lower():
                return formatted

        for candidate in [candidate_focus, topic_candidate]:
            if not candidate:
                continue
            formatted = self._format_policy_title(f"{candidate} policy")
            if formatted:
                return formatted

        return "Policy Guidance"

    def _format_policy_title(self, text: str) -> str:
        cleaned = text.strip().rstrip("?.")
        if not cleaned:
            return "Policy Guidance"
        words = cleaned.split()
        result_words: List[str] = []
        for index, word in enumerate(words):
            lower = word.lower()
            if index == 0 or lower not in {"and", "or", "of", "the"}:
                result_words.append(lower.capitalize())
            else:
                result_words.append(lower)
        if result_words and result_words[-1].lower() != "policy":
            result_words.append("Policy")
        return " ".join(result_words)

    def _extract_policy_payload(
        self,
        *,
        policy_name: str,
        topic_entry: TopicRecord,
        focus_answer: str,
        depth_mode: str,
    ) -> Dict[str, Any]:
        categories: Dict[str, List[str]] = defaultdict(list)

        for qa in topic_entry.qas:
            question_lower = qa.question.lower()
            sentences = self._split_sentences(qa.answer)
            for sentence in sentences:
                clause = self._clean_policy_clause(sentence)
                lowered = clause.lower()
                if re.search(r"\b(must|shall|need to|ensure|submit|retain|provide|keep)\b", lowered):
                    categories["requirements"].append(clause)
                if re.search(
                    r"\b(non-claimable|not allowed|cannot|can't|prohibit|forbidden|declined)\b",
                    lowered,
                ) or "non-claimable" in lowered:
                    categories["restrictions"].append(clause)
                if re.search(r"[£$€]\s*[\d,.]+", clause) or re.search(
                    r"\b(limit|cap|threshold|per person|per day|per month)\b",
                    lowered,
                ):
                    categories["spending_limits"].append(clause)
                if re.search(r"approval|approve|authoris|manager|finance", lowered):
                    categories["approval_process"].append(clause)
                if re.search(r"for example|such as|e.g.|include", lowered):
                    categories["examples"].append(clause)
                if re.search(r"unless|exception|exemption|waiver", lowered):
                    categories["exceptions"].append(clause)

            if "exception" in question_lower or "waiver" in question_lower:
                categories["exceptions"].extend(sentences)
            if "example" in question_lower:
                categories["examples"].extend(sentences)
            if any(keyword in question_lower for keyword in ["limit", "cap", "threshold"]):
                categories["spending_limits"].extend(sentences)
            if "approval" in question_lower or "pre-approv" in question_lower:
                categories["approval_process"].extend(sentences)
            if any(keyword in question_lower for keyword in ["must", "how do i comply"]):
                categories["requirements"].extend(sentences)

        overview_sentences = self._split_sentences(focus_answer)
        overview = overview_sentences[0] if overview_sentences else focus_answer

        payload = {
            "policy_name": policy_name,
            "overview": self._ensure_sentence(self._clean_policy_clause(overview)),
            "requirements": self._unique_ordered(categories.get("requirements", [])),
            "restrictions": self._unique_ordered(categories.get("restrictions", [])),
            "spending_limits": self._unique_ordered(categories.get("spending_limits", [])),
            "approval_process": self._unique_ordered(categories.get("approval_process", [])),
            "examples": self._unique_ordered(categories.get("examples", [])),
            "exceptions": self._unique_ordered(categories.get("exceptions", [])),
        }

        if depth_mode == "expanded":
            # Allow more contextual statements when expanding.
            for key in ["requirements", "restrictions", "spending_limits", "approval_process"]:
                values = payload[key]
                if not values and payload["overview"]:
                    values.append(payload["overview"])
                payload[key] = values

        return payload

    def _render_policy_response(
        self, payload: Dict[str, Any], depth_mode: str
    ) -> str:
        policy_name = self._format_policy_title(payload.get("policy_name", "Policy Guidance"))
        overview = payload.get("overview", "")
        overview_text = self._ensure_sentence(self._clean_policy_clause(overview)) if overview else ""

        lines: List[str] = [f"## {policy_name}"]
        if overview_text:
            lines.append(overview_text)

        sections = [
            ("What You Must Do", "requirements"),
            ("What's Prohibited", "restrictions"),
            ("Spending Limits", "spending_limits"),
            ("Approval Requirements", "approval_process"),
            ("Examples", "examples"),
            ("Exceptions", "exceptions"),
        ]

        for title, key in sections:
            bullets = self._policy_section_bullets(payload, key, depth_mode)
            if not bullets:
                continue
            lines.append("")
            lines.append(title)
            lines.extend(f"- {bullet}" for bullet in bullets)

        return "\n".join(lines)

    def _policy_section_bullets(
        self, payload: Dict[str, Any], key: str, depth_mode: str
    ) -> List[str]:
        sentences = list(payload.get(key, []))
        target = 5 if depth_mode == "expanded" else 3
        target = min(target, 6)

        bullets: List[str] = []
        for sentence in sentences:
            bullets.extend(self._expand_policy_sentence(sentence, key))

        if key == "requirements" and len(bullets) < target:
            derived = self._derive_requirement_from_restriction(payload.get("restrictions", []))
            for item in derived:
                if item not in bullets:
                    bullets.append(item)
                    if len(bullets) >= target:
                        break

        if key == "spending_limits" and len(bullets) < target:
            for restriction in payload.get("restrictions", []):
                amounts = re.findall(r"[£$€]\s*[\d,.]+", restriction)
                if not amounts:
                    continue
                bullet = self._ensure_sentence(
                    f"Treat {amounts[0]} as the ceiling unless Finance approves more."
                )
                if bullet not in bullets:
                    bullets.append(bullet)
                if len(bullets) >= target:
                    break

        if key == "approval_process" and len(bullets) < target:
            for requirement in payload.get("requirements", []):
                clause = requirement.lower()
                if "approve" in clause or "sign" in clause:
                    bullet = self._ensure_sentence(self._clean_policy_clause(requirement))
                    if bullet not in bullets:
                        bullets.append(bullet)
                if len(bullets) >= target:
                    break

        fallback_map = {
            "requirements": "Follow this policy before and after each purchase to stay compliant.",
            "restrictions": "Treat anything not explicitly allowed in the policy as prohibited.",
            "spending_limits": "Check the policy for specific monetary caps before spending.",
            "approval_process": "Capture written approval in advance for any exception.",
            "examples": "Model your claim on the compliant scenarios described in the policy.",
            "exceptions": "Escalate unusual circumstances to Finance for documented exceptions.",
        }

        bullets = [self._ensure_sentence(item) for item in self._unique_ordered(bullets)]

        while len(bullets) < target:
            fallback = fallback_map.get(key)
            if not fallback or fallback in bullets:
                break
            bullets.append(self._ensure_sentence(fallback))

        return bullets[:target]

    def _expand_policy_sentence(self, sentence: str, section: str) -> List[str]:
        clause = self._clean_policy_clause(sentence)
        lowered = clause.lower()
        bullets: List[str] = []

        if section == "requirements":
            if any(keyword in lowered for keyword in ["pre-approv", "approval"]):
                bullets.append("Obtain pre-approval before committing the spend this policy covers.")
            if any(keyword in lowered for keyword in ["submit", "claim", "provide", "retain"]):
                bullets.append("Submit only eligible, well-documented expenses so they are accepted.")
            if "policy" in lowered or "intranet" in lowered:
                bullets.append("Check the official policy on the intranet before you spend or submit claims.")
            if not bullets:
                bullets.append(clause)
        elif section == "restrictions":
            if "include" in lowered:
                parts = re.split(r"include[s]?", clause, flags=re.IGNORECASE)
                if len(parts) > 1:
                    items = re.split(r",| and | or ", parts[1])
                    for item in items:
                        cleaned_item = item.strip(" .")
                        if cleaned_item:
                            cleaned_item = re.sub(r"^the ", "", cleaned_item, flags=re.IGNORECASE)
                            bullets.append(f"Do not claim {cleaned_item}.")
            if any(keyword in lowered for keyword in ["declined", "breach", "disciplinary"]):
                bullets.append("Expect non-claimable expenses to be declined, logged, and escalated if repeated.")
            if not bullets:
                bullets.append(clause)
        elif section == "spending_limits":
            amounts = re.findall(r"[£$€]\s*[\d,.]+", clause)
            if amounts:
                for amount in amounts:
                    bullets.append(f"Stay within the {amount} limit unless you have written approval.")
            if "per person" in lowered:
                bullets.append("Keep client entertainment within the per-person threshold stated in the policy.")
            if not amounts and ("limit" in lowered or "threshold" in lowered):
                bullets.append(clause)
        elif section == "approval_process":
            if any(keyword in lowered for keyword in ["approval", "approve", "manager", "finance"]):
                bullets.append("Capture manager or Finance approval before using the card for unusual spend.")
            if "pre-approv" in lowered:
                bullets.append("Record pre-approval details with the expense submission.")
            if not bullets:
                bullets.append(clause)
        elif section == "examples":
            bullets.append(f"Example: {clause}")
        elif section == "exceptions":
            if "unless" in lowered:
                exception_text = clause.split("unless", 1)[1].strip()
                bullets.append(
                    f"Exception: Allowed when {exception_text}"
                    if exception_text
                    else f"Exception: {clause}"
                )
            else:
                bullets.append(f"Exception: {clause}")

        return bullets

    def _derive_requirement_from_restriction(
        self, restrictions: Sequence[str]
    ) -> List[str]:
        suggestions: List[str] = []
        for sentence in restrictions:
            clause = self._clean_policy_clause(sentence)
            lowered = clause.lower()
            if any(keyword in lowered for keyword in ["pre-approv", "unless"]):
                suggestions.append(
                    "Secure pre-approval before submitting anything that normally sits on the restricted list."
                )
            if any(keyword in lowered for keyword in ["declined", "disciplinary", "breach"]):
                suggestions.append(
                    "Validate each expense against the policy so it isn't declined or escalated."
                )
        return [self._ensure_sentence(item) for item in self._unique_ordered(suggestions)]

    def _clean_policy_clause(self, sentence: str) -> str:
        text = sentence.strip()
        text = re.sub(r"^[Yy]es,?\s*", "", text)
        text = re.sub(r"^[Nn]o,?\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _unique_ordered(self, items: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in items:
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cleaned)
        return ordered

    def _generate_exploratory_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        points = data.get("main_points", [])
        if points:
            sections.append(
                "Here's the context that frames this question: "
                + self._ensure_sentence(points[0])
            )
        if len(points) > 1:
            findings = ["Highlights worth noting:"]
            findings.extend(f"- {self._ensure_sentence(point)}" for point in points[1:4])
            sections.append("\n".join(findings).strip())
        numbers = data.get("numbers", [])
        if numbers:
            detail_lines = ["Figures that back this up:"]
            for number in numbers[:5]:
                detail_lines.append(f"- {number}")
            sections.append("\n".join(detail_lines).strip())
        return sections

    def _generate_closing_section(
        self, query: str, data: Dict[str, Any], query_type: str
    ) -> str:
        if query_type == "simple_lookup":
            return "Let me know if you'd like to dig any deeper."  # short and friendly

        if query_type == "supplier_overview":
            high_coverage = any(
                "%" in point and any(token in point for token in ["80", "85", "90"])
                for point in data.get("main_points", [])
            )
            if high_coverage:
                closing = (
                    "It looks like the supplier base is well-covered—worth reviewing leverage while keeping resilience in mind."
                )
            else:
                closing = (
                    "There's room to tighten the supplier mix for resilience, so consider follow-up reviews on coverage."
                )
        elif query_type == "financial_analysis":
            growth = any("increase" in point.lower() for point in data.get("main_points", []))
            if growth:
                closing = (
                    "Spend is trending upward, so a quick variance check would help keep savings on track."
                )
            else:
                closing = (
                    "With spend running steady, you have space to reassess supplier performance and pricing plays."
                )
        elif query_type == "comparison":
            closing = (
                "Each option leans into different strengths—pick the one that fits your priority, whether that's cost control, reliability, or innovation."
            )
        elif query_type == "policy_lookup":
            closing = (
                "Keep these guardrails close by for approvals, and I'm happy to pull up any supporting examples if you need them."
            )
        else:
            closing = (
                "Use these points to steer the next conversation, and just shout if you want deeper analysis on any thread."
            )

        return closing

    def _generate_simple_response(
        self, query: str, data: Dict[str, Any], docs: List[Any]
    ) -> str:
        main_points = data.get("main_points", [])
        if not main_points:
            return (
                "I checked the reference notes but couldn't find a concrete answer yet—"
                "let me know if you want me to broaden the search."
            )

        focus = self._derive_focus_from_query(query)
        sentences = [self._ensure_sentence(main_points[0])]
        if len(main_points) > 1:
            sentences.append(self._ensure_sentence(main_points[1]))
        response = (
            f"On {focus or 'this point'}, here's what the notes say: " + " ".join(sentences)
        )
        source_ids = data.get("source_ids", [])
        if source_ids:
            response += f" (Source: {source_ids[0]})"
        return response

    def _extract_supplier_info(
        self, main_points: List[str], docs: List[Any]
    ) -> List[Dict[str, Any]]:
        suppliers: List[Dict[str, Any]] = []
        seen: Dict[str, Dict[str, Any]] = {}

        for point in main_points:
            names = re.findall(r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*\b", point)
            for name in names:
                if len(name) < 3 or name.lower() in {"the", "and", "for"}:
                    continue
                entry = seen.setdefault(
                    name,
                    {
                        "name": name,
                        "description": point,
                        "metrics": [],
                        "source_id": None,
                    },
                )
                percentages = re.findall(r"\b\d{1,3}%\b", point)
                entry.setdefault("coverage", 0)
                if percentages:
                    try:
                        coverage_value = max(int(p.rstrip("%")) for p in percentages) / 100.0
                        entry["coverage"] = max(entry.get("coverage", 0), coverage_value)
                        entry["metrics"].append(f"Coverage: {percentages[0]}")
                    except ValueError:
                        pass
                currency = re.findall(r"[£$]\s*[\d,.]+", point)
                if currency:
                    entry["metrics"].append(f"Spend: {currency[0]}")

        for doc in docs:
            payload = getattr(doc, "payload", {}) or {}
            record_id = payload.get("record_id")
            summary = payload.get("summary", "")
            names = re.findall(r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*\b", summary)
            for name in names:
                entry = seen.setdefault(
                    name,
                    {
                        "name": name,
                        "description": summary,
                        "metrics": [],
                        "source_id": record_id,
                    },
                )
                if record_id and not entry.get("source_id"):
                    entry["source_id"] = record_id

        suppliers.extend(seen.values())
        suppliers.sort(key=lambda item: (item.get("coverage", 0), len(item.get("metrics", []))), reverse=True)
        return suppliers

    def _extract_financial_info(
        self, main_points: List[str], docs: List[Any]
    ) -> Dict[str, Any]:
        totals: Dict[str, str] = {}
        breakdown: List[Dict[str, Any]] = []
        trends: List[str] = []

        for point in main_points:
            amounts = re.findall(r"[£$]\s*[\d,.]+", point)
            if amounts:
                if "total" in point.lower() or "overall" in point.lower():
                    totals.setdefault("Total Spend", amounts[0])
                elif "savings" in point.lower():
                    totals.setdefault("Savings", amounts[0])
                elif "quarter" in point.lower() or "month" in point.lower():
                    totals.setdefault("Period Spend", amounts[0])
            if any(keyword in point.lower() for keyword in ["increase", "decrease", "growth", "decline"]):
                trends.append(point)

        for doc in docs:
            payload = getattr(doc, "payload", {}) or {}
            summary = payload.get("summary", "")
            matches = re.findall(r"([A-Z][A-Za-z\s]+):\s*([£$]\s*[\d,.]+)", summary)
            for category, amount in matches:
                breakdown.append(
                    {
                        "category": category.strip(),
                        "amount": amount.strip(),
                        "percentage": 0,
                    }
                )

        return {"totals": totals, "breakdown": breakdown, "trends": trends}

    def _extract_comparison_data(
        self, main_points: List[str], docs: List[Any]
    ) -> List[Dict[str, Any]]:
        entities: Dict[str, Dict[str, Any]] = {}

        for point in main_points:
            names = re.findall(r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*\b", point)
            for name in names:
                if len(name) < 3 or name.lower() in {"the", "and", "for"}:
                    continue
                entity = entities.setdefault(
                    name,
                    {"name": name, "strengths": [], "weaknesses": []},
                )
                if any(token in point.lower() for token in ["strong", "lead", "preferred", "faster"]):
                    entity["strengths"].append(point)
                if any(token in point.lower() for token in ["risk", "delay", "gap", "limited"]):
                    entity["weaknesses"].append(point)
                percents = re.findall(r"\b\d{1,3}%\b", point)
                if percents:
                    entity["coverage"] = percents[0]
                amounts = re.findall(r"[£$]\s*[\d,.]+", point)
                if amounts:
                    entity["spend"] = amounts[0]

        for doc in docs:
            payload = getattr(doc, "payload", {}) or {}
            summary = payload.get("summary", "")
            names = re.findall(r"\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*\b", summary)
            for name in names:
                entities.setdefault(name, {"name": name, "strengths": [], "weaknesses": []})

        return list(entities.values())

    def _generate_contextual_followups(
        self,
        query: str,
        query_type: str,
        data: Dict[str, Any],
        depth_mode: str = "standard",
    ) -> List[str]:
        if query_type == "supplier_overview":
            return [
                "Would you like to see contract details for any specific supplier?",
                "Should I analyse spending patterns across these suppliers?",
                "Want to explore consolidation opportunities?",
            ]
        if query_type == "financial_analysis":
            return [
                "Would you like a breakdown by supplier or category?",
                "Should I compare this to previous periods?",
                "Want to identify cost-saving opportunities?",
            ]
        if query_type == "comparison":
            return [
                "Would you like me to add more suppliers to this comparison?",
                "Should I analyse performance metrics for these suppliers?",
                "Want to see historical trends for comparison?",
            ]
        if query_type == "policy_lookup":
            prompts = [
                "Need a checklist you can share with your team?",
                "Should I pull the related forms or intranet links?",
                "Want a quick summary of exceptions versus standard rules?",
            ]
            if depth_mode == "expanded":
                prompts.append("Would examples for specific scenarios (travel, client meetings, etc.) help?")
            return prompts
        return [
            "Would you like more details on any specific aspect?",
            "Should I search for related procurement policies?",
            "Want to see how this compares across suppliers?",
        ]
