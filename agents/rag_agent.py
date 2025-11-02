import json
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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

        feedback_detected = False
        feedback_id: Optional[int] = None
        feedback_sentiment: Optional[FeedbackSentiment] = None
        feedback_confidence = 0.0
        acknowledgment_text: Optional[str] = None
        ack_prefix: Optional[str] = None

        previous_user = self._last_interaction.get("user_id")
        if previous_user and previous_user == user_id:
            feedback_sentiment, feedback_confidence = self.feedback_service.detect_feedback(query)
            if (
                feedback_sentiment != FeedbackSentiment.NEUTRAL
                and feedback_confidence > 0.3
            ):
                feedback_detected = True
                acknowledgment_text = self.feedback_service.generate_acknowledgment(
                    feedback_sentiment, query
                )
                feedback_id = self.feedback_service.store_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    query=self._last_interaction.get("query", ""),
                    response=self._last_interaction.get("response", ""),
                    feedback_message=query,
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
                        "question": self._last_interaction.get("query", ""),
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

        extracted = self._extract_answer_signals(qa_entry.answer, record_id)
        query_type = self._classify_query_type(query)
        plan = self._plan_response_structure(query, query_type, extracted)
        structured_answer = self._generate_structured_response(
            query, extracted, docs, plan
        )
        if ack_prefix:
            structured_answer = f"{ack_prefix}\n\n{structured_answer}" if structured_answer else ack_prefix
        followups = self._generate_contextual_followups(query, query_type, extracted)

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
        }

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
            "query": query,
            "response": structured_answer,
            "doc_ids": [record_id],
            "topic": topic_entry.topic,
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

    # ------------------------------------------------------------------
    # Structured response planning and generation
    # ------------------------------------------------------------------
    def _extract_answer_signals(self, answer: str, record_id: str) -> Dict[str, Any]:
        sentences = self._split_sentences(answer)
        main_points = sentences[:6] if sentences else [answer.strip()]

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
    ) -> str:
        query_type = plan["type"]
        opening = self._generate_opening_section(query, extracted_data, query_type)
        sections: List[str] = [opening]

        if query_type == "supplier_overview":
            sections.extend(self._generate_supplier_sections(extracted_data, retrieved_docs))
        elif query_type == "financial_analysis":
            sections.extend(self._generate_financial_sections(extracted_data, retrieved_docs))
        elif query_type == "comparison":
            sections.extend(self._generate_comparison_sections(extracted_data, retrieved_docs))
        elif query_type == "policy_lookup":
            sections.extend(self._generate_policy_sections(extracted_data, retrieved_docs))
        elif query_type == "simple_lookup":
            return self._generate_simple_response(extracted_data, retrieved_docs)
        else:
            sections.extend(self._generate_exploratory_sections(extracted_data, retrieved_docs))

        closing = self._generate_closing_section(query, extracted_data, query_type)
        if closing:
            sections.append(closing)

        return "\n\n".join([segment for segment in sections if segment.strip()])

    def _generate_opening_section(
        self, query: str, data: Dict[str, Any], query_type: str
    ) -> str:
        main_points = data.get("main_points", [])
        if not main_points:
            return "Let me look into this for you."

        if query_type == "supplier_overview":
            header = "## Your Supplier Landscape\n\n"
            context = "Here's what your procurement data shows about your supplier relationships.\n"
        elif query_type == "financial_analysis":
            header = "## Financial Overview\n\n"
            context = "I've pulled together the key financial data from your procurement records.\n"
        elif query_type == "comparison":
            header = "## Comparative Analysis\n\n"
            context = "Let me break down the key differences for you.\n"
        elif query_type == "policy_lookup":
            header = "## Policy Reference\n\n"
            context = "Here's what the procurement policies say about this.\n"
        else:
            header = "## Overview\n\n"
            context = "Based on the available data, here's what I found.\n"

        return header + context

    def _generate_supplier_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        supplier_data = self._extract_supplier_info(data.get("main_points", []), docs)

        if supplier_data:
            primary_lines = ["### Primary Suppliers\n"]
            for supplier in supplier_data[:3]:
                primary_lines.append(f"**{supplier['name']}**")
                if supplier.get("description"):
                    primary_lines.append(supplier["description"])
                if supplier.get("metrics"):
                    primary_lines.append("\n**Key metrics:**")
                    for metric in supplier["metrics"]:
                        primary_lines.append(f"- {metric}")
                if supplier.get("source_id"):
                    primary_lines.append(f"[Document ID: {supplier['source_id']}]\n")
            sections.append("\n".join(primary_lines).strip())

        if len(supplier_data) > 1:
            insights = ["### Key Insights\n"]
            if any(s.get("coverage", 0) >= 0.7 for s in supplier_data):
                insights.append(
                    "- You maintain high coverage with strategic suppliers, signalling mature procurement controls"
                )
            if len(supplier_data) > 5:
                insights.append(
                    f"- Supplier base spans {len(supplier_data)} relationships, offering diversification"
                )
            default_highlight = data.get("main_points", [])[:2]
            for point in default_highlight:
                insights.append(f"- {point}")
            sections.append("\n".join(insights).strip())

        return sections

    def _generate_financial_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        financial_data = self._extract_financial_info(data.get("main_points", []), docs)

        if financial_data.get("totals"):
            summary_lines = ["### Key Figures\n"]
            for key, value in financial_data["totals"].items():
                summary_lines.append(f"**{key}:** {value}")
            source = data.get("source_ids", [None])[0]
            if source:
                summary_lines.append(f"\n[Document ID: {source}]")
            sections.append("\n".join(summary_lines).strip())

        breakdown_rows = financial_data.get("breakdown") or []
        if breakdown_rows:
            table_lines = [
                "### Spending Breakdown\n",
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
            trend_lines = ["### Trends & Patterns\n"]
            for trend in trends:
                trend_lines.append(f"- {trend}")
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
                "### Side-by-Side Comparison\n",
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

            analysis_lines = ["### Analysis\n"]
            for entity in comparison_data[:3]:
                analysis_lines.append(f"**{entity['name']}**")
                strengths = entity.get("strengths") or []
                if strengths:
                    analysis_lines.append("Strengths:")
                    analysis_lines.extend(f"- {item}" for item in strengths)
                weaknesses = entity.get("weaknesses") or []
                if weaknesses:
                    analysis_lines.append("Areas to consider:")
                    analysis_lines.extend(f"- {item}" for item in weaknesses)
                analysis_lines.append("")
            sections.append("\n".join(analysis_lines).strip())

        return sections

    def _generate_policy_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        points = data.get("main_points", [])
        if points:
            sections.append("### Policy Summary\n\n" + points[0])
        if len(points) > 1:
            requirement_lines = ["### Key Requirements\n"]
            requirement_lines.extend(f"- {point}" for point in points[1:4])
            sections.append("\n".join(requirement_lines).strip())
        related_entities = data.get("entities", [])[:3]
        if related_entities:
            sections.append(
                "### Related Policies\n\n" + "\n".join(f"- {entity}" for entity in related_entities)
            )
        return sections

    def _generate_exploratory_sections(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> List[str]:
        sections: List[str] = []
        points = data.get("main_points", [])
        if points:
            sections.append("### Context\n\n" + points[0])
        if len(points) > 1:
            findings = ["### Findings\n"]
            findings.extend(f"- {point}" for point in points[1:4])
            sections.append("\n".join(findings).strip())
        numbers = data.get("numbers", [])
        if numbers:
            detail_lines = ["### Supporting Details\n"]
            for number in numbers[:5]:
                detail_lines.append(f"- {number}")
            sections.append("\n".join(detail_lines).strip())
        return sections

    def _generate_closing_section(
        self, query: str, data: Dict[str, Any], query_type: str
    ) -> str:
        if query_type == "simple_lookup":
            return ""

        closing = "### What This Means for You\n\n"

        if query_type == "supplier_overview":
            high_coverage = any(
                "%" in point and any(token in point for token in ["80", "85", "90"])
                for point in data.get("main_points", [])
            )
            if high_coverage:
                closing += (
                    "This supplier portfolio shows strong established relationships. "
                    "Consider reviewing commercial terms to capture leverage while managing dependency risk."
                )
            else:
                closing += (
                    "There may be opportunities to strengthen or consolidate supplier coverage for better resilience."
                )
        elif query_type == "financial_analysis":
            growth = any("increase" in point.lower() for point in data.get("main_points", []))
            if growth:
                closing += (
                    "Spending momentum is trending upward. Prioritise variance reviews to protect savings targets."
                )
            else:
                closing += (
                    "Spend levels look stable—use this window to reassess supplier performance and negotiation plays."
                )
        elif query_type == "comparison":
            closing += (
                "Each option brings distinct strengths. Align the choice with your priority—cost discipline, reliability, or innovation."
            )
        elif query_type == "policy_lookup":
            closing += (
                "Keep these rules on hand for approvals and ensure stakeholders follow the documented thresholds."
            )
        else:
            closing += (
                "Use these findings to guide next discussions and highlight any gaps that need deeper analysis."
            )

        return closing

    def _generate_simple_response(
        self, data: Dict[str, Any], docs: List[Any]
    ) -> str:
        main_points = data.get("main_points", [])
        if not main_points:
            return "I couldn't find specific information about this in the available documents."

        response = main_points[0]
        if len(main_points) > 1:
            response += f" {main_points[1]}"
        source_ids = data.get("source_ids", [])
        if source_ids:
            response += f" [Document ID: {source_ids[0]}]"
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
        self, query: str, query_type: str, data: Dict[str, Any]
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
        return [
            "Would you like more details on any specific aspect?",
            "Should I search for related procurement policies?",
            "Want to see how this compares across suppliers?",
        ]
