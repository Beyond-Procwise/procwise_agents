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
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


_MAX_ATOM_LIST_ITEMS = 6
_BANNED_TOKENS: Tuple[str, ...] = (
    "Thanks for flagging this",
    "[redacted]",
    "document_id",
    "static_policy",
)


ATOM_EXTRACTION_SYSTEM = """Return concise procurement guidance as JSON with exactly these keys: summary, scope, rules, exclusions, policies, next_steps, followup.
- summary: one or two short sentences that answer the question directly.
- scope, rules, exclusions, next_steps: arrays of up to two short strings (≤22 words) each; omit empty entries.
- policies: array of up to two [label, timing, note] triples; keep each value short and human readable.
- followup: single actionable sentence describing the most logical next artefact or question; no pleasantries or rhetorical questions.
Never emit HTML, markdown, bullet markers, or chit-chat. Stick to plain JSON.
Example:
{
  "summary": "Late-night rideshare fares linked to client meetings are reimbursable when you document the meeting and stay within the £80 limit.",
  "scope": ["Applies after client meetings when public transport is unavailable."],
  "rules": ["Submit the agenda or notes with the claim to evidence business purpose."],
  "exclusions": ["Do not expense personal detours or premium vehicle classes."],
  "policies": [["Standard limit", "Up to £80", "Manager approval required if you exceed the limit."]],
  "next_steps": ["Attach the receipt showing pickup and drop-off times before claiming."],
  "followup": "Share the hospitality policy if you also need alcohol guidance."
}
Always respond with valid JSON containing exactly the keys summary, scope, rules, exclusions, policies, next_steps, and followup."""


def compose_html_answer(
    summary: str,
    scope: Optional[List[str]] | None = None,
    rules: Optional[List[str]] | None = None,
    exclusions: Optional[List[str]] | None = None,
    policies: Optional[List[Tuple[str, str, str]]] | None = None,
    next_steps: Optional[List[str]] | None = None,
    followup: Optional[str] | None = None,
) -> str:
    summary_text = (summary or "No summary available.").strip() or "No summary available."

    def _trim(items: Optional[Sequence[Any]]) -> List[str]:
        if not items:
            return []
        result: List[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                result.append(text)
            if len(result) >= _MAX_ATOM_LIST_ITEMS:
                break
        return result

    def _opening(text: str) -> str:
        cleaned = text.strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned) if cleaned else []
        snippet = " ".join(sentences[:2]).strip() or cleaned or "Here is the latest documented guidance for this request."
        snippet = snippet.rstrip(" .") + "."
        if snippet.lower().startswith(("hi", "hello", "hey")):
            return snippet
        return f"Hi—{snippet}"

    def _collect_points() -> List[str]:
        points: List[str] = []
        sources = [_trim(scope), _trim(rules), _trim(exclusions), _trim(next_steps)]
        for source in sources:
            for entry in source:
                if entry not in points:
                    points.append(entry)
                if len(points) >= 3:
                    break
            if len(points) >= 3:
                break
        fallback = summary_text
        while len(points) < 3:
            candidate = fallback or "No additional guidance captured yet."
            if candidate not in points:
                points.append(candidate)
            else:
                points.append("No additional guidance captured yet.")
            fallback = None
        return points[:3]

    def _note_lines() -> List[str]:
        lines: List[str] = []
        trimmed_next = _trim(next_steps)
        if trimmed_next:
            joined = "; ".join(trimmed_next[:2])
            lines.append(f"Next steps: {joined}.")
        if policies:
            first = policies[0]
            if isinstance(first, (list, tuple)) and first:
                label = str(first[0]).strip()
                timing = str(first[1]).strip() if len(first) > 1 else ""
                remark = str(first[2]).strip() if len(first) > 2 else ""
                descriptor = ": ".join(bit for bit in [label, remark] if bit).strip(": ")
                if descriptor:
                    timing_note = f" (effective {timing})" if timing else ""
                    lines.append(f"{descriptor}{timing_note}.")
        follow_line = (followup or "Share another detail when you need deeper context.").strip()
        if follow_line and follow_line[-1] not in ".!?":
            follow_line = f"{follow_line}."
        if follow_line:
            lines.append(follow_line)
        return lines or ["Share another detail when you need deeper context."]

    opening_line = _opening(summary_text)
    key_points = _collect_points()
    note_lines = _note_lines()

    lines: List[str] = [opening_line, "", "Key Details"]
    for idx, point in enumerate(key_points, start=1):
        lines.append(f"- Point {idx} – {point}")
    lines.append("")
    lines.append("Additional Notes")
    lines.extend(note_lines)
    return "\n".join(line for line in lines if line).strip()


def sanitize_html(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    cleaned = re.sub(r"(?is)<\s*(script|iframe)[^>]*>.*?<\s*/\s*\1\s*>", "", answer)
    cleaned = re.sub(r"(?i)\s+on[a-z]+\s*=\s*\"[^\"]*\"", "", cleaned)
    cleaned = re.sub(r"(?i)\s+on[a-z]+\s*=\s*'[^']*'", "", cleaned)
    cleaned = re.sub(r"(?i)javascript:", "", cleaned)
    for token in _BANNED_TOKENS:
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
            "sections": [
                "summary",
                "allowed_conditions",
                "restricted_conditions",
                "compliance_notes",
                "next_steps",
            ],
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
        try:
            self.rag_service = RAGService(agent_nick)
        except Exception:
            logger.exception("Failed to initialise RAGService for RAGAgent", exc_info=True)
            self.rag_service = None
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
                        html_answer = self._build_html_answer(
                            query=classification_query,
                            raw_answer=acknowledgment_text or "Thank you for the feedback.",
                            extracted_data={
                                "main_points": [acknowledgment_text or "Appreciate the update."]
                            },
                            plan=None,
                        )
                        data = {
                            "question": self._last_interaction.get("query", classification_query),
                            "answer": html_answer,
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

        record_id = self._build_static_source_label(topic_entry, qa_entry)
        payload = {
            "record_id": record_id,
            "topic": topic_entry.topic,
            "question": qa_entry.question,
            "summary": qa_entry.answer,
            "document_type": "static_reference",
        }
        docs = [SimpleNamespace(payload=payload)]

        (
            dynamic_docs,
            primary_docs,
            policy_docs,
            dynamic_scope,
        ) = self._fetch_combined_context(classification_query, session_id)
        if dynamic_docs:
            docs.extend(dynamic_docs)

        source_labels = self._collect_source_labels(docs)

        if dynamic_scope:
            answer_scope = self._merge_context_texts(
                qa_entry.answer, dynamic_scope, depth_mode
            )
        else:
            answer_scope = self._compose_answer_scope(
                topic_entry, question_index, depth_mode
            )

        source_hint = source_labels[0] if source_labels else record_id
        extracted = self._extract_answer_signals(answer_scope, source_hint, depth_mode)
        if source_labels:
            extracted["source_ids"] = source_labels[:5]
        query_type = self._classify_query_type(classification_query)
        plan = self._plan_response_structure(classification_query, query_type, extracted)

        policy_payload: Optional[Dict[str, Any]] = None
        if query_type == "policy_lookup":
            policy_payload = self._extract_policy_payload(
                policy_name=self._derive_policy_name(classification_query, topic_entry),
                topic_entry=topic_entry,
                focus_answer=qa_entry.answer,
                depth_mode=depth_mode,
                policy_docs=policy_docs,
                primary_docs=primary_docs,
            )

        structured_answer = self._generate_structured_response(
            classification_query,
            extracted,
            docs,
            plan,
            depth_mode=depth_mode,
            policy_payload=policy_payload,
        )
        if not primary_docs and not policy_docs:
            fallback = self._compose_no_context_message(classification_query)
            structured_answer = (
                f"{fallback}\n\n{structured_answer}" if structured_answer else fallback
            )
        if ack_prefix:
            structured_answer = (
                f"{ack_prefix}\n\n{structured_answer}" if structured_answer else ack_prefix
            )
        followups = self._generate_contextual_followups(
            classification_query, query_type, extracted, depth_mode
        )
        html_answer = self._build_html_answer(
            query=classification_query,
            raw_answer=structured_answer,
            extracted_data=extracted,
            plan=plan,
            docs=docs,
            followups=followups,
        )

        response = {
            "question": qa_entry.question,
            "answer": html_answer,
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
            "2. Pull context from procurement records and policy guidance via Qdrant.\n"
            "3. Blend the findings into a structured, human-friendly summary."
        )

        context_snapshot = {
            "topic_similarity": float(topic_score),
            "question_similarity": float(question_score),
            "session_topic": topic_entry.topic,
            "response_depth": depth_mode,
            "procurement_hits": len(primary_docs),
            "policy_hits": len(policy_docs),
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
            "response": html_answer,
            "doc_ids": source_labels if source_labels else [record_id],
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

    def _build_static_source_label(
        self, topic_entry: TopicRecord, qa_entry: QARecord
    ) -> str:
        topic = (topic_entry.topic or "ProcWise reference").strip()
        question = (qa_entry.question or "").strip()
        if question:
            base = f"{topic} — {question}"
        else:
            base = topic
        return f"{base} (Reference Note)"

    def _fetch_combined_context(
        self, query: str, session_id: Optional[str]
    ) -> Tuple[List[SimpleNamespace], List[SimpleNamespace], List[SimpleNamespace], str]:
        if not self.rag_service:
            return [], [], [], ""

        try:
            collections = []
            primary_collection = getattr(
                self.rag_service, "primary_collection", None
            )
            policy_collection = getattr(
                self.rag_service, "static_policy_collection", None
            )
            if primary_collection:
                collections.append(primary_collection)
            if policy_collection and policy_collection != primary_collection:
                collections.append(policy_collection)

            hits = self.rag_service.search(
                query,
                top_k=8,
                session_id=session_id,
                collections=collections or None,
            )
        except Exception:
            logger.exception("RAG search failed inside RAGAgent", exc_info=True)
            return [], [], [], ""

        if not hits:
            return [], [], [], ""

        primary_docs: List[SimpleNamespace] = []
        policy_docs: List[SimpleNamespace] = []
        combined_segments: List[str] = []

        for index, hit in enumerate(hits):
            payload = dict(getattr(hit, "payload", {}) or {})
            nested_payload = payload.get("payload")
            if isinstance(nested_payload, dict):
                payload = {**nested_payload, **payload}
                payload.pop("payload", None)

            collection = payload.get("collection_name")
            if not isinstance(collection, str) or not collection.strip():
                collection = primary_collection or "procwise_document_embeddings"

            summary = self._extract_summary_from_payload(payload)
            title = self._extract_title_from_payload(payload)
            raw_document_type = payload.get("document_type") or payload.get("source_type")
            document_type = self._coerce_metadata_to_string(raw_document_type)
            source_label = self._build_source_label_from_details(
                title, document_type, collection
            )

            sanitized_payload = {
                "record_id": source_label,
                "summary": summary,
                "document_type": document_type,
                "collection_name": collection,
                "source_label": source_label,
            }

            for extra_key in (
                "highlights",
                "policy_references",
                "sections",
                "key_points",
            ):
                value = payload.get(extra_key)
                if value:
                    sanitized_payload[extra_key] = value

            doc = SimpleNamespace(
                payload=sanitized_payload,
                score=float(getattr(hit, "combined_score", getattr(hit, "score", 0.0))),
            )

            if summary:
                if collection == policy_collection:
                    combined_segments.append(f"Policy insight: {summary}")
                else:
                    combined_segments.append(f"Procurement context: {summary}")

            if collection == policy_collection:
                policy_docs.append(doc)
            else:
                primary_docs.append(doc)

        dynamic_docs = primary_docs + policy_docs
        combined_scope = "\n\n".join(combined_segments).strip()
        return dynamic_docs, primary_docs, policy_docs, combined_scope

    def _extract_summary_from_payload(self, payload: Dict[str, Any]) -> str:
        for key in ("text_summary", "summary", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        highlights = payload.get("highlights")
        if isinstance(highlights, (list, tuple)):
            for item in highlights:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return ""

    def _extract_title_from_payload(self, payload: Dict[str, Any]) -> str:
        for key in ("title", "document_title", "document_name", "source_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _coerce_metadata_to_string(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
            if value:
                return str(value[0])
            return ""
        if isinstance(value, dict):
            for key in ("name", "label", "type", "value"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    return item
            return json.dumps(value, ensure_ascii=False)
        if value is None:
            return ""
        return str(value)

    def _build_source_label_from_details(
        self, title: str, document_type: Optional[str], collection: str
    ) -> str:
        collection_lower = (collection or "").lower()
        if "policy" in collection_lower:
            collection_label = "Policy Library"
        else:
            collection_label = "Procurement Records"

        doc_label = (document_type or "").strip()
        if doc_label:
            doc_label = doc_label.title()

        base = title or doc_label or "Context"
        base = re.sub(r"\s+", " ", base).strip()
        if len(base) > 80:
            base = base[:77].rstrip() + "…"
        return f"{base} ({collection_label})"

    def _collect_source_labels(self, docs: List[SimpleNamespace]) -> List[str]:
        labels: List[str] = []
        for doc in docs:
            payload = getattr(doc, "payload", {}) or {}
            label = payload.get("source_label") or payload.get("record_id")
            if isinstance(label, str):
                cleaned = label.strip()
                if cleaned and cleaned not in labels:
                    labels.append(cleaned)
        return labels

    def _merge_context_texts(
        self, static_answer: str, dynamic_scope: str, depth_mode: str
    ) -> str:
        static_clean = (static_answer or "").strip()
        dynamic_clean = (dynamic_scope or "").strip()
        if not dynamic_clean:
            return static_clean
        if not static_clean:
            return dynamic_clean
        if depth_mode == "expanded":
            return f"{dynamic_clean}\n\nAdditional reference detail: {static_clean}"
        return f"{dynamic_clean}\n\nReference note: {static_clean}"

    def _compose_no_context_message(self, query: str) -> str:
        focus = self._derive_focus_from_query(query)
        focus_text = focus or "this request"
        return (
            "I checked both our procurement records and policy library but couldn't "
            f"find anything specific about {focus_text}. To keep things moving, please "
            "log the details in ProcWise or reach out to the procurement operations "
            "team so we can capture the right guidance."
        )

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
            "allowed_conditions": "Allowed Conditions",
            "restricted_conditions": "Restricted Conditions",
            "compliance_notes": "Compliance Notes",
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
            return self._render_policy_response(policy_payload, depth_mode, query)

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


    def _build_html_answer(
        self,
        *,
        query: str,
        raw_answer: Optional[str],
        extracted_data: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
        docs: Optional[Sequence[Any]] = None,
        followups: Optional[Sequence[str]] = None,
    ) -> str:
        """Render the final answer using JSON atoms and a controlled HTML composer."""

        extracted = extracted_data or {}
        plan = plan or {}
        raw_answer = raw_answer or ""
        heuristics = self._build_heuristic_atoms(
            query=query,
            raw_answer=raw_answer,
            extracted_data=extracted,
            plan=plan,
            followups=followups,
        )

        model_atoms: Dict[str, Any] = {}
        if self._should_invoke_atom_model():
            try:
                model_atoms = self._request_atoms_via_llm(
                    query=query,
                    raw_answer=raw_answer,
                    docs=list(docs or []),
                    heuristics=heuristics,
                )
            except Exception:  # pragma: no cover - defensive logging only
                logger.exception("Failed to extract HTML atoms via LLM; using heuristics")
                model_atoms = {}

        merged_atoms = self._merge_atom_payloads(heuristics, model_atoms)
        final_atoms = self._finalise_atoms(merged_atoms)

        html_answer = compose_html_answer(
            summary=final_atoms["summary"],
            scope=final_atoms["scope"],
            rules=final_atoms["rules"],
            exclusions=final_atoms["exclusions"],
            policies=final_atoms["policies"],
            next_steps=final_atoms["next_steps"],
            followup=final_atoms["followup"],
        )
        return sanitize_html(html_answer)

    def _should_invoke_atom_model(self) -> bool:
        disabled = getattr(self.agent_nick.settings, "disable_rag_atoms_llm", False)
        return not bool(disabled)

    def _build_heuristic_atoms(
        self,
        *,
        query: str,
        raw_answer: str,
        extracted_data: Dict[str, Any],
        plan: Dict[str, Any],
        followups: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        stripped_answer = self._strip_markdown(raw_answer)
        candidate_sentences = self._split_sentences(stripped_answer) if stripped_answer else []
        main_points: List[str] = [
            point
            for point in extracted_data.get("main_points", [])
            if isinstance(point, str) and point.strip()
        ]
        if not main_points and candidate_sentences:
            main_points = candidate_sentences[:_MAX_ATOM_LIST_ITEMS]
        if not main_points and stripped_answer:
            main_points = [stripped_answer]

        summary_text = (
            candidate_sentences[0]
            if candidate_sentences
            else (main_points[0] if main_points else "No documented guidance yet.")
        )

        candidate_lines = self._extract_candidate_lines(raw_answer)
        scope_markers = (
            "applies",
            "applicable",
            "available to",
            "covers",
            "eligible",
            "limited to",
            "for all",
            "for any",
            "governs",
            "responsible",
            "subject to",
        )
        positive_markers = (
            " can ",
            " can.",
            " allowed",
            " allow",
            " should ",
            " recommended",
            " may ",
            " must ",
        )
        negative_markers = (
            " cannot",
            " can't",
            " not allowed",
            " forbidden",
            " must not",
            " avoid",
            " prohibited",
            " restriction",
            " no longer",
        )

        scope_items: List[str] = []
        can_items: List[str] = []
        cannot_items: List[str] = []
        for line in candidate_lines:
            clean_line = self._normalise_line(line)
            if not clean_line:
                continue
            lowered = clean_line.lower()
            if any(marker in lowered for marker in scope_markers) or lowered.startswith(
                ("employees", "suppliers", "vendors", "requesters", "approvers", "teams")
            ):
                scope_items.append(clean_line)
            if any(marker in lowered for marker in positive_markers):
                can_items.append(clean_line)
            if any(marker in lowered for marker in negative_markers):
                cannot_items.append(clean_line)

        if not scope_items:
            entities = [
                str(entity).strip()
                for entity in extracted_data.get("entities", [])
                if str(entity).strip()
            ]
            focus = self._derive_focus_from_query(query)
            if entities:
                audience = ", ".join(entities[:2])
                scope_items.append(
                    f"Most relevant to {audience} when working on {focus or 'this topic'}."
                )
            else:
                scope_items.append(
                    f"Applies to procurement activity related to {focus or 'this topic'} unless the policy states otherwise."
                )
        scope_items = scope_items[:_MAX_ATOM_LIST_ITEMS]

        if not can_items and main_points:
            can_items = main_points[: min(len(main_points), 2)]
        if not can_items:
            can_items = [
                "No explicit allowances surfaced in the retrieved material.",
            ]

        if not cannot_items and plan.get("type") == "policy_lookup":
            cannot_items = [
                "No prohibitions were highlighted, so double-check the policy before proceeding.",
            ]
        elif not cannot_items:
            cannot_items = [
                "No specific restrictions were called out in the references.",
            ]

        table_rows: List[Tuple[str, str]] = []
        for idx, point in enumerate(main_points[: _MAX_ATOM_LIST_ITEMS], start=1):
            condition, description = self._split_condition_description(point, idx)
            table_rows.append((condition, description))
        if not table_rows:
            focus = self._derive_focus_from_query(query)
            table_rows = [("Key Insight", f"No detailed records found for {focus or 'this topic'}.")]

        notes: List[str] = []
        note_candidates = candidate_sentences[1:4]
        if not note_candidates and len(main_points) > 1:
            note_candidates = main_points[1:3]
        for text in note_candidates:
            cleaned = text.strip()
            if cleaned:
                notes.append(cleaned)
            if len(notes) >= _MAX_ATOM_LIST_ITEMS:
                break
        if not notes:
            notes = [
                "Capture any approvals or documentation before you proceed.",
            ]

        return {
            "summary": summary_text,
            "scope": scope_items,
            "rules": can_items,
            "exclusions": cannot_items,
            "policies": table_rows,
            "next_steps": notes,
            "followup": self._format_followup_text(followups),
        }

    def _format_followup_text(self, followups: Optional[Sequence[str]]) -> str:
        if followups:
            first = None
            for item in followups:
                if isinstance(item, str) and item.strip():
                    first = item.strip()
                    break
            if first:
                cleaned = first.rstrip("?.!").strip()
                if cleaned:
                    return f"Share {cleaned.lower()} next if you want deeper coverage."
        return "Share another detail when you want a deeper dive."

    def _request_atoms_via_llm(
        self,
        *,
        query: str,
        raw_answer: str,
        docs: Sequence[Any],
        heuristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        context_snippets = self._collect_atom_context(raw_answer, docs)
        payload = {
            "question": query,
            "context_snippets": context_snippets,
            "heuristics": heuristics,
        }
        user_instruction = (
            json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nReturn ONLY valid JSON with keys: summary, scope, rules, exclusions, policies, next_steps, followup."
        )

        model_name = self._resolve_atom_model_name()
        response = self.call_ollama(
            model=model_name,
            messages=[
                {"role": "system", "content": ATOM_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_instruction},
            ],
            options={
                "temperature": 0.4,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 1024,
            },
        )
        raw_text = self._extract_llm_message(response)
        return self._parse_atom_json(raw_text)

    def _resolve_atom_model_name(self, fallback: str = "phi4:latest") -> str:
        """Return the preferred LLM identifier for HTML atom extraction."""

        # ``BaseAgent`` exposes ``get_agent_model`` in newer deployments. Older
        # builds might not, so we defensively probe before invoking to avoid the
        # AttributeError observed in production.
        resolver = getattr(self, "get_agent_model", None)
        if callable(resolver):
            try:
                model_name = resolver("rag_agent_html_atoms", fallback=fallback)
            except TypeError:
                model_name = resolver("rag_agent_html_atoms")  # type: ignore[misc]
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()

        # Allow the agent nickname wrapper to supply overrides when available.
        nick_resolver = getattr(self.agent_nick, "get_agent_model", None)
        if callable(nick_resolver):
            try:
                model_name = nick_resolver("rag_agent_html_atoms", fallback=fallback)
            except TypeError:
                model_name = nick_resolver("rag_agent_html_atoms")  # type: ignore[misc]
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()

        configured = getattr(self.settings, "rag_model", None)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()

        return fallback

    @staticmethod
    def _extract_llm_message(response: Dict[str, Any]) -> str:
        if not isinstance(response, dict):
            return ""
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        content = response.get("response")
        if isinstance(content, str):
            return content.strip()
        return ""

    def _collect_atom_context(self, raw_answer: str, docs: Sequence[Any]) -> List[str]:
        snippets: List[str] = []
        stripped = self._clean_prompt_text(self._strip_markdown(raw_answer or ""))
        if stripped:
            snippets.append(self._truncate_context(f"Answer scope: {stripped}"))
        for doc in docs[:8]:
            payload = getattr(doc, "payload", {}) or {}
            summary = payload.get("summary") or payload.get("text_summary")
            label = payload.get("source_label") or payload.get("record_id") or "Context"
            snippet = f"{label}: {summary}" if summary else str(label)
            cleaned = self._clean_prompt_text(snippet)
            if cleaned:
                snippets.append(self._truncate_context(cleaned))
            if len(snippets) >= 8:
                break
        return snippets

    def _truncate_context(self, text: str, limit: int = 600) -> str:
        cleaned = text.strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    def _clean_prompt_text(self, text: str) -> str:
        cleaned = (text or "").replace("static_policy", "policy library")
        cleaned = re.sub(r"document_id\s*[:=]\s*\S+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"procwise_document_embeddings", "procurement knowledge base", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_atom_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r'\\(?!["\/bfnrtu])', "", candidate)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _merge_atom_payloads(
        self,
        fallback: Dict[str, Any],
        model_atoms: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(fallback)
        for key in ("summary", "followup"):
            value = model_atoms.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()

        list_mappings: Dict[str, Tuple[str, ...]] = {
            "scope": ("scope", "applicability", "audience", "coverage"),
            "rules": ("rules", "allowed"),
            "exclusions": ("exclusions", "prohibited"),
            "next_steps": ("next_steps", "notes"),
        }

        for target, candidates in list_mappings.items():
            for candidate in candidates:
                value = model_atoms.get(candidate)
                if isinstance(value, (list, tuple)):
                    cleaned: List[str] = []
                    for item in value:
                        text = str(item).strip()
                        if text:
                            cleaned.append(text)
                        if len(cleaned) >= _MAX_ATOM_LIST_ITEMS:
                            break
                    if cleaned:
                        merged[target] = cleaned
                        break
                elif isinstance(value, str) and value.strip():
                    merged[target] = [value.strip()]
                    break

        policy_value = None
        for key in ("policies", "policy_rows", "details"):
            value = model_atoms.get(key)
            if isinstance(value, (list, tuple)):
                policy_value = value
                break
        if policy_value is not None:
            rows: List[Tuple[str, str, str]] = []
            for idx, entry in enumerate(policy_value, start=1):
                name: str
                effective: str
                note_text: str
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 3:
                        name = str(entry[0]).strip()
                        effective = str(entry[1]).strip()
                        note_text = str(entry[2]).strip()
                    elif len(entry) == 2:
                        name = str(entry[0]).strip()
                        effective = ""
                        note_text = str(entry[1]).strip()
                    elif len(entry) == 1:
                        name = str(entry[0]).strip()
                        effective = ""
                        note_text = ""
                    else:
                        continue
                elif isinstance(entry, dict):
                    name = str(
                        entry.get("name")
                        or entry.get("policy")
                        or entry.get("title")
                        or entry.get("condition")
                        or f"Item {idx}"
                    ).strip()
                    effective = str(
                        entry.get("effective")
                        or entry.get("date")
                        or entry.get("timeline")
                        or entry.get("window")
                        or ""
                    ).strip()
                    note_text = str(
                        entry.get("notes")
                        or entry.get("owner")
                        or entry.get("description")
                        or entry.get("value")
                        or ""
                    ).strip()
                elif isinstance(entry, str) and entry.strip():
                    name, description = self._split_condition_description(entry, idx)
                    name = name.strip()
                    note_text = description.strip()
                    effective = ""
                else:
                    continue
                rows.append((name, effective, note_text))
                if len(rows) >= _MAX_ATOM_LIST_ITEMS:
                    break
            if rows:
                merged["policies"] = rows
        return merged

    def _finalise_atoms(self, atoms: Dict[str, Any]) -> Dict[str, Any]:
        summary = str(atoms.get("summary") or "").strip()
        if not summary:
            summary = "No documented guidance surfaced yet; consider checking with procurement operations."

        def _ensure_named_list(primary: str, default: List[str], *aliases: str) -> List[str]:
            candidates = (primary,) + aliases
            for name in candidates:
                value = atoms.get(name)
                if isinstance(value, (list, tuple)):
                    cleaned = [str(item).strip() for item in value if str(item).strip()]
                    if cleaned:
                        return cleaned[:_MAX_ATOM_LIST_ITEMS]
                elif isinstance(value, str) and value.strip():
                    return [value.strip()]
            return default

        scope = _ensure_named_list(
            "scope",
            [
                "Applies across procurement activity unless a policy clause narrows the audience.",
            ],
            "applicability",
            "audience",
        )
        rules = _ensure_named_list(
            "rules",
            ["No explicit allowances were identified in the retrieved records."],
            "allowed",
        )
        exclusions = _ensure_named_list(
            "exclusions",
            ["No prohibitions were captured; double-check the policy for edge cases."],
            "prohibited",
        )
        next_steps = _ensure_named_list(
            "next_steps",
            ["Raise a ticket if you need deeper analysis or updated approvals."],
            "notes",
        )

        policies_value = atoms.get("policies") or atoms.get("details")
        policies: List[Tuple[str, str, str]] = []
        if isinstance(policies_value, (list, tuple)):
            for idx, entry in enumerate(policies_value, start=1):
                name: str
                effective: str
                note_text: str
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 3:
                        name = str(entry[0]).strip() or f"Item {idx}"
                        effective = str(entry[1]).strip()
                        note_text = str(entry[2]).strip()
                    elif len(entry) == 2:
                        name = str(entry[0]).strip() or f"Item {idx}"
                        effective = ""
                        note_text = str(entry[1]).strip()
                    elif len(entry) == 1:
                        name = str(entry[0]).strip() or f"Item {idx}"
                        effective = ""
                        note_text = ""
                    else:
                        continue
                elif isinstance(entry, dict):
                    name = str(
                        entry.get("name")
                        or entry.get("policy")
                        or entry.get("title")
                        or entry.get("condition")
                        or f"Item {idx}"
                    ).strip()
                    effective = str(
                        entry.get("effective")
                        or entry.get("date")
                        or entry.get("timeline")
                        or entry.get("window")
                        or ""
                    ).strip()
                    note_text = str(
                        entry.get("notes")
                        or entry.get("owner")
                        or entry.get("description")
                        or entry.get("value")
                        or ""
                    ).strip()
                elif isinstance(entry, str) and entry.strip():
                    name, description = self._split_condition_description(entry, idx)
                    effective = ""
                    note_text = description.strip()
                else:
                    continue
                policies.append((name, effective, note_text))
                if len(policies) >= _MAX_ATOM_LIST_ITEMS:
                    break
        if not policies:
            policies = [
                (
                    "Reference",
                    "Not specified",
                    "No effective date or ownership details were highlighted in the retrieved material.",
                )
            ]

        followup = str(atoms.get("followup") or "").strip()
        if not followup:
            followup = "Let me know if you want me to gather additional procurement detail next."

        def _strip_banned(text: str) -> str:
            cleaned = text
            for token in _BANNED_TOKENS:
                cleaned = cleaned.replace(token, "")
            return cleaned.strip()

        summary = _strip_banned(summary)
        scope = [_strip_banned(item) for item in scope]
        rules = [_strip_banned(item) for item in rules]
        exclusions = [_strip_banned(item) for item in exclusions]
        next_steps = [_strip_banned(item) for item in next_steps]
        policies = [
            (
                _strip_banned(name),
                _strip_banned(effective) or "Not specified",
                _strip_banned(note) or "No notes captured.",
            )
            for name, effective, note in policies
        ]
        followup = _strip_banned(followup)

        return {
            "summary": summary,
            "scope": scope,
            "rules": rules,
            "exclusions": exclusions,
            "policies": policies,
            "next_steps": next_steps,
            "followup": followup,
        }

    def _strip_markdown(self, text: str) -> str:
        cleaned = text.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = re.sub(r"`([^`]*)`", r"\\1", cleaned)
        cleaned = re.sub(r"#+\\s*", "", cleaned)
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\\1", cleaned)
        return cleaned

    def _extract_candidate_lines(self, text: str) -> List[str]:
        if not text:
            return []
        lines = [segment.strip() for segment in text.splitlines() if segment.strip()]
        return lines

    def _normalise_line(self, text: str) -> str:
        cleaned = re.sub(r"^[\-•*]+", "", text).strip()
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        return cleaned.strip()

    def _split_condition_description(self, text: str, index: int) -> Tuple[str, str]:
        cleaned = self._normalise_line(self._strip_markdown(text))
        if ":" in cleaned:
            condition, description = cleaned.split(":", 1)
            return condition.strip() or f"Key Insight {index}", description.strip() or cleaned
        if "-" in cleaned:
            parts = cleaned.split("-", 1)
            left, right = parts[0].strip(), parts[1].strip()
            if left and right:
                return left, right
        return f"Key Insight {index}", cleaned or "Detail not provided."

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
        policy_docs: Sequence[SimpleNamespace] = (),
        primary_docs: Sequence[SimpleNamespace] = (),
    ) -> Dict[str, Any]:
        categories: Dict[str, List[str]] = defaultdict(list)
        dynamic_presence: Set[str] = set()
        doc_overview_candidates: List[str] = []

        def _record(key: str, clause: str, *, dynamic: bool) -> None:
            cleaned = self._clean_policy_clause(clause)
            if not cleaned:
                return
            if not dynamic and dynamic_presence and key in dynamic_presence:
                return
            categories[key].append(cleaned)
            if dynamic:
                dynamic_presence.add(key)

        def _ingest_text(block: str, *, dynamic: bool) -> None:
            if not block:
                return
            sentences = self._split_sentences(block)
            for sentence in sentences:
                lowered = sentence.lower()
                if re.search(r"\b(must|shall|need to|ensure|submit|retain|provide|keep|require)\b", lowered):
                    _record("requirements", sentence, dynamic=dynamic)
                if re.search(
                    r"\b(non-claimable|not allowed|cannot|can't|prohibit|forbidden|declined|never)\b",
                    lowered,
                ) or "non-claimable" in lowered:
                    _record("restrictions", sentence, dynamic=dynamic)
                if re.search(r"[£$€]\s*[\d,.]+", sentence) or re.search(
                    r"\b(limit|cap|threshold|per person|per day|per month|ceiling)\b",
                    lowered,
                ):
                    _record("spending_limits", sentence, dynamic=dynamic)
                if re.search(
                    r"approval|approve|authoris|manager|finance|sign-off|review",
                    lowered,
                ):
                    _record("approval_process", sentence, dynamic=dynamic)
                if re.search(r"for example|such as|e.g.|include", lowered):
                    _record("examples", sentence, dynamic=dynamic)
                if re.search(r"unless|exception|exemption|waiver|if you need an exception", lowered):
                    _record("exceptions", sentence, dynamic=dynamic)
                if re.search(
                    r"workflow|process|document|retain|attach|audit|reconcile|submit|evidence|record",
                    lowered,
                ) or re.search(r"\b(po|purchase order|invoice|coding)\b", lowered):
                    _record("operational_notes", sentence, dynamic=dynamic)

        def _ingest_payload(payload: Dict[str, Any], *, dynamic: bool) -> None:
            summary = payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                if dynamic:
                    doc_overview_candidates.append(summary)
                _ingest_text(summary, dynamic=dynamic)

            for key in ("highlights", "sections", "key_points"):
                value = payload.get(key)
                if isinstance(value, str):
                    _ingest_text(value, dynamic=dynamic)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            _ingest_text(item, dynamic=dynamic)
                        elif isinstance(item, dict):
                            for field in ("text", "content", "summary", "value"):
                                text_value = item.get(field)
                                if isinstance(text_value, str):
                                    _ingest_text(text_value, dynamic=dynamic)

        for doc in policy_docs or []:
            payload = getattr(doc, "payload", {}) or {}
            _ingest_payload(payload, dynamic=True)

        for doc in primary_docs or []:
            payload = getattr(doc, "payload", {}) or {}
            _ingest_payload(payload, dynamic=True)

        for qa in topic_entry.qas:
            question_lower = qa.question.lower()
            answer_sentences = self._split_sentences(qa.answer)

            _ingest_text(qa.answer, dynamic=False)

            if "exceptions" not in dynamic_presence and (
                "exception" in question_lower or "waiver" in question_lower
            ):
                categories["exceptions"].extend(answer_sentences)
            if "examples" not in dynamic_presence and "example" in question_lower:
                categories["examples"].extend(answer_sentences)
            if (
                "spending_limits" not in dynamic_presence
                and any(keyword in question_lower for keyword in ["limit", "cap", "threshold"])
            ):
                categories["spending_limits"].extend(answer_sentences)
            if (
                "approval_process" not in dynamic_presence
                and ("approval" in question_lower or "pre-approv" in question_lower)
            ):
                categories["approval_process"].extend(answer_sentences)
            if (
                "requirements" not in dynamic_presence
                and any(keyword in question_lower for keyword in ["must", "how do i comply"])
            ):
                categories["requirements"].extend(answer_sentences)

        overview = ""
        if doc_overview_candidates:
            overview = doc_overview_candidates[0]
        elif focus_answer:
            sentences = self._split_sentences(focus_answer)
            overview = sentences[0] if sentences else focus_answer

        payload = {
            "policy_name": policy_name,
            "overview": self._ensure_sentence(self._clean_policy_clause(overview)) if overview else "",
            "requirements": self._unique_ordered(categories.get("requirements", [])),
            "restrictions": self._unique_ordered(categories.get("restrictions", [])),
            "spending_limits": self._unique_ordered(categories.get("spending_limits", [])),
            "approval_process": self._unique_ordered(categories.get("approval_process", [])),
            "examples": self._unique_ordered(categories.get("examples", [])),
            "exceptions": self._unique_ordered(categories.get("exceptions", [])),
            "operational_notes": self._unique_ordered(categories.get("operational_notes", [])),
        }

        if depth_mode == "expanded":
            for key in ["requirements", "restrictions", "spending_limits", "approval_process", "operational_notes"]:
                values = payload.get(key, [])
                if not values and payload.get("overview"):
                    values.append(payload["overview"])
                payload[key] = values

        return payload

    def _render_policy_response(
        self,
        payload: Dict[str, Any],
        depth_mode: str,
        query: Optional[str] = None,
    ) -> str:
        policy_name = self._format_policy_title(payload.get("policy_name", "Policy Guidance"))
        focus = self._derive_focus_from_query(query or policy_name)
        focus_text = focus if focus and focus != "the topic" else "this request"
        overview = payload.get("overview", "")
        overview_text = self._ensure_sentence(self._clean_policy_clause(overview)) if overview else ""

        detail = overview_text or "I pulled the relevant guardrails straight from our policy library and procurement records."
        summary_line = f"Here’s what {policy_name} says about {focus_text}: {detail}"

        lines: List[str] = [self._ensure_sentence(summary_line)]

        def _append_section(title: str, items: Sequence[str]) -> None:
            bullets = [self._ensure_sentence(text) for text in self._unique_ordered(items)]
            if not bullets:
                return
            lines.append("")
            lines.append(title)
            lines.extend(f"- {bullet}" for bullet in bullets)

        allowed = self._policy_section_bullets(payload, "requirements", depth_mode)
        restricted = self._policy_section_bullets(payload, "restrictions", depth_mode)

        notes_pool: List[str] = []
        for key in ("spending_limits", "approval_process", "operational_notes"):
            notes_pool.extend(self._policy_section_bullets(payload, key, depth_mode))
        for key in ("examples", "exceptions"):
            notes_pool.extend(self._policy_section_bullets(payload, key, depth_mode))

        _append_section("✅ **Allowed Conditions / Requirements**", allowed)
        _append_section("❌ **Prohibited / Restricted Conditions**", restricted)
        _append_section("📎 **Important Notes / Compliance Obligations**", notes_pool)

        next_steps = "Need help submitting an expense claim or checking approvals? Just ask and I’ll walk you through it."
        _append_section("➡️ **Next Steps**", [next_steps])

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
            "operational_notes": "Keep documentation tidy—attach receipts, coding, and approvals in the workflow.",
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
        elif section == "operational_notes":
            bullets.append(self._ensure_sentence(clause))

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
