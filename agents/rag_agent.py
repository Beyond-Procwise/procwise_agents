"""Qdrant-backed retrieval agent powering Joshi's procurement RAG workflow."""

from __future__ import annotations

import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base_agent import AgentOutput, AgentStatus, BaseAgent
from services.conversation_memory import ConversationMemoryService
from services.feedback_service import FeedbackSentiment, FeedbackService
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StaticQAEntry:
    """Single FAQ-style memory used for instant responses."""

    topic: str
    question: str
    answer: str
    prompts: Tuple[str, ...]


class RAGAgent(BaseAgent):
    """Joshi's retrieval-augmented generation agent."""

    _STATIC_DATASET_PATH = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "reference_data"
        / "procwise_mvp_chat_questions.json"
    )

    _STATIC_DATASET: Optional[Tuple[_StaticQAEntry, ...]] = None
    _STATIC_EMBEDDINGS: Dict[int, np.ndarray] = {}

    def __init__(
        self,
        agent_nick: Any,
        *,
        rag_service: Optional[RAGService] = None,
        conversation_memory: Optional[ConversationMemoryService] = None,
    ) -> None:
        super().__init__(agent_nick)
        self.feedback_service = FeedbackService(agent_nick)
        self.rag_service = rag_service or RAGService(agent_nick)
        self.conversation_memory = conversation_memory or ConversationMemoryService(
            agent_nick, rag_service=self.rag_service
        )
        self._session_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._ensure_static_memory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        **_: Any,
    ) -> AgentOutput:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        cleaned_query = re.sub(r"\s+", " ", query).strip()
        session_key = self._session_key(user_id, session_id)
        state = self._session_state.setdefault(session_key, {"history": []})
        last_question = state.get("last_question")
        continuation_detail: Optional[str] = None

        continuation_detected = self.feedback_service.is_continuation_request(cleaned_query)
        if not continuation_detected and last_question:
            if self._is_follow_up_phrase(cleaned_query):
                continuation_detected = True
        if continuation_detected:
            if last_question:
                continuation_detail = cleaned_query
                cleaned_query = last_question
        else:
            sentiment, confidence = self.feedback_service.detect_feedback(cleaned_query)
            if (
                sentiment == FeedbackSentiment.POSITIVE
                and confidence >= 0.4
                and last_question
            ):
                acknowledgment = "Glad that helped! Let me know if you need anything else."
                payload = {
                    "answer": acknowledgment,
                    "follow_up_questions": ["Would you like support with another procurement task?"],
                    "retrieved_documents": [],
                }
                snapshot = {
                    "feedback_sentiment": sentiment.value,
                    "feedback_confidence": confidence,
                    "handled_as_feedback": True,
                }
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    confidence=1.0,
                    agentic_plan="1. Detect positive feedback.\n2. Acknowledge succinctly.",
                    context_snapshot=snapshot,
                )

        ack_text = self._build_acknowledgment(
            cleaned_query, continuation_detail, last_question
        )
        expanded_query = self._expand_query(cleaned_query, continuation_detail)

        static_hit = self._match_static_memory(expanded_query)
        if static_hit is not None:
            entry, similarity = static_hit
            answer_body = entry.answer.strip()
            answer = f"{ack_text}\n\n{answer_body}" if answer_body else ack_text
            followups = list(entry.prompts[:3])
            retrieved_docs = [
                {
                    "title": entry.topic,
                    "collection": "static_faq",
                    "source_type": "FAQ",
                }
            ]
            state.update({"last_question": entry.question, "last_answer": answer})
            self._update_history(state, entry.question, answer)
            snapshot = {
                "static_match": True,
                "static_similarity": float(similarity),
                "session_id": session_key[1],
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "answer": answer,
                    "follow_up_questions": followups,
                    "retrieved_documents": retrieved_docs,
                },
                confidence=float(max(0.0, min(1.0, similarity))),
                agentic_plan=(
                    "1. Detect question intent.\n"
                    "2. Match against static procurement FAQs.\n"
                    "3. Return the stored authoritative answer."
                ),
                context_snapshot=snapshot,
            )

        policy_mode = self._looks_like_policy(cleaned_query)
        memory_fragments = self._recent_history_snippets(state)

        hits = self._search_documents(
            expanded_query,
            session_id=session_key[1],
            memory_fragments=memory_fragments,
            policy_mode=policy_mode,
        )

        retrieved_docs = self._summarise_retrieved_documents(hits)
        context_sections = self._build_context_sections(hits)

        if not context_sections:
            fallback_answer = (
                f"{ack_text}\n\n"
                "I could not locate supporting material for that yet. "
                "Let me know if you can share more specifics so I can dig deeper."
            )
            followups = [
                "Do you want me to look at a particular policy or supplier record?",
            ]
            state.update({"last_question": cleaned_query, "last_answer": fallback_answer})
            self._update_history(state, cleaned_query, fallback_answer)
            snapshot = {
                "static_match": False,
                "retrieved_count": 0,
                "session_id": session_key[1],
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "answer": fallback_answer,
                    "follow_up_questions": followups,
                    "retrieved_documents": [],
                },
                confidence=0.35,
                agentic_plan=(
                    "1. Acknowledge the question.\n"
                    "2. Attempt retrieval across uploads, procurement docs, and policies.\n"
                    "3. Report limited context and invite clarification."
                ),
                context_snapshot=snapshot,
            )

        llm_answer, llm_followups = self._generate_answer_with_phi4(
            ack_text,
            cleaned_query,
            context_sections,
            history=memory_fragments,
            policy_mode=policy_mode,
        )

        followups = (
            llm_followups
            if llm_followups
            else self._default_followups(cleaned_query, policy_mode)
        )
        answer = llm_answer or ack_text
        if not answer.startswith(ack_text):
            answer = f"{ack_text}\n\n{answer.strip()}" if answer.strip() else ack_text

        state.update({"last_question": cleaned_query, "last_answer": answer})
        self._update_history(state, cleaned_query, answer)

        top_score = max((float(getattr(hit, "combined_score", 0.0)) for hit in hits), default=0.0)
        confidence = 1.0 - math.exp(-max(0.0, top_score) / 6.0)
        snapshot = {
            "static_match": False,
            "retrieved_count": len(retrieved_docs),
            "policy_mode": policy_mode,
            "session_id": session_key[1],
            "top_score": top_score,
        }

        agentic_plan = (
            "1. Interpret the request and craft an acknowledgement.\n"
            "2. Expand the query, retrieve supporting context across uploads, procurement documents, and policies.\n"
            "3. Synthesise a grounded response with phi4 and propose next steps."
        )

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "answer": answer,
                "follow_up_questions": followups,
                "retrieved_documents": retrieved_docs,
            },
            confidence=float(max(0.0, min(1.0, confidence))),
            agentic_plan=agentic_plan,
            context_snapshot=snapshot,
        )

    # ------------------------------------------------------------------
    # Static memory helpers
    # ------------------------------------------------------------------
    @classmethod
    def _load_static_dataset(cls) -> Tuple[_StaticQAEntry, ...]:
        if cls._STATIC_DATASET is not None:
            return cls._STATIC_DATASET
        path = cls._STATIC_DATASET_PATH
        if not path.exists():
            logger.warning("Static QA dataset missing at %s", path)
            cls._STATIC_DATASET = tuple()
            return cls._STATIC_DATASET
        with path.open("r", encoding="utf-8") as handle:
            raw_entries = json.load(handle)

        entries: List[_StaticQAEntry] = []
        for item in raw_entries:
            topic = str(item.get("topic", "")).strip()
            prompts = tuple(
                str(prompt).strip()
                for prompt in item.get("context_prompts", [])
                if str(prompt).strip()
            )
            for qa in item.get("qas", []):
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()
                if question and answer:
                    entries.append(
                        _StaticQAEntry(topic=topic, question=question, answer=answer, prompts=prompts)
                    )
        cls._STATIC_DATASET = tuple(entries)
        return cls._STATIC_DATASET

    def _ensure_static_memory(self) -> None:
        dataset = self._load_static_dataset()
        embedder = getattr(self.agent_nick, "embedding_model", None)
        if embedder is None or not hasattr(embedder, "encode"):
            logger.warning("RAGAgent missing embedding model; static QA disabled")
            return

        embedder_id = id(embedder)
        if embedder_id in self._STATIC_EMBEDDINGS:
            return

        if not dataset:
            self._STATIC_EMBEDDINGS[embedder_id] = np.zeros((0, 1), dtype="float32")
            return

        questions = [entry.question for entry in dataset]
        try:
            vectors = embedder.encode(questions)
        except Exception:
            logger.exception("Failed to encode static QA questions")
            vectors = np.zeros((len(questions), 1), dtype="float32")

        array = np.array(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = array / norms
        self._STATIC_EMBEDDINGS[embedder_id] = normalised

    def _match_static_memory(
        self, query: str
    ) -> Optional[Tuple[_StaticQAEntry, float]]:
        dataset = self._load_static_dataset()
        if not dataset:
            return None
        embedder = getattr(self.agent_nick, "embedding_model", None)
        if embedder is None or not hasattr(embedder, "encode"):
            return None
        embedder_id = id(embedder)
        vectors = self._STATIC_EMBEDDINGS.get(embedder_id)
        if vectors is None or not len(vectors):
            return None

        try:
            query_vec = embedder.encode([query])
        except Exception:
            logger.exception("Failed to encode query for static QA match")
            return None

        query_arr = np.array(query_vec, dtype="float32")
        if query_arr.ndim > 1:
            query_arr = query_arr[0]
        norm = np.linalg.norm(query_arr)
        if norm == 0:
            return None
        query_arr = query_arr / norm
        scores = vectors @ query_arr
        if scores.size == 0:
            return None
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < 0.82:
            return None
        return dataset[best_idx], best_score

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _search_documents(
        self,
        query: str,
        *,
        session_id: Optional[str],
        memory_fragments: Optional[List[str]],
        policy_mode: bool,
    ) -> List[SimpleNamespace]:
        try:
            hits = self.rag_service.search(
                query,
                top_k=6,
                session_hint=None,
                memory_fragments=memory_fragments,
                policy_mode=policy_mode,
                session_id=session_id,
                collections=(
                    self.rag_service.uploaded_collection,
                    self.rag_service.primary_collection,
                    self.rag_service.static_policy_collection,
                ),
            )
        except Exception:
            logger.exception("RAGService search failed; returning empty hits")
            return []
        return list(hits or [])

    def _summarise_retrieved_documents(
        self, hits: Iterable[SimpleNamespace]
    ) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            snippet = self._extract_snippet(payload)
            if not snippet:
                continue
            metadata = self._sanitize_metadata(payload)
            metadata["snippet"] = snippet
            documents.append(metadata)
        return documents[:5]

    def _extract_snippet(self, payload: Dict[str, Any]) -> str:
        for key in ("text_summary", "content", "summary", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                snippet = re.sub(r"\s+", " ", value).strip()
                if len(snippet) > 1200:
                    snippet = snippet[:1197] + "..."
                return snippet
        return ""

    def _sanitize_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed: Dict[str, Any] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if "id" in lowered or lowered in {"record_id", "doc_number", "supplier"}:
                continue
            if lowered in {"content", "text_summary", "summary", "full_text"}:
                continue
            if lowered in {"chunk_id", "chunk_index", "chunk_hash"}:
                continue
            if lowered == "collection_name":
                allowed["collection"] = str(value)
                continue
            if isinstance(value, (str, int, float)):
                allowed[key] = value
        if "title" not in allowed:
            allowed["title"] = payload.get("title") or payload.get("document_name") or "Document"
        if "collection" not in allowed:
            allowed["collection"] = payload.get("collection_name", "procwise_document_embeddings")
        if "source_type" not in allowed and payload.get("document_type"):
            allowed["source_type"] = payload.get("document_type")
        return allowed

    def _build_context_sections(
        self, hits: Iterable[SimpleNamespace]
    ) -> List[str]:
        sections: List[str] = []
        seen_snippets: set[str] = set()
        for idx, hit in enumerate(hits, start=1):
            payload = getattr(hit, "payload", {}) or {}
            snippet = self._extract_snippet(payload)
            if not snippet or snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            header_parts: List[str] = []
            title = payload.get("title") or payload.get("document_name")
            if title:
                header_parts.append(str(title))
            source_type = payload.get("source_type") or payload.get("document_type")
            if source_type:
                header_parts.append(str(source_type))
            collection = payload.get("collection_name")
            if collection:
                header_parts.append(str(collection))
            header = " • ".join(part for part in header_parts if part)
            cleaned = textwrap.dedent(snippet).strip()
            sections.append(f"Document {idx}: {header}\n{cleaned}")
            if len(sections) >= 5:
                break
        return sections

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _generate_answer_with_phi4(
        self,
        ack_text: str,
        question: str,
        context_sections: Sequence[str],
        *,
        history: Optional[List[str]],
        policy_mode: bool,
    ) -> Tuple[str, List[str]]:
        history_text = "\n\n".join(history or [])
        context_block = "\n\n".join(context_sections)
        instructions = [
            "Write as a helpful procurement advisor (Joshi) with a warm, semi-formal tone.",
            "Do not repeat the acknowledgement – start directly with the guidance.",
            "Use bullet points for rules, limits, or step-by-step instructions.",
            "Keep the response concise (roughly 3–6 sentences or bullet points).",
            "Never mention internal IDs, collection names, or record identifiers.",
            "If policies are included, organise content under headings such as 'What You Must Do', 'What’s Prohibited', 'Spending Limits', 'Approval Requirements', 'Examples', and 'Exceptions' when relevant.",
        ]
        if policy_mode:
            instructions.append(
                "Focus on policy obligations and compliance nuances while remaining practical."
            )

        prompt = (
            f"User question: {question}\n"
            f"Acknowledgement prefix: {ack_text}\n"
            "Prior conversation snippets (last few turns):\n"
        )
        prompt = f"{prompt}{history_text or 'None provided.'}\n\nRetrieved context:\n{context_block}".strip()

        system_message = (
            "You are Joshi, the ProcWise subject-matter expert."
            " Respond with grounded procurement guidance, cite no internal artefacts,"
            " and remain empathetic yet efficient."
            " Return your output wrapped inside <answer>...</answer> and optional"
            " <followups>...</followups> tags with one question per line."
        )
        final_instruction = "\n".join(instructions)

        try:
            response = self.call_ollama(
                messages=[
                    {"role": "system", "content": f"{system_message}\n{final_instruction}"},
                    {"role": "user", "content": prompt},
                ],
                model=getattr(self.settings, "rag_model", "phi4:latest"),
            )
        except Exception:
            logger.exception("phi4 generation failed")
            return ack_text, []

        content = ""
        if isinstance(response, dict):
            message = response.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")
            if not content:
                content = response.get("response", "")
        if not isinstance(content, str):
            content = str(content or "")

        answer = self._extract_tagged_section(content, "answer")
        raw_followups = [
            item
            for item in self._extract_tagged_section(content, "followups").splitlines()
            if item.strip()
        ]
        cleaned_followups: List[str] = []
        for item in raw_followups:
            cleaned = re.sub(r"^\s*[-*•]+\s*", "", item).strip()
            cleaned = re.sub(r"^\s*\d+[.)]\s*", "", cleaned)
            if cleaned:
                cleaned_followups.append(cleaned)
        return answer.strip(), cleaned_followups[:3]

    def _extract_tagged_section(self, text: str, tag: str) -> str:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text or "")
        if match:
            return match.group(1).strip()
        return ""

    def _default_followups(self, query: str, policy_mode: bool) -> List[str]:
        lowered = query.lower()
        followups: List[str] = []
        if policy_mode or "policy" in lowered:
            followups.append("Would you like me to clarify any prohibited spend examples?")
            followups.append("Do you need help confirming approval thresholds for your team?")
        if "supplier" in lowered:
            followups.append("Should I check recent supplier performance or relationship notes?")
        followups.append("Is there another procurement control or document you'd like to review?")
        deduped: List[str] = []
        for item in followups:
            if item not in deduped:
                deduped.append(item)
        return deduped[:3]

    # ------------------------------------------------------------------
    # Session utilities
    # ------------------------------------------------------------------
    def _session_key(self, user_id: str, session_id: Optional[str]) -> Tuple[str, str]:
        session = session_id or user_id or "default"
        return user_id, session

    def _update_history(self, state: Dict[str, Any], question: str, answer: str) -> None:
        history = state.setdefault("history", [])
        history.append({"question": question, "answer": answer})
        if len(history) > 5:
            del history[:-5]

    def _recent_history_snippets(self, state: Dict[str, Any]) -> List[str]:
        snippets: List[str] = []
        for item in state.get("history", [])[-3:]:
            question = item.get("question")
            answer = item.get("answer")
            if question and answer:
                snippets.append(f"User asked: {question}\nJoshi replied: {answer}")
        return snippets

    def _build_acknowledgment(
        self,
        query: str,
        continuation_detail: Optional[str],
        last_question: Optional[str],
    ) -> str:
        core_question = query.rstrip("?.!")
        if continuation_detail and last_question:
            detail = self._extract_followup_detail(continuation_detail)
            if detail:
                return (
                    f"Got it. You're asking about {last_question.rstrip('?.!')} and you'd like more detail on {detail}."
                )
            return f"Got it. You're asking for more detail on {last_question.rstrip('?.!')}"
        return f"Got it. You're asking about {core_question}."

    def _extract_followup_detail(self, text: str) -> str:
        cleaned = re.sub(r"^(please|kindly)\s+", "", text.strip(), flags=re.IGNORECASE)
        cleaned = cleaned.rstrip("?.!")
        fillers = {"could you elaborate", "can you elaborate", "any update", "more detail"}
        lowered = cleaned.lower()
        for filler in fillers:
            if lowered == filler:
                return "that"
        return cleaned

    def _expand_query(
        self, base: str, continuation: Optional[str]
    ) -> str:
        base_clean = base.strip()
        expansions: List[str] = []
        synonyms = {
            "policy": ["procedure", "guideline"],
            "spend": ["expense", "purchasing"],
            "supplier": ["vendor", "partner"],
            "card": ["credit card", "corporate card"],
            "approval": ["authorisation", "sign-off"],
            "limit": ["threshold", "cap"],
        }
        lowered = base_clean.lower()
        for token, extras in synonyms.items():
            if token in lowered:
                expansions.extend(extras)
        if continuation:
            detail = self._extract_followup_detail(continuation)
            if detail and detail not in base_clean:
                expansions.append(detail)
        unique_expansions = " ".join(dict.fromkeys(expansions))
        return f"{base_clean} {unique_expansions}".strip()

    def _looks_like_policy(self, query: str) -> bool:
        lowered = query.lower()
        policy_tokens = ["policy", "procedure", "credit card", "code of conduct", "expense"]
        return any(token in lowered for token in policy_tokens)

    def _is_follow_up_phrase(self, query: str) -> bool:
        lowered = query.lower().strip()
        if not lowered:
            return False
        if lowered.startswith("could you") and "elaborate" in lowered:
            return True
        if lowered.startswith("can you") and "elaborate" in lowered:
            return True
        if lowered.startswith("what about") or lowered.startswith("how about"):
            return True
        if "more detail" in lowered or "more details" in lowered:
            return True
        return False

