# ProcWise/services/model_selector.py

import hashlib
import inspect
import json
import logging
import re
import ollama
import pdfplumber
from io import BytesIO
from pathlib import Path
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from sentence_transformers import CrossEncoder
from config.settings import settings
from qdrant_client import models
from agents.base_agent import AgentStatus
from agents.rag_agent import RAGAgent
from .rag_service import RAGService
from utils.gpu import configure_gpu, load_cross_encoder

_CITATION_GUIDELINES_PATH = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "training"
    / "rag"
    / "citation_guidelines.json"
)

logger = logging.getLogger(__name__)

configure_gpu()


class ChatHistoryManager:
    """Manages chat history using an AWS S3 bucket."""

    def __init__(self, s3_client, bucket_name):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = 'chat_history/'

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        key = f"{self.prefix}{user_id}.json"
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            history: List[Dict[str, Any]] = json.loads(obj['Body'].read().decode('utf-8'))
            # Ensure answers are JSON-serialisable. Non-string primitives are cast to strings
            # while structured data (dicts/lists) is preserved for downstream consumers.
            for item in history:
                ans = item.get("answer")
                if ans is not None and not isinstance(ans, (str, list, dict)):
                    item["answer"] = str(ans)
            return history
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return []
            logger.error(f"S3 get_object error for key {key}: {e}")
            raise

    def save_history(self, user_id: str, history: List):
        key = f"{self.prefix}{user_id}.json"
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=json.dumps(history, indent=2))
        except Exception as e:
            logger.error(f"S3 put_object error for key {key}: {e}")


class RAGPipeline:
    _BLOCKED_DOC_TYPE_TOKENS = ("learning", "workflow", "event", "log", "trace", "audit")
    _SENSITIVE_EXACT_KEYS = {
        "record_id",
        "source",
        "source_system",
        "source_agent",
        "source_event",
        "source_name",
        "filename",
        "file_name",
        "file_path",
        "path",
        "uri",
        "unique_id",
        "workflow_ref",
        "workflow_reference",
        "effective_date",
        "supplier",
        "doc_version",
        "round_id",
    }
    _SENSITIVE_KEY_MARKERS = (
        "workflow",
        "session",
        "event",
        "agent",
        "log",
        "trace",
        "dispatch",
        "routing",
        "draft",
        "message",
    )
    _BLOCKED_PAYLOAD_KEYS = {
        "workflow_id",
        "session_reference",
        "workflow_reference",
        "workflow_run_id",
        "workflow_execution_id",
        "workflow_event",
        "workflow_stage",
        "workflow_step",
        "workflow_state",
        "workflow_status",
        "workflow_signature",
        "workflow_payload",
        "workflow_context",
        "event_type",
        "event_name",
        "event_payload",
        "event_id",
        "event_reference",
        "message_id",
        "message_reference",
        "routing_key",
        "routing_event",
        "trace_id",
        "log_payload",
        "log_level",
        "log_message",
        "agent_name",
        "agent_id",
        "agent_reference",
        "learning_context",
        "learning_summary",
        "learning_id",
        "dispatch_id",
        "email_thread_id",
    }

    def __init__(self, agent_nick, cross_encoder_cls: Type[CrossEncoder] = CrossEncoder):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.history_manager = ChatHistoryManager(agent_nick.s3_client, agent_nick.settings.s3_bucket_name)
        default_rag_model = getattr(self.settings, "rag_model", None)
        if not default_rag_model:
            default_rag_model = getattr(self.settings, "extraction_model", settings.extraction_model)
        self.default_llm_model = default_rag_model
        self._static_agent = RAGAgent(agent_nick)
        threshold = getattr(self.settings, "static_qa_confidence_threshold", 0.68)
        self._static_confidence_threshold = max(0.0, min(float(threshold), 1.0))
        self.rag = RAGService(agent_nick)
        model_name = getattr(
            self.settings,
            "reranker_model",
            "BAAI/bge-reranker-large",
        )
        self._reranker = load_cross_encoder(
            model_name, cross_encoder_cls, getattr(self.agent_nick, "device", None)
        )
        self._citation_guidelines = self._load_citation_guidelines()

    def _format_static_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return ""
        return f"Sure, I can help with that. {cleaned}" if not cleaned.lower().startswith("sure") else cleaned

    def _try_static_answer(self, query: str, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            output = self._static_agent.run(query=query, user_id=user_id, session_id=user_id)
        except Exception:
            logger.exception("Static procurement QA lookup failed")
            return None

        if output.status is not AgentStatus.SUCCESS:
            return None

        confidence = float(output.confidence or 0.0)
        if confidence < self._static_confidence_threshold:
            return None

        payload = output.data or {}
        answer_text = payload.get("answer")
        if not isinstance(answer_text, str) or not answer_text.strip():
            return None

        formatted_answer = self._format_static_answer(answer_text)
        follow_ups = [
            item.strip()
            for item in (payload.get("related_prompts") or [])
            if isinstance(item, str) and item.strip()
        ][:3]

        history = self.history_manager.get_history(user_id)
        history.append({"query": query, "answer": formatted_answer})
        self.history_manager.save_history(user_id, history)

        retrieved = {
            "source": "static_procurement_qa",
            "topic": payload.get("topic"),
            "question": payload.get("question"),
            "confidence": confidence,
        }

        return {
            "answer": formatted_answer,
            "follow_ups": follow_ups,
            "retrieved_documents": [retrieved],
        }

    def _extract_text_from_uploads(self, files: List[tuple[bytes, str]]):
        """Return extracted text and lightweight summaries for uploaded PDFs."""
        results: List[Dict[str, str]] = []
        for content_bytes, filename in files:
            try:
                if filename.lower().endswith('.pdf'):
                    with pdfplumber.open(BytesIO(content_bytes)) as pdf:
                        text = "\n".join(
                            page.extract_text() for page in pdf.pages if page.extract_text()
                        )
                        summary = self._condense_snippet(text, max_sentences=4, max_chars=500)
                        results.append({
                            "name": filename,
                            "text": text,
                            "summary": summary,
                        })
            except Exception as e:
                logger.error(f"Failed to process uploaded file {filename}: {e}")
        return results

    def _rerank_search(self, query: str, hits: List, top_k: int = 5):
        """Re-rank search hits using a cross-encoder for improved accuracy."""
        if not hits:
            return []
        pairs = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            summary = payload.get("summary") or payload.get("text_summary")
            text = summary or payload.get("content", "")
            pairs.append((query, text))
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Context synthesis helpers
    # ------------------------------------------------------------------
    def _load_citation_guidelines(self) -> Dict[str, Any]:
        """Load structured response preferences authored by layered training."""

        defaults: Dict[str, Any] = {
            "acknowledgements": [
                "Here's what I found in the policy documentation.",
                "Here's a concise summary from the knowledge base.",
            ],
            "summary_intro": "These points capture the essentials you should know.",
            "actions_lead": "Let me know if you want me to escalate, refresh the data, or prep outreach notes.",
            "fallback": "I do not have that information as per my knowledge.",
            "default_follow_ups": [
                "Would you like me to surface the related purchase order or contract details?",
                "Should I queue a supplier relationship summary for review?",
                "Do you want me to capture this question for the next procurement sync?",
            ],
        }
        if not _CITATION_GUIDELINES_PATH.exists():
            try:
                _CITATION_GUIDELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CITATION_GUIDELINES_PATH.write_text(
                    json.dumps(defaults, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("Unable to persist default citation guidelines", exc_info=True)
            return defaults

        try:
            loaded = json.loads(
                _CITATION_GUIDELINES_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            logger.warning("Failed to load citation guidelines; using defaults")
            return defaults

        merged = dict(defaults)
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key in {"acknowledgements", "default_follow_ups"} and isinstance(value, list):
                    merged[key] = [str(item) for item in value if str(item).strip()]
                elif isinstance(value, str):
                    merged[key] = value
        return merged

    def _redact_identifiers(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        patterns = [
            r"\b(?:supplier|rfq|po|invoice|contract)[-_\s]*\w+\b",
            r"PROC-?WF-\w+",
            r"\b(?:ID|Ref)[-:\s]*\d+\b",
            r"\bworkflow[-_ ]?(?:id|run|ref|context)?[-:=\s]*[A-Za-z0-9_-]{4,}\b",
            r"\b(?:session|event|trace|dispatch|message)[-_: ]*(?:id|ref)?[-:=\s]*[A-Za-z0-9_-]{4,}\b",
            r"\b[a-z]+_agent\b",
            r"\blearning[_-]?(?:event|record|entry)?\b",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "a sensitive identifier",
                cleaned,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", cleaned).strip()

    def _condense_snippet(
        self,
        text: str,
        *,
        max_sentences: int = 2,
        max_chars: int = 280,
    ) -> str:
        """Return a lightly summarised version of ``text`` suitable for prompts."""

        if not isinstance(text, str):
            return ""

        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(sentences[:max_sentences]) if sentences else cleaned
        if len(summary) > max_chars:
            truncated = summary[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;")
            summary = f"{truncated}…" if truncated else summary[:max_chars]
        return self._redact_identifiers(summary)

    def _extract_focus_phrase(self, query: str, max_words: int = 8) -> str:
        if not isinstance(query, str):
            return ""

        trimmed = re.sub(r"\s+", " ", query).strip()
        if not trimmed:
            return ""

        lowered = trimmed.lower()
        for token in (" about ", " regarding ", " on ", " for ", " concerning "):
            idx = lowered.find(token)
            if idx != -1 and idx + len(token) < len(trimmed):
                candidate = trimmed[idx + len(token) :].strip(" ?.!;:")
                if candidate:
                    trimmed = candidate
                    lowered = trimmed.lower()
                    break

        trimmed = trimmed.lstrip("? ")
        words = re.findall(r"[A-Za-z0-9'/-]+", trimmed)
        if not words:
            return ""

        stopwords = {
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "do",
            "does",
            "did",
            "please",
            "kindly",
            "let",
            "help",
            "need",
            "want",
            "looking",
            "look",
            "tell",
            "give",
            "provide",
            "share",
            "explain",
            "clarify",
            "me",
            "us",
            "our",
            "my",
            "their",
            "the",
            "a",
            "an",
            "any",
            "some",
            "current",
            "latest",
            "on",
            "for",
            "about",
            "regarding",
            "of",
            "to",
            "with",
            "in",
            "and",
            "or",
            "vs",
            "versus",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "into",
            "from",
            "can",
            "could",
            "would",
            "should",
            "will",
            "shall",
        }

        filtered: List[str] = []
        for word in words:
            if word.lower() in stopwords:
                continue
            filtered.append(word)
            if len(filtered) >= max_words:
                break

        if not filtered:
            filtered = words[:max_words]

        return " ".join(filtered)

    def _topic_descriptor(self, topic: str) -> str:
        if not topic:
            return ""

        cleaned = re.sub(r"\s+", " ", topic).strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        if lowered.startswith(
            (
                "the ",
                "this ",
                "that ",
                "these ",
                "those ",
                "any ",
                "my ",
                "our ",
                "your ",
                "a ",
                "an ",
            )
        ):
            descriptor = cleaned
        else:
            descriptor = f"the {cleaned}"

        if "sensitive identifier" in descriptor.lower():
            return "this request"

        return descriptor

    def _extract_snippet(self, item: Dict[str, Any]) -> str:
        summary = item.get("summary") if isinstance(item, dict) else ""
        if isinstance(summary, str) and summary.strip():
            return self._redact_identifiers(summary.strip())
        payload = item.get("payload") if isinstance(item, dict) else {}
        if isinstance(payload, dict):
            candidate = payload.get("content") or payload.get("text_summary") or ""
        else:
            candidate = ""
        return self._condense_snippet(candidate)

    def _friendly_opening(
        self,
        query: str,
        acknowledgements: List[str],
        focus_items: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        banned_tokens = (
            "thanks for flagging",
            "here's what i can confirm",
        )
        focus_phrase = self._extract_focus_phrase(query)
        if not focus_phrase and focus_items:
            for item in focus_items:
                candidate = item.get("document") or item.get("source_label")
                if isinstance(candidate, str) and candidate.strip():
                    focus_phrase = candidate
                    break

        focus_phrase = self._redact_identifiers(focus_phrase)
        topic_descriptor = self._topic_descriptor(focus_phrase)

        options: List[str] = []
        for ack in acknowledgements:
            if not ack:
                continue
            lowered = ack.lower()
            if any(token in lowered for token in banned_tokens):
                continue
            options.append(ack.strip())

        if topic_descriptor:
            personalised: List[str] = []
            for template in options:
                if "{topic}" in template:
                    personalised.append(
                        template.replace("{topic}", topic_descriptor)
                    )
                else:
                    base = template.rstrip(". ")
                    personalised.append(
                        f"{base} Let's focus on {topic_descriptor}."
                    )
            options = personalised
        else:
            fallback_topic = "this topic"
            options = [
                template.replace("{topic}", fallback_topic)
                if "{topic}" in template
                else template
                for template in options
            ]

        if not options:
            if topic_descriptor:
                options = [
                    f"Sure, I can help you with {topic_descriptor}.",
                    f"Most definitely, I'll help you get the answer on {topic_descriptor}.",
                    f"Great! Here's what the guidance says about {topic_descriptor}.",
                    f"You're in the right place for questions about {topic_descriptor}.",
                ]
            else:
                options = [
                    "Sure, I can help you with this topic.",
                    "Most definitely, I'll help you get the answer you need.",
                    "Great! Here is the response based on what I can see.",
                    "You're in the right place for this question.",
                ]

        digest = hashlib.sha256((query or "").encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(options)
        return options[index]

    def _select_focus_items(
        self, items: List[Dict[str, Any]], limit: int = 4
    ) -> List[Dict[str, Any]]:
        if not items or limit <= 0:
            return []

        sortable: List[Dict[str, Any]] = []
        for item in items:
            score = item.get("score")
            if not isinstance(score, (int, float)):
                score = 0.0
            item_with_score = dict(item)
            item_with_score["_ordering_score"] = float(score)
            sortable.append(item_with_score)

        sortable.sort(
            key=lambda entry: entry.get("_ordering_score", 0.0), reverse=True
        )

        selected: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for entry in sortable:
            doc_label = str(entry.get("document") or "").lower()
            collection = str(entry.get("collection") or "")
            key = (doc_label, collection)
            if key in seen:
                continue
            cleaned_entry = dict(entry)
            cleaned_entry.pop("_ordering_score", None)
            selected.append(cleaned_entry)
            seen.add(key)
            if len(selected) >= limit:
                break

        if len(selected) < min(limit, len(sortable)):
            for entry in sortable:
                candidate = dict(entry)
                candidate.pop("_ordering_score", None)
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= limit:
                    break

        def _original_index(value: Dict[str, Any]) -> int:
            for idx, item in enumerate(items):
                if all(item.get(key) == value.get(key) for key in item.keys()):
                    return idx
            return len(items)

        selected.sort(key=_original_index)
        return selected[:limit]

    def _format_primary_statement(self, item: Dict[str, Any]) -> str:
        snippet = self._extract_snippet(item)
        if not snippet:
            return ""
        doc_label = item.get("document") or item.get("source_label") or "the referenced guidance"
        doc_label_str = str(doc_label).strip()
        lowered_doc = doc_label_str.lower()
        lowered_snippet = snippet.lower()
        if doc_label_str and lowered_doc in lowered_snippet:
            body = f"The answer for your query is that {snippet}"
        elif doc_label_str:
            body = f"The answer for your query is that {doc_label_str} explains {snippet}"
        else:
            body = f"The answer for your query is that {snippet}"
        return self._to_sentence(body)

    def _format_support_statement(self, item: Dict[str, Any], position: int) -> str:
        snippet = self._extract_snippet(item)
        if not snippet:
            return ""
        connectors = [
            "Additionally",
            "It also clarifies",
            "Finally",
        ]
        connector = connectors[min(position - 1, len(connectors) - 1)]
        doc_label = item.get("document") or item.get("source_label") or "the same guidance"
        doc_label_str = str(doc_label).strip()
        lowered_doc = doc_label_str.lower()
        lowered_snippet = snippet.lower()
        if doc_label_str and lowered_doc in lowered_snippet:
            body = f"{connector}, {snippet}"
        elif doc_label_str:
            body = f"{connector}, {doc_label_str} notes {snippet}"
        else:
            body = f"{connector}, {snippet}"
        return self._to_sentence(body)

    def _craft_summary_intro(
        self,
        query: str,
        template: str,
        focus_items: List[Dict[str, Any]],
    ) -> str:
        topic = self._extract_focus_phrase(query)
        if not topic:
            for item in focus_items:
                candidate = (
                    item.get("document")
                    or item.get("source_label")
                    or item.get("collection")
                )
                if isinstance(candidate, str) and candidate.strip():
                    topic = candidate
                    break

        topic = self._redact_identifiers(topic)
        descriptor = self._topic_descriptor(topic) if topic else "this topic"
        base = (template or "Here’s what stands out about {topic}:").strip()
        redacted_query = self._redact_identifiers(query)
        if "{query}" in base:
            base = base.replace("{query}", redacted_query or "your question")

        if "{topic}" in base:
            line = base.replace("{topic}", descriptor)
        elif descriptor:
            cleaned_base = base.rstrip(" :")
            if cleaned_base.lower().endswith("about"):
                line = f"{cleaned_base} {descriptor}"
            else:
                line = f"{cleaned_base} about {descriptor}"
        else:
            line = base

        source_hint = ""
        for item in focus_items:
            candidate = item.get("source_label") or item.get("collection")
            if isinstance(candidate, str) and candidate.strip():
                source_hint = self._redact_identifiers(candidate)
                break

        if source_hint:
            lowered_line = line.lower()
            if source_hint.lower() not in lowered_line:
                line = f"{line.rstrip('.')} drawing from {source_hint}"

        return line.rstrip(" :")

    def _analyse_session_history(
        self, history: List[Dict[str, Any]]
    ) -> tuple[str, List[str], bool]:
        if not history:
            return "", [], False

        recent_entries = history[-3:]
        hint_parts: List[str] = []
        fragments: List[str] = []
        policy_signal = False

        for entry in reversed(recent_entries):
            question = str(entry.get("query", "")).strip()
            if question:
                cleaned_question = self._redact_identifiers(question)
                if cleaned_question:
                    hint_parts.append(f"Earlier question: {cleaned_question}")
                    lowered_q = cleaned_question.lower()
                    if any(token in lowered_q for token in ("policy", "supplier", "approval")):
                        policy_signal = True

            answer = entry.get("answer")
            answer_text: str = ""
            if isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, list):
                answer_text = " ".join(str(item) for item in answer if item)
            elif isinstance(answer, dict):
                answer_text = json.dumps(answer, ensure_ascii=False)
            elif answer is not None:
                answer_text = str(answer)

            snippet = self._condense_snippet(
                answer_text, max_sentences=2, max_chars=220
            )
            if snippet:
                fragments.append(snippet)
                lowered_snippet = snippet.lower()
                if any(token in lowered_snippet for token in ("policy", "supplier", "approval")):
                    policy_signal = True

        session_hint = " ".join(hint_parts[:2])
        return session_hint, fragments[:3], policy_signal

    def _is_policy_question(
        self,
        query: str,
        doc_type: Optional[str],
        session_hint: str,
        history_policy_signal: bool,
    ) -> bool:
        tokens = [query or "", session_hint or ""]
        if doc_type:
            tokens.append(doc_type)
        combined = " ".join(token for token in tokens if token).lower()
        keyword_triggers = (
            "policy",
            "approved supplier",
            "non-approved",
            "maverick",
            "delegation",
            "compliance",
            "procurement rule",
            "sourcing rule",
            "policy breach",
        )
        if any(keyword in combined for keyword in keyword_triggers):
            return True
        return history_policy_signal

    def _supported_search_kwargs(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(self.rag.search)
        except (TypeError, ValueError):
            return {}

        parameters = signature.parameters
        has_var_kw = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        accepted: Dict[str, Any] = {}
        for key, value in candidate.items():
            param = parameters.get(key)
            if param is not None:
                if param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    accepted[key] = value
            elif has_var_kw:
                accepted[key] = value
        return accepted

    def _is_sensitive_key(self, key: Any) -> bool:
        key_str = str(key or "").strip()
        if not key_str:
            return True
        lowered = key_str.lower()
        if lowered in self._SENSITIVE_EXACT_KEYS:
            return True
        return any(marker in lowered for marker in self._SENSITIVE_KEY_MARKERS)

    def _sanitise_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_identifiers(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            cleaned = [self._sanitise_value(item) for item in value if item is not None]
            return [item for item in cleaned if item not in ("", [], {})]
        if isinstance(value, dict):
            return {
                str(key): self._sanitise_value(val)
                for key, val in value.items()
                if not self._is_sensitive_key(key)
            }
        return self._redact_identifiers(str(value))

    def _sanitise_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in (payload or {}).items():
            if self._is_sensitive_key(key):
                continue
            cleaned[str(key)] = self._sanitise_value(value)
        return cleaned

    def _is_internal_payload(
        self, payload: Dict[str, Any], collection: Optional[str]
    ) -> bool:
        if collection and collection == getattr(self.rag, "learning_collection", None):
            return True
        doc_type = str(payload.get("document_type") or "").lower()
        if any(token in doc_type for token in self._BLOCKED_DOC_TYPE_TOKENS):
            return True
        for key in payload.keys():
            lowered = str(key or "").lower()
            if lowered in self._BLOCKED_PAYLOAD_KEYS:
                return True
            if any(
                marker in lowered
                for marker in (
                    "workflow_",
                    "session_",
                    "event_",
                    "agent_",
                    "learning_",
                    "dispatch_",
                    "trace_",
                    "proc_wf",
                )
            ):
                return True
        return False

    def _to_sentence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    def _label_for_collection(self, collection: Optional[str]) -> str:
        if not collection:
            return "Knowledge base insight"
        if collection == self.rag.primary_collection:
            return "ProcWise knowledge base"
        if collection == self.rag.uploaded_collection:
            return "Uploaded reference"
        static_policy_collection = getattr(self.rag, "static_policy_collection", None)
        if collection == static_policy_collection:
            return "Procurement policy"
        if collection == "static_procurement_qa":
            return "Static procurement guidance"
        return collection.replace("_", " ").title()

    def _prepare_knowledge_items(self, hits: List) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            payload = dict(payload)
            collection = payload.get("collection_name", self.rag.primary_collection)
            if self._is_internal_payload(payload, collection):
                continue
            payload = self._sanitise_payload(payload)
            collection = payload.get("collection_name", collection)
            source_label = self._label_for_collection(collection)
            payload.setdefault("collection_name", collection)
            payload.setdefault("source_label", source_label)
            combined_raw = getattr(hit, "combined_score", getattr(hit, "aggregated", 0.0))
            rerank_raw = getattr(hit, "rerank_score", getattr(hit, "score", 0.0))
            combined_score = float(combined_raw) if isinstance(combined_raw, (int, float)) else 0.0
            rerank_score = float(rerank_raw) if isinstance(rerank_raw, (int, float)) else 0.0
            summary_text = (
                payload.get("summary")
                or payload.get("text_summary")
                or payload.get("content", "")
            )
            condensed = self._condense_snippet(summary_text)
            document_label = (
                payload.get("document_name")
                or payload.get("title")
                or payload.get("document_type")
                or "Procurement reference"
            )
            document_label = self._redact_identifiers(str(document_label))
            items.append(
                {
                    "payload": payload,
                    "collection": collection,
                    "source_label": source_label,
                    "document": document_label,
                    "summary": condensed,
                    "score": combined_score,
                    "rerank_score": rerank_score,
                }
            )
        return items

    def _synthesise_context(self, knowledge_items: List[Dict[str, Any]]) -> str:
        if not knowledge_items:
            return ""

        ordered_collections: List[str] = []
        for collection in (
            self.rag.primary_collection,
            self.rag.uploaded_collection,
            getattr(self.rag, "static_policy_collection", None),
            "static_procurement_qa",
        ):
            if any(item["collection"] == collection for item in knowledge_items):
                ordered_collections.append(collection)

        # Include any other collections deterministically afterwards
        for item in knowledge_items:
            if item["collection"] not in ordered_collections:
                ordered_collections.append(item["collection"])

        paragraphs: List[str] = []
        for collection in ordered_collections:
            relevant = [item for item in knowledge_items if item["collection"] == collection]
            if not relevant:
                continue
            source_label = self._label_for_collection(collection)
            snippets: List[str] = []
            for idx, item in enumerate(relevant):
                snippet = item.get("summary") or self._condense_snippet(
                    item["payload"].get("content", "")
                )
                if not snippet:
                    continue
                doc_label = item.get("document")
                if idx == 0:
                    if doc_label:
                        snippets.append(f"{doc_label} shows {snippet}")
                    else:
                        snippets.append(snippet)
                else:
                    connector = "Additionally" if idx == 1 else "Meanwhile"
                    if doc_label:
                        snippets.append(f"{connector}, {doc_label} adds that {snippet}")
                    else:
                        snippets.append(f"{connector}, {snippet}")
            if not snippets:
                continue
            paragraphs.append(f"{source_label}: {' '.join(snippets)}")

        return "\n\n".join(paragraphs)

    def _build_structured_answer(
        self,
        query: str,
        enumerated_items: List[Dict[str, Any]],
        ad_hoc_context: str,
    ) -> str:
        guidelines = self._citation_guidelines
        acknowledgements = [
            str(item).strip()
            for item in guidelines.get("acknowledgements", [])
            if str(item).strip()
        ]
        summary_intro_template = str(
            guidelines.get(
                "summary_intro",
                "Here's a quick summary based on the latest procurement guidance.",
            )
        ).strip()

        focus_items = self._select_focus_items(enumerated_items)

        paragraphs: List[str] = []
        opening_line = self._friendly_opening(query, acknowledgements, focus_items)
        if opening_line:
            paragraphs.append(self._to_sentence(opening_line))

        if (focus_items or ad_hoc_context) and summary_intro_template:
            intro_line = self._craft_summary_intro(
                query,
                summary_intro_template,
                focus_items,
            )
            if intro_line:
                paragraphs.append(self._to_sentence(intro_line))
        if focus_items:
            statements: List[str] = []
            for idx, item in enumerate(focus_items):
                if idx == 0:
                    sentence = self._format_primary_statement(item)
                else:
                    sentence = self._format_support_statement(item, idx)
                if sentence:
                    statements.append(sentence)
            if statements:
                paragraphs.append(" ".join(statements))

        if ad_hoc_context:
            uploads_sentence = self._to_sentence(
                f"I also reviewed your uploaded notes: {self._redact_identifiers(ad_hoc_context)}"
            )
            if uploads_sentence:
                paragraphs.append(uploads_sentence)

        actions_lead = guidelines.get("actions_lead")
        if actions_lead:
            final_sentence = self._to_sentence(actions_lead)
            if final_sentence:
                paragraphs.append(final_sentence)

        return "\n\n".join(paragraphs)

    def _build_followups(
        self, query: str, enumerated_items: List[Dict[str, Any]]
    ) -> List[str]:
        redacted_query = self._redact_identifiers(query)
        suggestions: List[str] = []
        if redacted_query:
            suggestions.append(
                f'Should I gather further detail on "{redacted_query}"?'
            )
        if enumerated_items:
            top_doc = enumerated_items[0].get("document")
            if top_doc:
                suggestions.append(
                    f"Would you like me to brief stakeholders responsible for {top_doc}?"
                )
        for template in self._citation_guidelines.get("default_follow_ups", []):
            replacement = template.replace("{query}", redacted_query)
            suggestions.append(replacement)

        deduped: List[str] = []
        for suggestion in suggestions:
            cleaned = suggestion.strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped[:3]

    def _format_history_context(self, history: List[Dict[str, Any]], limit: int = 3) -> str:
        if not history:
            return ""
        trimmed = history[-limit:]
        formatted = []
        for item in trimmed:
            question = str(item.get("query", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question and not answer:
                continue
            formatted.append(f"Question: {question}\nAnswer: {answer}")
        return "\n\n".join(formatted)

    def _postprocess_answer(self, answer: str) -> str:
        if not isinstance(answer, str):
            return str(answer)

        cleaned = answer.strip()
        if not cleaned:
            return ""

        lower = cleaned.lower()
        if lower.startswith("as an ai"):
            cleaned = cleaned.split(".", 1)[-1].lstrip() or cleaned
        if lower.startswith("based on"):
            cleaned = f"From what I can see, {cleaned[len('based on'):].lstrip()}"

        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _generate_response(self, prompt: str, model: str) -> Dict:
        """Calls :func:`ollama.chat` once to get answer and follow-ups."""
        system = (
            "You are ProcWise's procurement intelligence analyst. Respond in valid JSON with keys 'answer' and 'follow_ups'. "
            "Write the answer as two or three short paragraphs using confident, conversational business English. "
            "Blend factual details with procedural guidance drawn from the supplied context, acknowledge any gaps, and suggest practical next steps. "
            "Never reference system internals, raw identifiers, or database terminology. "
            "Ensure 'follow_ups' contains three concise, forward-looking questions that keep the procurement dialogue moving."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            base_options = dict(self.agent_nick.ollama_options() or {})
        except Exception:
            base_options = {}
        temperature = base_options.get("temperature", 0.0)
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = 0.0
        if temperature <= 0.0:
            base_options["temperature"] = 0.35
        else:
            base_options["temperature"] = min(0.7, temperature + 0.1)
        base_options.setdefault("top_p", 0.9)
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options=base_options,
                format="json",
            )
            content = response.get("message", {}).get("content", "")
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {"answer": content.strip() or "Could not generate an answer.", "follow_ups": []}
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"answer": "Could not generate an answer.", "follow_ups": []}

    def answer_question(
        self,
        query: str,
        user_id: str,
        model_name: Optional[str] = None,
        files: Optional[List[tuple[bytes, str]]] = None,
        doc_type: Optional[str] = None,
        product_type: Optional[str] = None,
    ) -> Dict:
        llm_to_use = model_name or self.default_llm_model
        logger.info(
            "Answering query with model '%s' and filters: doc_type='%s', product_type='%s'",
            llm_to_use,
            doc_type,
            product_type,
        )

        history = self.history_manager.get_history(user_id)
        session_hint, memory_fragments, history_policy_signal = self._analyse_session_history(
            history
        )

        must_conditions: List[models.FieldCondition] = []
        if doc_type:
            normalised_doc_type = doc_type.lower()
            must_conditions.append(
                models.FieldCondition(
                    key="document_type", match=models.MatchValue(value=normalised_doc_type)
                )
            )
        else:
            normalised_doc_type = None
        if product_type:
            product_type = product_type.lower()
            must_conditions.append(
                models.FieldCondition(
                    key="product_type", match=models.MatchValue(value=product_type)
                )
            )
        qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

        uploaded = self._extract_text_from_uploads(files) if files else []
        ad_hoc_notes: List[str] = []
        for idx, file_info in enumerate(uploaded):
            filename = file_info.get("name") or f"uploaded-reference-{idx + 1}"
            safe_name = self._redact_identifiers(filename)
            text = file_info.get("text", "")
            summary = file_info.get("summary") or self._condense_snippet(text)
            if summary:
                ad_hoc_notes.append(f"{safe_name}: {summary}")
            metadata = {"record_id": filename, "document_type": normalised_doc_type or "uploaded"}
            if product_type:
                metadata["product_type"] = product_type
            if text:
                self.rag.upsert_texts([text], metadata)
        ad_hoc_context = "\n".join(ad_hoc_notes)

        policy_mode = self._is_policy_question(
            query, doc_type, session_hint, history_policy_signal
        )
        search_kwargs = {
            "top_k": 6,
            "filters": qdrant_filter,
        }
        hint_kwargs = {
            "session_hint": session_hint,
            "memory_fragments": memory_fragments,
            "policy_mode": policy_mode,
        }
        search_kwargs.update(self._supported_search_kwargs(hint_kwargs))
        reranked = self.rag.search(query, **search_kwargs)
        knowledge_items = self._prepare_knowledge_items(reranked)

        if knowledge_items:
            try:
                static_output = self._static_agent.run(
                    query=query, user_id=user_id, session_id=user_id
                )
            except Exception:
                logger.exception("Static procurement QA lookup failed during context build")
            else:
                if static_output.status is AgentStatus.SUCCESS:
                    payload = static_output.data or {}
                    answer_text = payload.get("answer")
                    if isinstance(answer_text, str) and answer_text.strip():
                        safe_document = self._redact_identifiers(
                            payload.get("topic")
                            or payload.get("question")
                            or "Procurement guidance"
                        )
                        static_context_payload = {
                            "source": "static_procurement_qa",
                            "topic": payload.get("topic"),
                            "question": payload.get("question"),
                            "answer": answer_text,
                            "confidence": float(static_output.confidence or 0.0),
                            "collection_name": "static_procurement_qa",
                            "source_label": self._label_for_collection("static_procurement_qa"),
                        }
                        static_context_payload = self._sanitise_payload(
                            static_context_payload
                        )
                        static_context_payload.setdefault(
                            "collection_name", "static_procurement_qa"
                        )
                        static_context_payload.setdefault(
                            "source_label",
                            self._label_for_collection("static_procurement_qa"),
                        )
                        knowledge_items.append(
                            {
                                "payload": static_context_payload,
                                "collection": "static_procurement_qa",
                                "source_label": self._label_for_collection(
                                    "static_procurement_qa"
                                ),
                                "document": safe_document,
                                "summary": self._condense_snippet(
                                    answer_text, max_sentences=2, max_chars=320
                                ),
                            }
                        )

        if not knowledge_items and not ad_hoc_context:
            fallback = self._citation_guidelines.get(
                "fallback",
                "I'm sorry, but I couldn't find that information in the available knowledge base.",
            )
            history = self.history_manager.get_history(user_id)
            history.append({"query": self._redact_identifiers(query), "answer": fallback})
            self.history_manager.save_history(user_id, history)
            follow_ups = self._build_followups(query, [])
            return {
                "answer": fallback,
                "follow_ups": follow_ups,
                "retrieved_documents": [],
            }

        focus_items = self._select_focus_items(knowledge_items)
        enumerated_items: List[Dict[str, Any]] = [dict(item) for item in focus_items]

        retrieved_documents_payloads: List[Dict[str, Any]] = []
        for item in enumerated_items:
            payload = dict(item.get("payload", {}))
            payload = self._sanitise_payload(payload)
            payload.setdefault("collection_name", item.get("collection"))
            payload.setdefault("source_label", item.get("source_label"))
            retrieved_documents_payloads.append(payload)

        answer = self._build_structured_answer(query, enumerated_items, ad_hoc_context)
        follow_ups = self._build_followups(query, enumerated_items)

        history.append({"query": self._redact_identifiers(query), "answer": answer})
        self.history_manager.save_history(user_id, history)

        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "retrieved_documents": retrieved_documents_payloads,
        }
