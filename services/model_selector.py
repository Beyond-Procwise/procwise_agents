# ProcWise/services/model_selector.py

import copy
import hashlib
import inspect
import json
import logging
import re
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Set

from html import escape
import pdfplumber
from botocore.exceptions import ClientError
from sentence_transformers import CrossEncoder
from config.settings import settings
from qdrant_client import models
from agents.base_agent import AgentStatus
from agents.rag_agent import RAGAgent
from services.redis_client import get_redis_client
from services.lmstudio_client import (
    LMStudioClientError,
    get_lmstudio_client,
)
from .rag_service import RAGService
from .nltk_pipeline import NLTKProcessor
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
    """Manages chat history in S3 with a Redis-backed ephemeral cache."""

    _CACHE_KEY_PREFIX = "chat_history_cache:data:"
    _CACHE_INDEX_KEY = "chat_history_cache:index"

    def __init__(
        self,
        s3_client,
        bucket_name,
        *,
        cache_ttl: float = 0.0,
        max_cache_entries: int = 0,
        redis_client=None,
    ):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = 'chat_history/'
        self._cache_ttl = max(0.0, float(cache_ttl))
        self._max_cache_entries = max(0, int(max_cache_entries))
        self._cache_lock = threading.RLock()
        self._redis = redis_client or get_redis_client()
        self._cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}

    def _cache_enabled(self) -> bool:
        return self._cache_ttl > 0 and self._max_cache_entries > 0

    def _use_redis(self) -> bool:
        return self._redis is not None and self._cache_enabled()

    def _redis_cache_key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{self._CACHE_KEY_PREFIX}{digest}"

    def _evict_cache_entry(self, key: str) -> None:
        if not key or not self._cache_enabled():
            return
        if self._use_redis():
            redis_key = self._redis_cache_key(key)
            try:
                pipe = self._redis.pipeline()
                pipe.delete(redis_key)
                pipe.zrem(self._CACHE_INDEX_KEY, redis_key)
                pipe.execute()
            except Exception:
                logger.exception("Failed to evict chat history cache entry from Redis")
        else:
            with self._cache_lock:
                self._cache.pop(key, None)

    def _evict_redis_excess(self) -> None:
        if not self._use_redis():
            return
        try:
            current_size = self._redis.zcard(self._CACHE_INDEX_KEY) or 0
            if current_size <= self._max_cache_entries:
                return
            excess = int(current_size - self._max_cache_entries)
            if excess <= 0:
                return
            stale_keys = self._redis.zrange(self._CACHE_INDEX_KEY, 0, excess - 1) or []
            if not stale_keys:
                return
            pipe = self._redis.pipeline()
            pipe.delete(*stale_keys)
            pipe.zrem(self._CACHE_INDEX_KEY, *stale_keys)
            pipe.execute()
        except Exception:
            logger.exception("Failed to evict excess chat history cache entries from Redis")

    def _get_cached(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if not self._cache_enabled() or not key:
            return None
        if self._use_redis():
            redis_key = self._redis_cache_key(key)
            try:
                raw = self._redis.get(redis_key)
            except Exception:
                logger.exception("Failed to read chat history cache from Redis")
                return None
            if raw is None:
                return None
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except Exception:
                    self._evict_cache_entry(key)
                    return None
            try:
                payload = json.loads(raw)
            except Exception:
                self._evict_cache_entry(key)
                return None
            if not isinstance(payload, list):
                return []
            ttl_seconds = max(int(self._cache_ttl), 1)
            try:
                pipe = self._redis.pipeline()
                pipe.expire(redis_key, ttl_seconds)
                pipe.zadd(self._CACHE_INDEX_KEY, {redis_key: time.time()})
                pipe.expire(self._CACHE_INDEX_KEY, max(ttl_seconds * 2, ttl_seconds + 60))
                pipe.execute()
            except Exception:
                logger.exception("Failed to refresh chat history cache TTL in Redis")
            return copy.deepcopy(payload)

        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                return None
            deadline, payload = cached
            if deadline <= now:
                self._cache.pop(key, None)
                return None
            return copy.deepcopy(payload)

    def _store_cache(self, key: str, value: List[Dict[str, Any]]) -> None:
        if not self._cache_enabled() or not key:
            return
        if self._use_redis():
            try:
                payload = json.dumps(value)
            except Exception:
                logger.exception("Failed to serialise chat history for Redis cache")
                return
            redis_key = self._redis_cache_key(key)
            ttl_seconds = max(int(self._cache_ttl), 1)
            try:
                pipe = self._redis.pipeline()
                pipe.set(redis_key, payload, ex=ttl_seconds)
                pipe.zadd(self._CACHE_INDEX_KEY, {redis_key: time.time()})
                pipe.expire(self._CACHE_INDEX_KEY, max(ttl_seconds * 2, ttl_seconds + 60))
                pipe.execute()
            except Exception:
                logger.exception("Failed to write chat history cache entry to Redis")
                return
            self._evict_redis_excess()
            return

        deadline = time.monotonic() + self._cache_ttl
        snapshot = copy.deepcopy(value)
        with self._cache_lock:
            self._cache[key] = (deadline, snapshot)
            if len(self._cache) > self._max_cache_entries:
                # Drop the stalest entry to keep the cache bounded.
                oldest_key = min(self._cache.items(), key=lambda item: item[1][0])[0]
                if oldest_key != key:
                    self._cache.pop(oldest_key, None)

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        key = f"{self.prefix}{user_id}.json"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                history: List[Dict[str, Any]] = []
                self._store_cache(key, history)
                return []
            logger.error(f"S3 get_object error for key {key}: {e}")
            raise

        history = json.loads(obj['Body'].read().decode('utf-8'))
        if not isinstance(history, list):
            history = []
        # Ensure answers are JSON-serialisable. Non-string primitives are cast to strings
        # while structured data (dicts/lists) is preserved for downstream consumers.
        for item in history:
            if not isinstance(item, dict):
                continue
            ans = item.get("answer")
            if ans is not None and not isinstance(ans, (str, list, dict)):
                item["answer"] = str(ans)

        self._store_cache(key, history)
        return history

    def save_history(self, user_id: str, history: List):
        key = f"{self.prefix}{user_id}.json"
        try:
            payload = json.dumps(history, indent=2)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=payload)
        except Exception as e:
            logger.error(f"S3 put_object error for key {key}: {e}")
        else:
            if history:
                self._store_cache(key, list(history))
            else:
                self._evict_cache_entry(key)


class RAGPipeline:
    _BLOCKED_DOC_TYPE_TOKENS = ("learning", "workflow", "event", "log", "trace", "audit")
    _HTML_TAG_ALLOWLIST: Set[str] = {
        "a",
        "article",
        "b",
        "blockquote",
        "body",
        "br",
        "code",
        "div",
        "em",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "i",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "span",
        "strong",
        "sub",
        "sup",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
    }
    _IDENTIFIER_FIELD_KEYS = {
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
        "document_id",
        "documentid",
        "doc_id",
        "docid",
        "document_reference",
        "documentref",
        "workflow_ref",
        "workflow_reference",
        "workflow_id",
        "message_id",
        "metadata",
    }
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

    def __init__(
        self,
        agent_nick,
        cross_encoder_cls: Type[CrossEncoder] = CrossEncoder,
        *,
        use_nltk: bool = True,
    ):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        history_cache_ttl = getattr(self.settings, "chat_history_cache_ttl", 15.0)
        history_cache_size = getattr(self.settings, "chat_history_cache_max_entries", 256)
        self.history_manager = ChatHistoryManager(
            agent_nick.s3_client,
            agent_nick.settings.s3_bucket_name,
            cache_ttl=float(history_cache_ttl),
            max_cache_entries=int(history_cache_size),
        )
        default_rag_model = getattr(self.settings, "rag_model", None)
        fallback_default = default_rag_model or getattr(
            self.settings, "extraction_model", settings.extraction_model
        )
        resolver = getattr(self.agent_nick, "get_agent_model", None)
        if callable(resolver):
            try:
                default_rag_model = resolver(
                    "rag_pipeline", fallback=fallback_default
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Failed to resolve agent-specific model for RAG", exc_info=True)
                default_rag_model = fallback_default
        else:
            default_rag_model = fallback_default
        self.default_llm_model = self._ensure_phi4_default(default_rag_model)
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
        self._session_uploads: Dict[str, Dict[str, Any]] = {}
        self._uploaded_context: Optional[Dict[str, Any]] = None
        self._nltk_processor = NLTKProcessor() if use_nltk else None
        if self._nltk_processor and not getattr(self._nltk_processor, "available", False):
            self._nltk_processor = None
        self._cache_ttl = float(getattr(self.settings, "ask_cache_ttl_seconds", 90.0))
        self._cache_max_entries = int(getattr(self.settings, "ask_cache_max_entries", 32))
        self._answer_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_lock = threading.RLock()

    def _cache_enabled(self) -> bool:
        return self._cache_ttl > 0 and self._cache_max_entries > 0

    def _purge_expired_cache(self, *, now: Optional[float] = None) -> None:
        if not self._cache_enabled():
            return
        threshold = now if now is not None else time.monotonic()
        with self._cache_lock:
            expired_keys = [
                key for key, (deadline, _)
                in self._answer_cache.items()
                if deadline <= threshold
            ]
            for key in expired_keys:
                self._answer_cache.pop(key, None)

    def _stringify_for_cache(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return repr(value)

    def _prepare_metadata_for_cache(self, metadata: Any) -> List[str]:
        if not metadata:
            return []
        if isinstance(metadata, dict):
            entries = [
                f"{self._stringify_for_cache(key)}={self._stringify_for_cache(value)}"
                for key, value in metadata.items()
            ]
            entries.sort()
            return entries
        if isinstance(metadata, (list, tuple, set)):
            normalised = [self._stringify_for_cache(item) for item in metadata]
            normalised.sort()
            return normalised
        return [self._stringify_for_cache(metadata)]

    def _normalise_document_ids_for_cache(self, document_ids: Any) -> List[str]:
        results: List[str] = []
        for value in document_ids or []:
            text = self._stringify_for_cache(value).strip()
            if text:
                results.append(text)
        results.sort()
        return results

    def _build_upload_fingerprint(
        self,
        *,
        candidate_keys: Sequence[str],
        uploaded_scope: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        payload: Dict[str, Any] = {}
        session_entries: List[Dict[str, Any]] = []
        for key in candidate_keys:
            record = self._session_uploads.get(key)
            if not record:
                continue
            session_entries.append(
                {
                    "key": self._stringify_for_cache(key),
                    "documents": self._normalise_document_ids_for_cache(
                        record.get("document_ids")
                    ),
                    "metadata": self._prepare_metadata_for_cache(
                        record.get("metadata")
                    ),
                    "registered_at": self._stringify_for_cache(
                        record.get("registered_at")
                    ),
                }
            )
        if session_entries:
            payload["session_uploads"] = session_entries

        context = uploaded_scope or {}
        if context:
            payload["uploaded_context"] = {
                "session": self._stringify_for_cache(context.get("session_id")),
                "activated_at": self._stringify_for_cache(
                    context.get("activated_at")
                ),
                "documents": self._normalise_document_ids_for_cache(
                    context.get("document_ids")
                ),
                "metadata": self._prepare_metadata_for_cache(
                    context.get("metadata")
                ),
            }

        if not payload:
            return None

        try:
            normalised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        except Exception:
            normalised = repr(payload)
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def _build_history_fingerprint(
        self, history: Sequence[Dict[str, Any]]
    ) -> Optional[str]:
        if not history:
            return None
        window: Sequence[Dict[str, Any]] = history[-5:]
        payload = {
            "length": len(history),
            "tail": [],
        }
        for item in window:
            try:
                query = item.get("query")
            except AttributeError:
                query = None
            try:
                answer = item.get("answer")
            except AttributeError:
                answer = None
            payload["tail"].append(
                {
                    "query": self._stringify_for_cache(query)[:256],
                    "answer": self._stringify_for_cache(answer)[:256],
                }
            )
        try:
            serialised = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            serialised = repr(payload)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def _build_cache_key(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str],
        model_name: Optional[str],
        doc_type: Optional[str],
        product_type: Optional[str],
        *,
        context_fingerprint: Optional[str] = None,
        history_fingerprint: Optional[str] = None,
    ) -> str:
        payload = {
            "query": str(query or "").strip(),
            "user": str(user_id or "").strip(),
            "session": str(session_id).strip() if session_id is not None else "",
            "model": str(model_name).strip() if model_name is not None else "",
            "doc_type": str(doc_type).strip() if doc_type is not None else "",
            "product_type": str(product_type).strip() if product_type is not None else "",
            "uploads": str(context_fingerprint or "").strip(),
            "history": str(history_fingerprint or "").strip(),
        }
        normalised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not cache_key or not self._cache_enabled():
            return None
        now = time.monotonic()
        with self._cache_lock:
            entry = self._answer_cache.get(cache_key)
            if not entry:
                return None
            deadline, value = entry
            if deadline <= now:
                self._answer_cache.pop(cache_key, None)
                return None
            return copy.deepcopy(value)

    def _store_cached_response(self, cache_key: Optional[str], value: Dict[str, Any]) -> None:
        if not cache_key or not self._cache_enabled():
            return
        snapshot = copy.deepcopy(value)
        expires_at = time.monotonic() + max(self._cache_ttl, 0.0)
        with self._cache_lock:
            self._purge_expired_cache(now=time.monotonic())
            if len(self._answer_cache) >= self._cache_max_entries:
                oldest_key: Optional[str] = None
                oldest_deadline: Optional[float] = None
                for key, (deadline, _) in self._answer_cache.items():
                    if oldest_deadline is None or deadline < oldest_deadline:
                        oldest_key = key
                        oldest_deadline = deadline
                if oldest_key is not None:
                    self._answer_cache.pop(oldest_key, None)
            self._answer_cache[cache_key] = (expires_at, snapshot)

    def _render_html_answer(self, answer_text: str) -> str:
        return self._normalise_answer_html(answer_text)

    def _ensure_phi4_default(self, configured_model: Optional[str]) -> str:
        """Guarantee Joshi relies on phi4 (or a fine-tuned variant)."""

        candidate = (configured_model or "").strip()
        if candidate and "qwen3" in candidate.lower():
            return candidate
        if candidate:
            logger.warning(
                "Configured RAG model '%s' is not phi4; defaulting to phi4:latest for Joshi.",
                candidate,
            )
        return "phi4:latest"

    def register_session_upload(
        self,
        session_id: str,
        document_ids: Sequence[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record uploaded document identifiers for the active user session."""

        if not session_id or not document_ids:
            return

        cleaned_id = session_id.strip() if isinstance(session_id, str) else str(session_id)
        cleaned_id = cleaned_id.strip()
        if not cleaned_id:
            return

        filtered_docs = [doc for doc in document_ids if str(doc).strip()]
        if not filtered_docs:
            return

        try:
            timestamp = datetime.utcnow().isoformat(timespec="seconds")
        except Exception:
            timestamp = datetime.utcnow().isoformat()

        self._session_uploads[cleaned_id] = {
            "document_ids": list(filtered_docs),
            "metadata": dict(metadata or {}),
            "registered_at": timestamp,
        }

        self._set_uploaded_context(
            filtered_docs,
            metadata=metadata,
            session_id=cleaned_id,
        )

    def activate_uploaded_context(
        self,
        document_ids: Sequence[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Force the pipeline to prioritise ad-hoc uploaded documents."""

        self._set_uploaded_context(
            document_ids,
            metadata=metadata,
            session_id=session_id,
        )

    def clear_uploaded_context(self) -> None:
        """Disable prioritisation of uploaded document context."""

        self._uploaded_context = None

    def uploaded_context_active(self) -> bool:
        """Return ``True`` when uploaded document prioritisation is active."""

        record = self._uploaded_context or {}
        docs = record.get("document_ids")
        return bool(docs)

    def _set_uploaded_context(
        self,
        document_ids: Sequence[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        cleaned: List[str] = []
        for value in document_ids or []:
            try:
                text = str(value).strip()
            except Exception:
                text = ""
            if text:
                cleaned.append(text)

        if not cleaned:
            return

        if session_id is not None:
            try:
                session_token = str(session_id).strip()
            except Exception:
                session_token = ""
        else:
            session_token = ""

        try:
            activated_at = datetime.utcnow().isoformat(timespec="seconds")
        except Exception:
            activated_at = datetime.utcnow().isoformat()

        self._uploaded_context = {
            "document_ids": cleaned,
            "metadata": dict(metadata or {}),
            "session_id": session_token or None,
            "activated_at": activated_at,
        }

    _STATIC_CLOSINGS: Tuple[str, ...] = (
        "Let me know if you'd like supporting detail or next steps.",
        "Happy to pull the supporting policy or spend breakdown if that's useful.",
        "We can dig into related metrics whenever you need it.",
    )

    def _format_static_answer(
        self,
        answer: str,
        *,
        question: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return ""

        intro = self._craft_static_intro(question, topic)
        sentences = self._split_sentences(cleaned)
        softened = [self._soften_sentence(sentence) for sentence in sentences if sentence]
        if not softened:
            softened = [cleaned]

        primary = softened[0]
        remainder = [item for item in softened[1:] if item]

        if remainder:
            bullet_lines = "\n".join(f"- {item}" for item in remainder)
            narrative = f"{primary}\n{bullet_lines}"
        else:
            narrative = primary

        closing = self._pick_static_closing(question, topic)

        return f"{intro} {narrative}\n{closing}"

    def _craft_static_intro(
        self, question: Optional[str], topic: Optional[str]
    ) -> str:
        if question:
            subject = question.strip()
            if len(subject) > 120:
                subject = subject[:117].rstrip() + "..."
            return f'For your question "{subject}", here\'s the quick take.'
        if topic:
            subject = topic.strip()
            if subject:
                return f"Here's the quick take on {subject}."
        return "Here's what the procurement playbook highlights."

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9£])", text.strip())
        return [segment.strip() for segment in raw if segment and segment.strip()]

    def _soften_sentence(self, sentence: str) -> str:
        text = sentence.strip()
        if not text:
            return ""
        replacements = (
            (r"\bApproximately\b", "About"),
            (r"\bapproximately\b", "about"),
            (r"\bmainly\b", "largely"),
            (r"\bdue to\b", "thanks to"),
            (r"\bNon-claimable\b", "Non-claimable"),
            (r"\bnon-claimable\b", "non-claimable"),
        )
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        if not text.endswith(('.', '!', '?')):
            text = f"{text}."
        else:
            text = text.rstrip(" ")
        return text

    def _pick_static_closing(
        self, question: Optional[str], topic: Optional[str]
    ) -> str:
        options = self._STATIC_CLOSINGS
        key = (question or topic or "").strip().lower()
        if not key:
            return options[0]
        digest = hashlib.sha1(key.encode("utf-8")).digest()
        index = digest[0] % len(options)
        return options[index]

    def _try_static_answer(
        self, query: str, user_id: str, *, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            output = self._static_agent.run(
                query=query,
                user_id=user_id,
                session_id=session_id or user_id,
            )
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

        is_feedback_ack = bool(
            payload.get("structure_type") == "feedback_acknowledgment"
            or (
                isinstance(payload.get("feedback"), dict)
                and payload["feedback"].get("captured")
                and payload.get("style") == "acknowledgment"
            )
        )

        stripped_answer = answer_text.lstrip()
        structured = (
            bool(payload.get("structured"))
            or stripped_answer.startswith("##")
            or stripped_answer.startswith("<section")
            or "<section" in stripped_answer
            or is_feedback_ack
        )

        if structured:
            formatted_answer = answer_text.strip()
        else:
            formatted_answer = self._format_static_answer(
                answer_text,
                question=query,
                topic=payload.get("topic"),
            )
        follow_ups = [
            item.strip()
            for item in (payload.get("related_prompts") or [])
            if isinstance(item, str) and item.strip()
        ][:3]

        html_answer = self._normalise_answer_html(formatted_answer)
        history = self.history_manager.get_history(user_id)
        history.append({"query": query, "answer": html_answer})
        self.history_manager.save_history(user_id, history)

        html_answer = self._render_html_answer(formatted_answer)

        retrieved = {
            "source": "static_procurement_qa",
            "topic": payload.get("topic"),
            "question": payload.get("question"),
            "confidence": confidence,
        }

        return {
            "answer": html_answer,
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

    def _strip_metadata_terms(self, text: str) -> str:
        """Remove internal metadata phrases such as document identifiers."""

        if not isinstance(text, str):
            return ""

        cleaned = text
        metadata_patterns = (
            r"(?i)\bDocument\s+ID[:#\-\s]*[A-Za-z0-9._-]+\b",
            r"(?i)\bDocument\s+(?:Reference|Ref|Number)[:#\-\s]*[A-Za-z0-9._-]+\b",
            r"(?i)\bPolicy\s+(?:Reference|Ref|Number|Version)[:#\-\s]*[A-Za-z0-9._-]+\b",
            r"(?i)\bSee\s+Document\s+[A-Za-z0-9._-]+\b",
        )

        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        return cleaned.strip(" ,;-")

    def _redact_identifiers(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = self._strip_metadata_terms(text)
        patterns = (
            r"(?i)PROC-?WF-[A-Za-z0-9_-]+",
            r"(?i)\bworkflow[-_ ]?(?:id|run|ref|context)?[-:=\s]*[A-Za-z0-9_-]{4,}\b",
            r"(?i)\b(?:session|event|trace|dispatch|message)[-_: ]*(?:id|ref)?[-:=\s]*[A-Za-z0-9_-]{4,}\b",
            r"(?i)\b(?:supplier|rfq|po|invoice|contract)\s*(?:number|no\.?|id|reference)?[:#\- ]*[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b",
            r"(?i)\b[a-z0-9]+_agent\b",
            r"\b(?:[A-Z][a-z]+){1,4}Agent\b",
            r"(?i)\blearning[_-]?[A-Za-z0-9]+\b",
        )

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        return cleaned.strip(" ,;-")

    def _remove_placeholders(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(
            r"\[(?:redacted sensitive reference|doc\s*\d+|document\s*\d+|source\s*\d+)\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

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

        cleaned = self._strip_metadata_terms(text)
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
            return ""

        digest = hashlib.sha256((query or "").encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(options)
        return options[index]

    def _conversation_context_line(
        self, query: str, focus_items: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        redacted_query = self._redact_identifiers(query)
        if not redacted_query:
            return ""

        focus_phrase = self._extract_focus_phrase(query)
        if not focus_phrase and focus_items:
            for item in focus_items:
                candidate = (
                    item.get("document")
                    or item.get("source_label")
                    or item.get("collection")
                )
                if isinstance(candidate, str) and candidate.strip():
                    focus_phrase = candidate
                    break

        focus_phrase = self._redact_identifiers(focus_phrase)
        if focus_phrase:
            descriptor = self._topic_descriptor(focus_phrase)
            if descriptor:
                return f"You asked about {descriptor}, so here's what I found"
        return f"You asked: {redacted_query}. Here's what stood out"

    def _select_focus_items(
        self, items: List[Dict[str, Any]], limit: int = 4
    ) -> List[Dict[str, Any]]:
        if not items or limit <= 0:
            return []

        sortable: List[Dict[str, Any]] = []
        uploaded_collection = getattr(self.rag, "uploaded_collection", "")
        static_policy_collection = getattr(self.rag, "static_policy_collection", "")

        for item in items:
            score = item.get("score")
            if not isinstance(score, (int, float)):
                score = 0.0
            collection = str(item.get("collection") or "")
            if collection and collection == uploaded_collection:
                score += 0.9
            elif collection and collection == static_policy_collection:
                score += 0.35
            elif collection == "static_procurement_qa":
                score += 0.3
            item_with_score = dict(item)
            item_with_score["_ordering_score"] = float(score)
            sortable.append(item_with_score)

        sortable.sort(
            key=lambda entry: entry.get("_ordering_score", 0.0), reverse=True
        )

        selected: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()

        label_counts: Dict[Tuple[str, str], int] = {}
        for entry in sortable:
            doc_label = str(entry.get("document") or "").strip().lower()
            collection = str(entry.get("collection") or "")
            if not doc_label:
                continue
            key = (doc_label, collection)
            label_counts[key] = label_counts.get(key, 0) + 1

        generic_labels = {"", "procurement reference"}

        def _stable_identifier(entry: Dict[str, Any]) -> Optional[tuple[str, ...]]:
            payload = entry.get("payload")
            if isinstance(payload, dict):
                for candidate in (
                    "chunk_id",
                    "chunk",
                    "chunkId",
                    "chunk_index",
                    "id",
                    "document_id",
                    "documentId",
                    "reference_id",
                    "referenceId",
                    "hash",
                    "checksum",
                ):
                    value = payload.get(candidate)
                    if isinstance(value, (str, int)):
                        text = str(value).strip()
                        if text:
                            return ("payload", candidate.lower(), text.lower())
            doc_label = str(entry.get("document") or "").strip().lower()
            collection = str(entry.get("collection") or "")
            if not doc_label:
                return None
            if doc_label in generic_labels or "redacted" in doc_label:
                return None
            label_key = (doc_label, collection)
            if label_counts.get(label_key, 0) != 1:
                return None
            return ("label", collection, doc_label)

        for entry in sortable:
            key = _stable_identifier(entry)
            if key and key in seen:
                continue
            cleaned_entry = dict(entry)
            cleaned_entry["_focus_score"] = float(entry.get("_ordering_score", 0.0))
            cleaned_entry.pop("_ordering_score", None)
            selected.append(cleaned_entry)
            if key:
                seen.add(key)
            if len(selected) >= limit:
                break

        if len(selected) < min(limit, len(sortable)):
            for entry in sortable:
                candidate = dict(entry)
                candidate["_focus_score"] = float(entry.get("_ordering_score", 0.0))
                candidate.pop("_ordering_score", None)
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= limit:
                    break

        primary_collection = getattr(self.rag, "primary_collection", "")
        if primary_collection:
            has_primary_focus = any(
                item.get("collection") == primary_collection for item in selected
            )
            if not has_primary_focus:
                primary_candidates: List[Dict[str, Any]] = []
                for entry in sortable:
                    if entry.get("collection") != primary_collection:
                        continue
                    candidate = dict(entry)
                    candidate["_focus_score"] = float(entry.get("_ordering_score", 0.0))
                    candidate.pop("_ordering_score", None)
                    primary_candidates.append(candidate)
                if primary_candidates:
                    replacement = primary_candidates[0]
                    if len(selected) < limit:
                        selected.append(replacement)
                    else:
                        lowest_idx = min(
                            range(len(selected)),
                            key=lambda idx: selected[idx].get("_focus_score", 0.0),
                        )
                        selected[lowest_idx] = replacement

        def _original_index(value: Dict[str, Any]) -> int:
            for idx, item in enumerate(items):
                if all(item.get(key) == value.get(key) for key in item.keys()):
                    return idx
            return len(items)

        selected.sort(key=_original_index)
        trimmed = selected[:limit]
        for entry in trimmed:
            entry.pop("_focus_score", None)
        return trimmed

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
    ) -> tuple[str, List[str], Dict[str, bool]]:
        if not history:
            return "", [], {"policy": False, "supplier": False}

        recent_entries = history[-3:]
        hint_parts: List[str] = []
        fragments: List[str] = []
        signals = {"policy": False, "supplier": False}
        policy_tokens = ("policy", "policies", "compliance", "compliant", "approval", "approvals")
        supplier_tokens = ("supplier", "suppliers", "vendor", "vendors")

        for entry in reversed(recent_entries):
            question = str(entry.get("query", "")).strip()
            if question:
                cleaned_question = self._redact_identifiers(question)
                if cleaned_question:
                    hint_parts.append(f"Earlier question: {cleaned_question}")
                    lowered_q = cleaned_question.lower()
                    if any(token in lowered_q for token in policy_tokens):
                        signals["policy"] = True
                    if any(token in lowered_q for token in supplier_tokens):
                        signals["supplier"] = True

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
                if any(token in lowered_snippet for token in policy_tokens):
                    signals["policy"] = True
                if any(token in lowered_snippet for token in supplier_tokens):
                    signals["supplier"] = True

        session_hint = " ".join(hint_parts[:2])
        return session_hint, fragments[:3], signals

    def _is_policy_question(
        self,
        query: str,
        doc_type: Optional[str],
        session_hint: str,
        history_signals: Optional[Dict[str, bool]],
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
        history_flags = history_signals or {}
        return bool(history_flags.get("policy"))

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

    def _filter_identifier_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Remove document and system identifier fields while preserving content."""

        filtered: Dict[str, Any] = {}
        for key, value in (payload or {}).items():
            key_str = str(key or "").strip()
            if not key_str:
                continue
            lowered = key_str.lower()
            if lowered in self._IDENTIFIER_FIELD_KEYS:
                continue
            if any(token in lowered for token in ("document_id", "doc_id", "metadata")):
                continue
            filtered[key_str] = value
        return filtered

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
            payload = self._filter_identifier_fields(payload)
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

    def _sentiment_descriptor(self, sentiment: Optional[Dict[str, float]]) -> str:
        if not sentiment:
            return "neutral"
        compound = sentiment.get("compound")
        if compound is None:
            return "neutral"
        if compound <= -0.2:
            return "negative"
        if compound >= 0.4:
            return "positive"
        return "neutral"

    def _compose_llm_prompt(
        self,
        query: str,
        context: str,
        draft_answer: str,
        nltk_features: Optional[Dict[str, Any]],
        ad_hoc_context: str,
    ) -> str:
        redacted_query = self._redact_identifiers(query)
        lines: List[str] = [f"User question: {redacted_query}"]

        sentiment = (nltk_features or {}).get("sentiment") if nltk_features else None
        descriptor = self._sentiment_descriptor(sentiment)

        if nltk_features:
            keywords = [
                str(item).strip()
                for item in nltk_features.get("keywords", [])
                if isinstance(item, str) and str(item).strip()
            ]
            key_phrases = [
                str(item).strip()
                for item in nltk_features.get("key_phrases", [])
                if isinstance(item, str) and str(item).strip()
            ]
            if keywords:
                lines.append(f"Key terms to emphasise: {', '.join(keywords[:8])}")
            if key_phrases:
                lines.append(f"Relevant phrases: {'; '.join(key_phrases[:5])}")

        if descriptor == "negative":
            tone_line = "Tone guidance: be empathetic and solutions-oriented."
        elif descriptor == "positive":
            tone_line = "Tone guidance: keep the response upbeat while staying factual."
        else:
            tone_line = "Tone guidance: respond in a calm, professional manner."
        lines.append(tone_line)

        if ad_hoc_context:
            lines.append(
                "User supplied context: "
                + self._redact_identifiers(ad_hoc_context)
            )

        if context:
            lines.append("Knowledge snippets:\n" + context)

        lines.append("Draft summary derived from retrieval:\n" + draft_answer)
        lines.append(
            "Transform the draft into a natural response (ideally one or two concise paragraphs). Start with a brief acknowledgement or collegial lead-in, deliver the direct answer, weave in the most relevant knowledge details, introduce short bullet or numbered lists when clarifying multiple points, point out any gaps or next steps, and keep the tone warm, conversational, and unscripted—sound like a trusted teammate rather than a script."
        )
        return "\n\n".join(lines)

    def _finalise_llm_answer(
        self,
        llm_answer: Any,
        nltk_features: Optional[Dict[str, Any]],
        fallback: str,
    ) -> str:
        candidate = llm_answer if isinstance(llm_answer, str) else ""
        candidate = candidate.strip()
        fallback_clean = self._remove_placeholders(fallback)
        if not candidate or candidate.lower().startswith("could not generate"):
            return fallback_clean

        sentiment = (nltk_features or {}).get("sentiment") if nltk_features else None
        keywords = (nltk_features or {}).get("keywords") if nltk_features else None
        key_phrases = (nltk_features or {}).get("key_phrases") if nltk_features else None
        if self._nltk_processor:
            cleaned = self._nltk_processor.postprocess(
                candidate,
                sentiment=sentiment,
                keywords=keywords,
                key_phrases=key_phrases,
            )
        else:
            cleaned = self._postprocess_answer(candidate)
        cleaned = self._remove_placeholders(cleaned)
        return cleaned or fallback_clean

    def _merge_followups(
        self, base_followups: List[str], llm_followups: Optional[List[Any]]
    ) -> List[str]:
        suggestions: List[str] = []
        if llm_followups:
            for item in llm_followups:
                if isinstance(item, str) and item.strip():
                    suggestions.append(item.strip())
        suggestions.extend(base_followups)
        deduped: List[str] = []
        for suggestion in suggestions:
            cleaned = self._remove_placeholders(suggestion)
            if not cleaned:
                continue
            if cleaned not in deduped:
                deduped.append(cleaned)
        return deduped[:3]

    def _plain_text_to_html(self, text: str) -> str:
        normalised = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not normalised:
            return '<section class="llm-answer__segment"><p>No answer available.</p></section>'

        def _insert_inline_breaks(block: str) -> str:
            pattern = re.compile(r"(?<!\n)(?:\s*)(\d+[\.)])\s+(?=[A-Za-z])")

            def repl(match: re.Match) -> str:
                marker = match.group(1)
                return "\n" + marker + " "

            return pattern.sub(repl, block)

        normalised = _insert_inline_breaks(normalised)

        header = (
            '<header class="llm-answer__heading">'
            "<h2>Here’s what I found</h2>"
            "<p class=\"llm-answer__intro\">I pulled the key details into an easy-to-scan summary for you.</p>"
            "</header>"
        )

        html_parts: List[str] = [
            header,
            '<div class="llm-answer__segment llm-answer__segment--body">',
        ]
        current_list: List[str] = []
        list_type: Optional[str] = None
        definition_items: List[Tuple[str, str]] = []
        first_paragraph_rendered = False
        sources: List[str] = []

        bullet_pattern = re.compile(r"^(?:[-*\u2022])\s+")
        ordered_pattern = re.compile(r"^\s*\d+[\.)]\s+")
        definition_pattern = re.compile(r"^\s*([^:]{1,80})\s*:\s*(.+)$")

        def format_inline(value: str) -> str:
            escaped = escape(value)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)
            escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
            return escaped

        def flush_list() -> None:
            nonlocal current_list, list_type
            if not current_list:
                return
            tag = "ol" if list_type == "ordered" else "ul"
            items = "".join(f"<li>{item}</li>" for item in current_list)
            html_parts.append(f'<{tag} class="llm-answer__list">{items}</{tag}>')
            current_list = []
            list_type = None

        def flush_definitions() -> None:
            nonlocal definition_items
            if not definition_items:
                return
            rows = "".join(
                f'<div class="llm-answer__definition"><dt>{format_inline(term)}</dt>'
                f"<dd>{format_inline(explanation)}</dd></div>"
                for term, explanation in definition_items
            )
            html_parts.append(f'<dl class="llm-answer__definitions">{rows}</dl>')
            definition_items = []

        def append_paragraph(content: str) -> None:
            nonlocal first_paragraph_rendered
            css_class = "llm-answer__lead" if not first_paragraph_rendered else "llm-answer__paragraph"
            first_paragraph_rendered = True
            html_parts.append(f'<p class="{css_class}">{format_inline(content)}</p>')

        for raw_line in normalised.split("\n"):
            line = raw_line.strip()
            if not line:
                flush_list()
                flush_definitions()
                continue

            source_idx = line.lower().find("sources:")
            if source_idx != -1:
                before = line[:source_idx].strip()
                after = line[source_idx + len("sources:") :].strip()
                if after:
                    for token in re.split(r"[;,]", after):
                        doc = token.strip()
                        if doc:
                            sources.append(doc)
                if not before:
                    continue
                line = before

            if bullet_pattern.match(line):
                flush_definitions()
                content = bullet_pattern.sub("", line).strip()
                if not content:
                    continue
                if list_type not in (None, "unordered"):
                    flush_list()
                list_type = "unordered"
                current_list.append(format_inline(content))
                continue

            if ordered_pattern.match(line):
                flush_definitions()
                content = ordered_pattern.sub("", line).strip()
                if not content:
                    continue
                if list_type not in (None, "ordered"):
                    flush_list()
                list_type = "ordered"
                current_list.append(format_inline(content))
                continue

            definition_match = definition_pattern.match(line)
            if definition_match and len(definition_match.group(1).split()) <= 12:
                flush_list()
                term = definition_match.group(1).strip()
                explanation = definition_match.group(2).strip()
                if term and explanation:
                    definition_items.append((term, explanation))
                continue

            flush_list()
            flush_definitions()
            append_paragraph(line)

        flush_list()
        flush_definitions()

        if len(html_parts) == 2:
            html_parts.append('<p class="llm-answer__paragraph">No answer available.</p>')

        html_parts.append("</div>")

        if sources:
            unique_sources: List[str] = []
            seen: Set[str] = set()
            for doc in sources:
                lowered = doc.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                unique_sources.append(doc)
            chips = "".join(
                f'<span class="llm-answer__source-chip">{escape(doc)}</span>'
                for doc in unique_sources
            )
            html_parts.append(
                '<footer class="llm-answer__sources">'
                "<h3>Sources</h3>"
                f"<div class=\"llm-answer__source-list\">{chips}</div>"
                "</footer>"
            )

        return "".join(html_parts)

    def _normalise_answer_html(self, answer: Any) -> str:
        if isinstance(answer, (list, dict)):
            try:
                serialised = json.dumps(answer, ensure_ascii=False)
            except TypeError:
                serialised = str(answer)
            return self._normalise_answer_html(serialised)

        text = "" if answer is None else str(answer)
        stripped = text.strip()
        if not stripped:
            body = "<p>No answer available.</p>"
        else:
            if stripped.startswith("<section") and stripped.endswith("</section>"):
                return stripped

            tag_pattern = re.compile(r"</?\s*([A-Za-z][A-Za-z0-9:-]*)[^>]*>")
            cleaned_parts: List[str] = []
            last_index = 0

            for match in tag_pattern.finditer(stripped):
                cleaned_parts.append(stripped[last_index : match.start()])
                tag_name = match.group(1).lower()
                if tag_name not in self._HTML_TAG_ALLOWLIST:
                    cleaned_parts.append(match.group(0))
                last_index = match.end()

            cleaned_parts.append(stripped[last_index:])
            cleaned = "".join(cleaned_parts)
            body = self._plain_text_to_html(cleaned)

        return (
            '<section class="llm-answer">'
            '<article class="llm-answer__content">'
            f"{body}"
            "</article>"
            "</section>"
        )

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
        context_line = self._conversation_context_line(query, focus_items)
        intro_sentences: List[str] = []
        if opening_line:
            intro_sentences.append(self._to_sentence(opening_line))
        if context_line:
            intro_sentences.append(self._to_sentence(context_line))
        if intro_sentences:
            paragraphs.append(" ".join(intro_sentences))

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

        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = self._strip_metadata_terms(cleaned)
        return self._apply_structured_formatting(cleaned)

    def _apply_structured_formatting(self, text: str) -> str:
        """Normalise whitespace and add paragraph/list spacing for readability."""

        if not isinstance(text, str):
            return ""

        normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalised:
            return ""

        normalised = re.sub(r"\n{3,}", "\n\n", normalised)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalised)
        paragraphs: List[str] = []

        intro_count = 3 if len(sentences) > 3 else len(sentences)
        if len(sentences) > intro_count:
            lead = " ".join(sentences[:intro_count]).strip()
            if lead:
                paragraphs.append(lead)
            remainder_sentences = sentences[intro_count:]
        else:
            remainder_sentences = []
            lead = " ".join(sentences).strip()
            if lead:
                paragraphs.append(lead)

        if remainder_sentences:
            remainder = " ".join(remainder_sentences).strip()
            if remainder:
                remainder = re.sub(r"(:)\s*([1-9]\d*[\.)])", r"\1\n\n\2", remainder)
                remainder = re.sub(
                    r"(?<!\n)([1-9]\d*[\.)])\s+",
                    lambda match: f"\n\n{match.group(1)} ",
                    remainder,
                )
                remainder = re.sub(
                    r"(?<!\n)([-*•])\s+",
                    lambda match: f"\n\n{match.group(1)} ",
                    remainder,
                )
                remainder = re.sub(r"\n{3,}", "\n\n", remainder)
                for block in remainder.split("\n\n"):
                    stripped = block.strip()
                    if stripped:
                        paragraphs.append(stripped)

        formatted = "\n\n".join(paragraphs)
        formatted = re.sub(
            r"(\d+[\.)][^\n]*\.)\s+(?=(?:If|Next|This|That|They|We|You)\b)",
            r"\1\n\n",
            formatted,
        )
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        return formatted.strip()

    def _generate_response(self, prompt: str, model: str) -> Dict:
        """Calls the LM Studio chat endpoint once to get answer and follow-ups."""
        system = (
            "System (Joshi)\n"
            "You are Joshi, the ProcWise SME. Sound like a caring, capable coworker—warm, semi-formal, and concise without seeming scripted. "
            "Open with a brief acknowledgement or collegial greeting when it feels natural (e.g., 'Thanks for the question—', 'Happy to help!'). "
            "Answer only from the provided retrieval context or known static guidance. If the context is thin, explain the gap in one sentence or ask a single clarifying question instead of guessing. "
            "Paraphrase the source material instead of copying it verbatim, and translate jargon into plain language so a busy sourcing manager can act quickly. "
            "Structure the answer as one or two short paragraphs, adding short bullet or numbered lists whenever you walk through multiple considerations, steps, or recommendations. Wrap up with a clear takeaway or next step. "
            "Do not expose internal details, identifiers, or placeholders, and avoid boilerplate openers or stock phrases. "
            "Respond in valid JSON with keys 'answer' and 'follow_ups'. Keep 'answer' friendly, collegial, and firmly grounded in the supplied knowledge while noting any limits transparently. "
            "Ensure 'follow_ups' contains three concise, context-aware questions that naturally progress the procurement discussion without repeating each other."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        client = get_lmstudio_client()
        try:
            base_options = dict(self.agent_nick.lmstudio_options() or {})
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
        base_options.setdefault("max_tokens", -1)

        try:
            response = client.chat(
                model=model,
                messages=messages,
                options=base_options,
                response_format={"type": "json_object"},
            )
            content = response.get("message", {}).get("content", "")
            return json.loads(content or "{}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            fallback = content.strip() if content else ""
            return {"answer": fallback or "Could not generate an answer.", "follow_ups": []}
        except LMStudioClientError as e:
            logger.error(f"Error generating answer from LM Studio: %s", e)
            return {"answer": "Could not generate an answer.", "follow_ups": []}

    def answer_question(
        self,
        query: str,
        user_id: str,
        *,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        files: Optional[List[tuple[bytes, str]]] = None,
        doc_type: Optional[str] = None,
        product_type: Optional[str] = None,
    ) -> Dict:
        llm_to_use = self.default_llm_model
        logger.info(
            "Answering query with model '%s' and filters: doc_type='%s', product_type='%s'",
            llm_to_use,
            doc_type,
            product_type,
        )

        history = self.history_manager.get_history(user_id)
        history_fingerprint = self._build_history_fingerprint(history)

        candidate_keys: List[str] = []
        cleaned_session = (
            session_id.strip()
            if isinstance(session_id, str)
            else str(session_id).strip()
            if session_id is not None
            else None
        )
        if cleaned_session:
            candidate_keys.append(cleaned_session)
        cleaned_user = user_id.strip() if isinstance(user_id, str) else str(user_id).strip()
        if cleaned_user:
            if cleaned_user not in candidate_keys:
                candidate_keys.append(cleaned_user)

        session_key: Optional[str] = None
        session_record: Optional[Dict[str, Any]] = None
        for candidate in candidate_keys:
            record = self._session_uploads.get(candidate)
            if record:
                session_key = candidate
                session_record = record
                break

        if session_key is None and candidate_keys:
            session_key = candidate_keys[0]

        uploaded_scope = dict(self._uploaded_context or {})
        upload_fingerprint = self._build_upload_fingerprint(
            candidate_keys=candidate_keys,
            uploaded_scope=uploaded_scope,
        )

        cache_key: Optional[str] = None
        cached_response: Optional[Dict[str, Any]] = None
        if not files:
            try:
                cache_key = self._build_cache_key(
                    query,
                    user_id,
                    session_id,
                    model_name or llm_to_use,
                    doc_type,
                    product_type,
                    context_fingerprint=upload_fingerprint,
                    history_fingerprint=history_fingerprint,
                )
            except Exception:
                cache_key = None
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response

        raw_nltk_features: Optional[Dict[str, Any]] = None
        if self._nltk_processor:
            nltk_feature_record = self._nltk_processor.preprocess(query)
            raw_nltk_features = {
                "keywords": nltk_feature_record.keywords,
                "key_phrases": nltk_feature_record.key_phrases,
                "sentiment": nltk_feature_record.sentiment,
            }
            logger.debug(
                "NLTK query profile: keywords=%s | phrases=%s | sentiment=%s",
                [
                    self._redact_identifiers(item)
                    for item in nltk_feature_record.keywords[:6]
                ],
                [
                    self._redact_identifiers(item)
                    for item in nltk_feature_record.key_phrases[:4]
                ],
                nltk_feature_record.sentiment,
            )

        static_session_token = (
            session_key
            or cleaned_session
            or cleaned_user
            or (user_id.strip() if isinstance(user_id, str) else str(user_id))
        )
        static_response = self._try_static_answer(
            query,
            user_id,
            session_id=static_session_token,
        )
        if static_response:
            self._store_cached_response(cache_key, static_response)
            return static_response

        uploaded_documents: List[str] = []
        try:
            uploaded_documents = [
                str(doc).strip()
                for doc in uploaded_scope.get("document_ids", [])
                if str(doc).strip()
            ]
        except Exception:
            uploaded_documents = []

        matched_owner: Optional[str] = None
        scope_session_token: Optional[str] = None
        scope_metadata: Dict[str, Any] = {}
        allowed_identifiers: List[str] = []
        if uploaded_scope:
            try:
                scope_metadata = dict(uploaded_scope.get("metadata") or {})
            except Exception:
                scope_metadata = {}

            raw_scope_session = uploaded_scope.get("session_id") if isinstance(uploaded_scope, dict) else None
            if raw_scope_session is not None:
                try:
                    cleaned_scope_session = str(raw_scope_session).strip()
                except Exception:
                    cleaned_scope_session = ""
                if cleaned_scope_session:
                    scope_session_token = cleaned_scope_session
                    allowed_identifiers.append(cleaned_scope_session)

            uploaded_by = scope_metadata.get("uploaded_by")
            if isinstance(uploaded_by, str):
                cleaned_uploaded_by = uploaded_by.strip()
                if cleaned_uploaded_by:
                    allowed_identifiers.append(cleaned_uploaded_by)

        restrict_to_uploaded = False
        if uploaded_documents and allowed_identifiers:
            for candidate in candidate_keys:
                if candidate and candidate in allowed_identifiers:
                    matched_owner = candidate
                    restrict_to_uploaded = True
                    break

        if restrict_to_uploaded:
            combined_record = dict(session_record or {})
            combined_metadata = dict(combined_record.get("metadata") or {})
            combined_metadata.update(scope_metadata)
            session_record = {
                "document_ids": list(uploaded_documents),
                "metadata": combined_metadata,
            }
            if scope_session_token and not session_key:
                session_key = scope_session_token
            elif matched_owner and not session_key:
                session_key = matched_owner

        session_hint, memory_fragments, history_signals = self._analyse_session_history(
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
        if restrict_to_uploaded and uploaded_documents:
            if len(uploaded_documents) == 1:
                doc_match = models.MatchValue(value=uploaded_documents[0])
            else:
                doc_match = models.MatchAny(any=uploaded_documents)
            must_conditions.append(
                models.FieldCondition(key="document_id", match=doc_match)
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

        policy_mode = False
        if not restrict_to_uploaded:
            policy_mode = self._is_policy_question(
                query, doc_type, session_hint, history_signals
            )
        search_kwargs = {
            "top_k": 6,
            "filters": qdrant_filter,
        }
        if session_key:
            search_kwargs["session_id"] = session_key
        hint_kwargs = {
            "session_hint": session_hint,
            "memory_fragments": memory_fragments,
            "policy_mode": policy_mode,
        }
        if raw_nltk_features:
            hint_kwargs["nltk_features"] = raw_nltk_features
        if session_key:
            hint_kwargs["session_id"] = session_key
        if session_record:
            hint_kwargs["session_documents"] = session_record.get("document_ids")
        search_kwargs.update(self._supported_search_kwargs(hint_kwargs))
        if restrict_to_uploaded:
            search_kwargs["collections"] = (self.rag.uploaded_collection,)
        reranked = self.rag.search(query, **search_kwargs)
        knowledge_items = self._prepare_knowledge_items(reranked)

        if knowledge_items and not restrict_to_uploaded:
            try:
                static_output = self._static_agent.run(
                    query=query,
                    user_id=user_id,
                    session_id=session_key or cleaned_user,
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
                        static_context_payload = self._filter_identifier_fields(
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
            fallback = self._remove_placeholders(fallback)
            html_fallback = self._normalise_answer_html(fallback)
            history.append({"query": self._redact_identifiers(query), "answer": fallback})
            self.history_manager.save_history(user_id, history)
            history_fingerprint = self._build_history_fingerprint(history)
            if cache_key:
                try:
                    cache_key = self._build_cache_key(
                        query,
                        user_id,
                        session_id,
                        model_name or llm_to_use,
                        doc_type,
                        product_type,
                        context_fingerprint=upload_fingerprint,
                        history_fingerprint=history_fingerprint,
                    )
                except Exception:
                    pass
            follow_ups = self._build_followups(query, [])
            result_payload = {
                "answer": html_fallback,
                "follow_ups": follow_ups,
                "retrieved_documents": [],
            }
            self._store_cached_response(cache_key, result_payload)
            return result_payload

        focus_items = self._select_focus_items(knowledge_items)
        enumerated_items: List[Dict[str, Any]] = [dict(item) for item in focus_items]
        context_body = self._synthesise_context(knowledge_items)

        retrieved_documents_payloads: List[Dict[str, Any]] = []
        for item in enumerated_items:
            payload = dict(item.get("payload", {}))
            payload = self._filter_identifier_fields(payload)
            payload.setdefault("collection_name", item.get("collection"))
            payload.setdefault("source_label", item.get("source_label"))
            retrieved_documents_payloads.append(payload)

        draft_answer = self._build_structured_answer(query, enumerated_items, ad_hoc_context)
        base_followups = self._build_followups(query, enumerated_items)

        prompt = self._compose_llm_prompt(
            query,
            context_body,
            draft_answer,
            raw_nltk_features,
            ad_hoc_context,
        )
        llm_payload = self._generate_response(prompt, llm_to_use)
        answer = self._finalise_llm_answer(
            llm_payload.get("answer"), raw_nltk_features, draft_answer
        )
        follow_ups = self._merge_followups(
            base_followups, llm_payload.get("follow_ups")
        )

        html_answer = self._normalise_answer_html(answer)
        history.append({"query": self._redact_identifiers(query), "answer": html_answer})
        self.history_manager.save_history(user_id, history)
        history_fingerprint = self._build_history_fingerprint(history)
        if cache_key:
            try:
                cache_key = self._build_cache_key(
                    query,
                    user_id,
                    session_id,
                    model_name or llm_to_use,
                    doc_type,
                    product_type,
                    context_fingerprint=upload_fingerprint,
                    history_fingerprint=history_fingerprint,
                )
            except Exception:
                pass

        result_payload = {
            "answer": html_answer,
            "follow_ups": follow_ups,
            "retrieved_documents": retrieved_documents_payloads,
        }
        self._store_cached_response(cache_key, result_payload)
        return result_payload
