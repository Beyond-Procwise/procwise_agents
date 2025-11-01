# ProcWise/services/model_selector.py

import json
import logging
import re
import ollama
import pdfplumber
from io import BytesIO
from pathlib import Path
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Type
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
                "Thanks for flagging this — here's what I can confirm.",
                "Appreciate the context. Here's the current view.",
            ],
            "summary_intro": "Current highlights from the knowledge base:",
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
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "[redacted]", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _condense_snippet(
        self,
        text: str,
        *,
        max_sentences: int = 3,
        max_chars: int = 360,
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
        if collection == self.rag.learning_collection:
            return "Procurement playbook"
        if collection == "static_procurement_qa":
            return "Static procurement guidance"
        return collection.replace("_", " ").title()

    def _prepare_knowledge_items(self, hits: List) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            payload = dict(payload)
            collection = payload.get("collection_name", self.rag.primary_collection)
            source_label = self._label_for_collection(collection)
            payload.setdefault("source_label", source_label)
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
            self.rag.learning_collection,
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
        acknowledgements = guidelines.get("acknowledgements", [])
        ack = (
            acknowledgements[0]
            if acknowledgements
            else "Thanks for flagging this — here's what I can confirm."
        )
        summary_intro = guidelines.get(
            "summary_intro", "Here's what I was able to confirm from the knowledge base:"
        )

        lines: List[str] = [ack, summary_intro]
        for item in enumerated_items:
            doc_label = item.get("document") or "Procurement reference"
            snippet = item.get("summary") or self._condense_snippet(
                (item.get("payload") or {}).get("content", "")
            )
            snippet = self._redact_identifiers(snippet)
            citation = item.get("citation")
            line = f"- {doc_label}: {snippet}".strip()
            if citation:
                line = f"{line} {citation}".strip()
            lines.append(line)

        if ad_hoc_context:
            lines.append(
                f"- Uploaded notes: {self._redact_identifiers(ad_hoc_context)} [uploads]"
            )

        actions_lead = guidelines.get("actions_lead")
        if actions_lead:
            lines.append(actions_lead)

        return "\n".join(line for line in lines if line)

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

        reranked = self.rag.search(query, top_k=6, filters=qdrant_filter)
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
                "fallback", "I do not have that information as per my knowledge."
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

        enumerated_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(knowledge_items, start=1):
            enriched = dict(item)
            enriched["citation"] = f"[doc {idx}]"
            enumerated_items.append(enriched)

        retrieved_documents_payloads: List[Dict[str, Any]] = []
        for item in enumerated_items:
            payload = dict(item.get("payload", {}))
            payload.setdefault("collection_name", item.get("collection"))
            payload.setdefault("source_label", item.get("source_label"))
            payload["citation"] = item.get("citation")
            retrieved_documents_payloads.append(payload)

        answer = self._build_structured_answer(query, enumerated_items, ad_hoc_context)
        follow_ups = self._build_followups(query, enumerated_items)

        history = self.history_manager.get_history(user_id)
        history.append({"query": self._redact_identifiers(query), "answer": answer})
        self.history_manager.save_history(user_id, history)

        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "retrieved_documents": retrieved_documents_payloads,
        }
