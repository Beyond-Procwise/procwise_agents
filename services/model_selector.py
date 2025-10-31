# ProcWise/services/model_selector.py

import json
import logging
import re
import ollama
import pdfplumber
from io import BytesIO
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Type
from sentence_transformers import CrossEncoder
from config.settings import settings
from qdrant_client import models
from agents.base_agent import AgentStatus
from agents.rag_agent import RAGAgent
from .rag_service import RAGService
from utils.gpu import configure_gpu, load_cross_encoder

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
        return summary

    def _label_for_collection(self, collection: Optional[str]) -> str:
        if not collection:
            return "Knowledge base insight"
        if collection == self.rag.primary_collection:
            return "ProcWise knowledge base"
        if collection == self.rag.uploaded_collection:
            return "Uploaded reference"
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
                or payload.get("record_id")
                or str(getattr(hit, "id", "Document"))
            )
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

    def answer_question(self, query: str, user_id: str, model_name: Optional[str] = None,
                        files: Optional[List[tuple[bytes, str]]] = None, doc_type: Optional[str] = None,
                        product_type: Optional[str] = None) -> Dict:
        static_candidate = self._try_static_answer(query, user_id)
        if static_candidate:
            return static_candidate

        llm_to_use = model_name or self.default_llm_model
        logger.info(
            f"Answering query with model '{llm_to_use}' and filters: doc_type='{doc_type}', product_type='{product_type}'")

        # The vector collection is initialized once during application startup
        # via ``AgentNick``. Re-initializing here would trigger unnecessary
        # Qdrant calls and slow down the ``/ask`` endpoint.

        # --- Normalise filters and build Vector DB filter conditions ---
        must_conditions: List[models.FieldCondition] = []
        if doc_type:
            doc_type = doc_type.lower()
            must_conditions.append(
                models.FieldCondition(key="document_type", match=models.MatchValue(value=doc_type))
            )
        if product_type:
            product_type = product_type.lower()
            must_conditions.append(
                models.FieldCondition(key="product_type", match=models.MatchValue(value=product_type))
            )
        qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

        # --- Process Uploaded Files ---
        uploaded = self._extract_text_from_uploads(files) if files else []
        ad_hoc_notes: List[str] = []
        for idx, file_info in enumerate(uploaded):
            filename = file_info.get("name") or f"uploaded-reference-{idx + 1}"
            text = file_info.get("text", "")
            summary = file_info.get("summary") or self._condense_snippet(text)
            if summary:
                ad_hoc_notes.append(f"{filename}: {summary}")
            metadata = {"record_id": filename, "document_type": doc_type or "uploaded"}
            if product_type:
                metadata["product_type"] = product_type.lower()
            if text:
                self.rag.upsert_texts([text], metadata)
        ad_hoc_context = "\n".join(ad_hoc_notes)

        # --- Retrieve from Vector DB ---
        top_k = 6
        reranked = self.rag.search(query, top_k=top_k, filters=qdrant_filter)
        knowledge_items = self._prepare_knowledge_items(reranked)
        retrieved_documents_payloads: List[Dict[str, Any]] = [
            item["payload"] for item in knowledge_items
        ]

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
                    static_context_payload = {
                        "source": "static_procurement_qa",
                        "topic": payload.get("topic"),
                        "question": payload.get("question"),
                        "answer": answer_text,
                        "confidence": float(static_output.confidence or 0.0),
                    }
                    retrieved_documents_payloads.append(static_context_payload)
                    knowledge_items.append(
                        {
                            "payload": static_context_payload,
                            "collection": "static_procurement_qa",
                            "source_label": self._label_for_collection("static_procurement_qa"),
                            "document": payload.get("topic")
                            or payload.get("question")
                            or "Procurement guidance",
                            "summary": self._condense_snippet(
                                answer_text, max_sentences=2, max_chars=320
                            ),
                        }
                    )

        retrieved_context = self._synthesise_context(knowledge_items)

        if not knowledge_items and not ad_hoc_context:
            history = self.history_manager.get_history(user_id)
            history_context = self._format_history_context(history)
            if not history_context:
                return {
                    "answer": "I could not find relevant knowledge or prior conversation to answer that just yet.",
                    "follow_ups": [],
                    "retrieved_documents": [],
                }
            prompt = f"""No knowledge base excerpts were retrieved for the latest query. Use the recent conversation to craft a helpful response that stays within procurement context and acknowledges any gaps.

### Recent Conversation:
{history_context}

### User's Question:
{query}

Guidelines:
1. Continue the discussion in a warm, professional tone.
2. If information is missing, suggest the most practical next step.
3. Provide three relevant follow-up questions that build on the dialogue.
"""
            model_output = self._generate_response(prompt, llm_to_use)
            answer_raw = model_output.get("answer", "Could not generate an answer.")
            follow_ups_raw = model_output.get("follow_ups", [])
            answer = self._postprocess_answer(answer_raw)
            if not isinstance(follow_ups_raw, list):
                follow_ups = []
            else:
                follow_ups = [
                    str(item).strip() for item in follow_ups_raw if str(item).strip()
                ][:5]
            history.append({"query": query, "answer": answer})
            self.history_manager.save_history(user_id, history)
            return {
                "answer": answer,
                "follow_ups": follow_ups,
                "retrieved_documents": [],
            }

        prompt = f"""Use the context below to answer the procurement question with a natural, consultant-style summary.

### Uploaded References:
{ad_hoc_context if ad_hoc_context else "No user uploads were provided for this query."}

### Knowledge Base Highlights:
{retrieved_context if retrieved_context else "No direct knowledge snippets matched; rely on procurement best practice and explain any gaps."}

### User's Question:
{query}

Guidelines:
1. Weave the information into two or three short paragraphs that read like a human analyst.
2. Paraphrase instead of quoting directly, and clarify whether items are complete, pending, or uncertain.
3. Flag any missing data and suggest the most useful next action.
4. Provide three forward-looking follow-up questions tailored to the conversation.
"""

        model_output = self._generate_response(prompt, llm_to_use)
        answer_raw = model_output.get("answer", "Could not generate an answer.")
        follow_ups_raw = model_output.get("follow_ups", [])
        answer = self._postprocess_answer(answer_raw)
        if not isinstance(follow_ups_raw, list):
            follow_ups = []
        else:
            follow_ups = [
                str(item).strip() for item in follow_ups_raw if str(item).strip()
            ][:5]

        history = self.history_manager.get_history(user_id)
        history.append({"query": query, "answer": answer})
        self.history_manager.save_history(user_id, history)

        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "retrieved_documents": retrieved_documents_payloads,
        }
