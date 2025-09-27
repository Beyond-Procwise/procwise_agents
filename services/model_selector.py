# ProcWise/services/model_selector.py

import json
import logging
import ollama
import pdfplumber
from io import BytesIO
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Type
from sentence_transformers import CrossEncoder
from config.settings import settings
from qdrant_client import models
from .rag_service import RAGService
from utils.gpu import configure_gpu

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
        self.rag = RAGService(agent_nick)
        model_name = getattr(
            self.settings,
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        )
        self._reranker = cross_encoder_cls(model_name, device=self.agent_nick.device)

    def _extract_text_from_uploads(self, files: List[tuple[bytes, str]]) -> List[tuple[str, str]]:
        """Return extracted text for each uploaded PDF."""
        results: List[tuple[str, str]] = []
        for content_bytes, filename in files:
            try:
                if filename.lower().endswith('.pdf'):
                    with pdfplumber.open(BytesIO(content_bytes)) as pdf:
                        text = "\n".join(
                            page.extract_text() for page in pdf.pages if page.extract_text()
                        )
                        results.append((filename, text))
            except Exception as e:
                logger.error(f"Failed to process uploaded file {filename}: {e}")
        return results

    def _rerank_search(self, query: str, hits: List, top_k: int = 5):
        """Re-rank search hits using a cross-encoder for improved accuracy."""
        if not hits:
            return []
        pairs = [
            (query, h.payload.get("summary", h.payload.get("content", "")))
            for h in hits
        ]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]

    def _generate_response(self, prompt: str, model: str) -> Dict:
        """Calls :func:`ollama.chat` once to get answer and follow-ups."""
        system = (
            "You are a procurement-focused assistant. Respond in valid JSON with keys 'answer' and 'follow_ups' "
            "where 'follow_ups' is a list of 3 to 5 short questions. Use only the supplied context, "
            "external procurement knowledge, and never manipulate or infer customer data beyond the prompt." 
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options=self.agent_nick.ollama_options(),
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
        ad_hoc_context = []
        for fname, text in uploaded:
            ad_hoc_context.append(f"--- Content from uploaded file: {fname} ---\n{text}")
            meta = {"record_id": fname, "document_type": doc_type or "uploaded"}
            if product_type:
                meta["product_type"] = product_type.lower()
            self.rag.upsert_texts([text], meta)
        ad_hoc_context = "\n\n".join(ad_hoc_context)

        # --- Retrieve from Vector DB ---
        top_k = 6
        reranked = self.rag.search(query, top_k=top_k, filters=qdrant_filter)
        retrieved_context = "\n---\n".join(
            [
                (
                    f"{hit.payload.get('document_type', 'document').title()} "
                    f"{hit.payload.get('record_id', hit.id)}\n"
                    f"Summary: {hit.payload.get('summary', hit.payload.get('content', ''))}"
                )
                for hit in reranked
            ]
        ) if reranked else ""

        history = []
        if not reranked:
            history = self.history_manager.get_history(user_id)
            history_context = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history])
            if not history_context:
                return {
                    "answer": "Could not find relevant documents or chat history to answer the question.",
                    "follow_ups": [],
                    "retrieved_documents": [],
                }
            prompt = f"""Use the following information to answer the user's question and suggest follow-ups.
Only leverage procurement-focused external knowledge when the retrieved context is insufficient, and never infer or manipulate customer-specific data beyond what is provided.

### Ad-hoc Context from Uploaded Files:
{ad_hoc_context if ad_hoc_context else "No files were uploaded for this query."}

### Chat History:
{history_context}

### User's Question:
{query}
"""
            model_output = self._generate_response(prompt, llm_to_use)
            answer = model_output.get("answer", "Could not generate an answer.")
            follow_ups = model_output.get("follow_ups", [])
            history.append({"query": query, "answer": answer})
            self.history_manager.save_history(user_id, history)
            return {
                "answer": answer,
                "follow_ups": follow_ups,
                "retrieved_documents": [],
            }

        # When documents are found, prioritise them over chat history
        prompt = f"""Use the following information to answer the user's question and suggest follow-ups.
Only leverage procurement-focused external knowledge when the retrieved context is insufficient, and never infer or manipulate customer-specific data beyond what is provided.

### Ad-hoc Context from Uploaded Files:
{ad_hoc_context if ad_hoc_context else "No files were uploaded for this query."}

### Retrieved Documents from Knowledge Base:
{retrieved_context}

### User's Question:
{query}
"""

        model_output = self._generate_response(prompt, llm_to_use)
        answer = model_output.get("answer", "Could not generate an answer.")
        follow_ups = model_output.get("follow_ups", [])

        # Save history after answering
        history = self.history_manager.get_history(user_id)
        history.append({"query": query, "answer": answer})
        self.history_manager.save_history(user_id, history)
        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "retrieved_documents": [hit.payload for hit in reranked],
        }
