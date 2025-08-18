# ProcWise/services/model_selector.py

import json
import logging
import os
import ollama
import pdfplumber
from io import BytesIO
from botocore.exceptions import ClientError
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from typing import Type
from config.settings import settings
from qdrant_client import models
from .rag_service import RAGService

logger = logging.getLogger(__name__)

# Ensure GPU is utilised when available
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")


class ChatHistoryManager:
    """Manages chat history using an AWS S3 bucket."""

    def __init__(self, s3_client, bucket_name):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = 'chat_history/'

    def get_history(self, user_id: str) -> List:
        key = f"{self.prefix}{user_id}.json"
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(obj['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey': return []
            logger.error(f"S3 get_object error for key {key}: {e}");
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
        self.default_llm_model = settings.extraction_model
        self.rag = RAGService(agent_nick)
        model_name = getattr(self.settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
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
            "You are a helpful assistant. Respond in valid JSON with keys 'answer' and 'follow_ups' "
            "where 'follow_ups' is a list of 3 to 5 short questions."
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
        top_k = 5
        reranked = self.rag.search(query, top_k=top_k, filters=qdrant_filter)
        # --- Retrieve from Vector DB using the unified collection name ---
        top_k = 5
        query_vector = self.agent_nick.embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()
        search_results = self.agent_nick.qdrant_client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(hnsw_ef=256, exact=False),
            score_threshold=0.6,
        )
        reranked = self._rerank_search(query, search_results, top_k)
        retrieved_context = "\n---\n".join(
            [
                (
                    f"{hit.payload.get('document_type', 'document').title()} "
                    f"{hit.payload.get('record_id', hit.id)}\n"
                    f"Summary: {hit.payload.get('summary', hit.payload.get('content', ''))}"
                )
                for hit in reranked
            ]
        ) if reranked else "No relevant documents found."

        # --- Chat History ---
        history = self.history_manager.get_history(user_id)
        history_context = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history])

        # --- Build Final Prompt for single chat call ---
        prompt = f"""Use the following information in priority order to answer the user's question and suggest follow-ups.

### Ad-hoc Context from Uploaded Files:
{ad_hoc_context if ad_hoc_context else "No files were uploaded for this query."}

### Retrieved Documents from Knowledge Base:
{retrieved_context}

### Chat History:
{history_context if history_context else "No previous conversation history."}

### User's Question:
{query}
"""

        model_output = self._generate_response(prompt, llm_to_use)
        answer = model_output.get("answer", "Could not generate an answer.")
        follow_ups = model_output.get("follow_ups", [])

        # --- Save History to S3 ---
        history.append({"query": query, "answer": answer})
        self.history_manager.save_history(user_id, history)
        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "retrieved_documents": [hit.payload for hit in reranked],
        }
