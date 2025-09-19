import json
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from qdrant_client import models
from sentence_transformers import CrossEncoder

from services.rag_service import RAGService
from services.intelligent_retrieval import IntelligentRetrievalService
from .base_agent import BaseAgent
from utils.gpu import configure_gpu

configure_gpu()


class RAGAgent(BaseAgent):
    """Retrieval augmented generation agent with GPU optimisations."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        model_name = getattr(self.settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Cross encoder ensures accurate ranking while utilising the GPU
        self._reranker = CrossEncoder(model_name, device=self.agent_nick.device)
        # Thread pool enables parallel answer and follow-up generation
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Service providing FAISS and BM25 retrieval
        self.rag_service = RAGService(agent_nick)
        # Intelligent retrieval service with enhanced capabilities
        self.intelligent_retrieval = IntelligentRetrievalService(self.rag_service, agent_nick)

    def run(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        doc_type: str | None = None,
        product_type: str | None = None,
    ):
        """Answer questions while maintaining (but not utilising) chat history."""

        print(
            f"RAGAgent received query: '{query}' for user: '{user_id}'"
            + (f" in session '{session_id}'" if session_id else "")
        )

        must: list[models.FieldCondition] = []
        if doc_type:
            must.append(
                models.FieldCondition(
                    key="document_type", match=models.MatchValue(value=doc_type.lower())
                )
            )
        if product_type:
            must.append(
                models.FieldCondition(
                    key="product_type", match=models.MatchValue(value=product_type.lower())
                )
            )
        qdrant_filter = models.Filter(must=must) if must else None

        search_hits = []
        seen_ids = set()
        
        # Use intelligent retrieval for better results
        try:
            search_hits = self.intelligent_retrieval.adaptive_search(
                query, 
                top_k=top_k * 2,  # Get more candidates for better filtering
                filters=qdrant_filter
            )
            # Remove duplicates while preserving order
            unique_hits = []
            for hit in search_hits:
                if hit.id not in seen_ids:
                    unique_hits.append(hit)
                    seen_ids.add(hit.id)
            search_hits = unique_hits
        except Exception as e:
            # Fallback to original method if intelligent retrieval fails
            print(f"Intelligent retrieval failed, falling back to original method: {e}")
            query_variants = [query] + self._expand_query(query)
            for q in query_variants:
                hits = self.rag_service.search(q, top_k=top_k, filters=qdrant_filter)
                for h in hits:
                    if h.id not in seen_ids:
                        search_hits.append(h)
                        seen_ids.add(h.id)

        if not search_hits:
            answer = "I could not find any relevant documents to answer your question."
            history = self._load_chat_history(user_id, session_id) if session_id else self._load_chat_history(user_id)
            history.append({"query": query, "answer": answer})
            if session_id:
                self._save_chat_history(user_id, history, session_id)
            else:
                self._save_chat_history(user_id, history)
            return {"answer": answer, "follow_up_questions": [], "retrieved_documents": []}

        reranked = self._rerank(query, search_hits, top_k)
        if not reranked:
            answer = "I could not find any relevant documents to answer your question."
            history = self._load_chat_history(user_id, session_id) if session_id else self._load_chat_history(user_id)
            history.append({"query": query, "answer": answer})
            if session_id:
                self._save_chat_history(user_id, history, session_id)
            else:
                self._save_chat_history(user_id, history)
            return {"answer": answer, "follow_up_questions": [], "retrieved_documents": []}

        context = "".join(
            f"Document ID: {hit.payload.get('record_id', hit.id)}, Score: {hit.score:.2f}\n"
            f"Content: {hit.payload.get('content', hit.payload.get('summary', ''))}\n---\n"
            for hit in reranked
        )
        rag_prompt = (
            "You are a helpful procurement assistant. Use ONLY the provided RETRIEVED CONTENT to answer "
            "the USER QUESTION. if required use external knowledge beyond the context only related to procurement process.\n\n"
            "Instructions:\n"
            "1) Return a single plain string as the final answer. Do NOT return JSON, YAML, Markdown, lists, or any extra metadata — only the answer text.\n"
            "2) If the context contains relevant information, provide a concise, human-readable answer that integrates any structured content into natural language.\n"
            "3) If the context does not contain the answer, return exactly the following string (no extra text):\n"
            "   I could not find any relevant information in the provided documents.\n"
            "4) For any factual claims or new information derived from the retrieved content, append an inline citation immediately after the claim in this format: [Document ID: <id>] or for multiple documents [Document IDs: id1,id2].\n"
            "5) If quoting text verbatim from the context, enclose the quote in double quotes and include the document ID citation after the quote.\n"
            "6) Do NOT hallucinate, invent facts, or cite documents that were not present in the RETRIEVED CONTENT. If uncertain, state that the information is unclear and cite the relevant document(s).\n"
            "7) Preserve numeric values, dates, currencies and units exactly as presented in the context.\n"
            "8) Keep the answer concise (aim for one short paragraph in simple terms).\n\n"
            f"RETRIEVED CONTENT:\n{context}\n\nUSER QUESTION: {query}\n\nReturn only the final answer string below:\n"
        )

        # Generate answer and follow-up suggestions in parallel
        answer_future = self._executor.submit(
            self.call_ollama, rag_prompt, model=self.settings.extraction_model
        )
        follow_future = self._executor.submit(self._generate_followups, query, context)

        answer_resp = answer_future.result()
        followups = follow_future.result()
        answer = answer_resp.get("response", "I am sorry, I could not generate an answer.")

        history = self._load_chat_history(user_id, session_id) if session_id else self._load_chat_history(user_id)
        history.append({"query": query, "answer": answer})
        if session_id:
            self._save_chat_history(user_id, history, session_id)
        else:
            self._save_chat_history(user_id, history)

        return {
            "answer": answer,
            "follow_up_questions": followups,
            "retrieved_documents": [hit.payload for hit in search_hits] if search_hits else [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rerank(self, query: str, hits, top_k: int):
        """Re-rank search hits using a cross-encoder for improved accuracy."""
        pairs = [
            (query, hit.payload.get("content", hit.payload.get("summary", "")))
            for hit in hits
        ]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        # Filter out obviously irrelevant documents using a neutral score
        filtered = [hit for hit, score in ranked if score > 0][:top_k]
        return filtered

    def _expand_query(self, query: str) -> list:
        """Generate alternative phrasings to boost retrieval recall."""
        prompt = (
            "Provide up to three alternate phrasings or related search terms for the "
            "following procurement question. Return JSON {\"expansions\": [\"...\"]}.\n"
            f"QUESTION: {query}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(prompt, model=self.settings.extraction_model, format="json")
            data = json.loads(resp.get("response", "{}"))
            expansions = data.get("expansions", [])
            if isinstance(expansions, list):
                return [e for e in expansions if isinstance(e, str)]
        except Exception:
            pass
        return []

    def _generate_followups(self, query: str, context: str):
        """Generate follow-up questions based on current context and query intent."""
        # Detect query intent for more targeted follow-ups
        try:
            query_intent = self.intelligent_retrieval.detect_query_intent(query)
            intent_context = f" The user asked a {query_intent.value} question."
        except Exception:
            intent_context = ""
        
        prompt = (
            "You are a helpful procurement assistant. Based on the user's question and the "
            "retrieved context, suggest three concise follow-up questions that would clarify "
            f"the request or gather more details.{intent_context} Return each question on a new line.\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nFOLLOW_UP_QUESTIONS:"
        )
        resp = self.call_ollama(prompt, model=self.settings.extraction_model)
        questions = resp.get("response", "").strip().splitlines()
        return [q for q in questions if q]

    def _load_chat_history(self, user_id: str, session_id: str | None = None) -> list:
        history_key = (
            f"chat_history/{user_id}/{session_id}.json" if session_id else f"chat_history/{user_id}.json"
        )
        try:
            s3_object = self.agent_nick.s3_client.get_object(
                Bucket=self.settings.s3_bucket_name, Key=history_key
            )
            history = json.loads(s3_object["Body"].read().decode("utf-8"))
            print(
                f"Loaded {len(history)} items from chat history for user '{user_id}'"
                + (f" session '{session_id}'" if session_id else "")
                + "."
            )
            return history
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(
                    f"No chat history found for user '{user_id}'"
                    + (f" session '{session_id}'" if session_id else "")
                    + ". Starting new session."
                )
                return []
            raise

    def _save_chat_history(
        self, user_id: str, history: list, session_id: str | None = None
    ) -> None:
        history_key = (
            f"chat_history/{user_id}/{session_id}.json" if session_id else f"chat_history/{user_id}.json"
        )
        bucket_name = getattr(self.settings, "s3_bucket_name", None)
        if not bucket_name:
            print(
                "ERROR: S3 bucket name is not configured. Please set 's3_bucket_name' in settings."
            )
            return
        try:
            self.agent_nick.s3_client.put_object(
                Bucket=bucket_name,
                Key=history_key,
                Body=json.dumps(history, indent=2),
            )
            print(
                f"Saved chat history for user '{user_id}'"
                + (f" session '{session_id}'" if session_id else "")
                + f" in bucket '{bucket_name}'."
            )
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                print(
                    f"ERROR: S3 bucket '{bucket_name}' does not exist. Please check configuration."
                )
            else:
                print(
                    f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}"
                )
        except Exception as e:
            print(
                f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}"
            )

