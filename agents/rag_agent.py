
import json
import os
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from qdrant_client import models
from sentence_transformers import CrossEncoder
import torch

from .base_agent import BaseAgent

# Ensure GPU variables are set for execution environments that provide CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if torch.cuda.is_available():  # pragma: no cover - hardware dependent
    torch.set_default_device("cuda")


class RAGAgent(BaseAgent):
    """Retrieval augmented generation agent with GPU optimisations."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        model_name = getattr(self.settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Cross encoder ensures accurate ranking while utilising the GPU
        self._reranker = CrossEncoder(model_name, device=self.agent_nick.device)
        # Thread pool enables parallel answer and follow-up generation
        self._executor = ThreadPoolExecutor(max_workers=2)

    def run(self, query: str, user_id: str, top_k: int = 5):
        """Answer questions while maintaining chat history."""
        print(f"RAGAgent received query: '{query}' for user: '{user_id}'")

        history = self._load_chat_history(user_id)

        search_hits = []
        seen_ids = set()

        # Expand the user's query to improve recall in the vector search stage
        query_variants = [query] + self._expand_query(query)
        for q in query_variants:
            q_vec = self.agent_nick.embedding_model.encode(q).tolist()
            hits = self.agent_nick.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=q_vec,
                limit=top_k * 3,
                with_payload=True,
                search_params=models.SearchParams(hnsw_ef=256, exact=False),
            )
            for h in hits:
                if h.id not in seen_ids:
                    search_hits.append(h)
                    seen_ids.add(h.id)

        if not search_hits:
            return {"answer": "I could not find any relevant documents to answer your question."}

        reranked = self._rerank(query, search_hits, top_k)

        if not reranked:
            return {"answer": "I could not find any relevant documents to answer your question."}

        context = "".join(
            f"Document ID: {hit.payload.get('record_id', hit.id)}, Score: {hit.score:.2f}\n"
            f"Content: {hit.payload.get('content', hit.payload.get('summary', ''))}\n---\n"
            for hit in reranked
        )

        history_context = "\n".join(
            [f"Previous Q: {h['query']}\nPrevious A: {h['answer']}" for h in history]
        )

        rag_prompt = (
            "You are a helpful procurement assistant.\n"
            "You have access to a large set of procurement documents.\n"
            "Answer the user's question based on the provided context.\n"
            "If the context contains relevant information, provide a concise answer.\n"
            "If the context does not contain the answer, reply with \"I could not find any relevant information in the provided documents.\"\n"
            "Cite the Document ID for new information you use.\n\n"
            f"CHAT HISTORY:\n{history_context}\n\nRETRIEVED CONTENT:\n{context}\n"
            f"USER QUESTION: {query}\nANSWER:"
        )

        # Generate answer and follow-up suggestions in parallel
        answer_future = self._executor.submit(
            self.call_ollama, rag_prompt, model=self.settings.extraction_model
        )
        follow_future = self._executor.submit(self._generate_followups, query, context)

        answer_resp = answer_future.result()
        followups = follow_future.result()
        answer = answer_resp.get("response", "I am sorry, I could not generate an answer.")

        history.append({"query": query, "answer": answer})
        self._save_chat_history(user_id, history)

        return {
            "answer": answer,
            "follow_up_questions": followups,
            "retrieved_documents": [hit.payload for hit in reranked],
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
        """Generate follow-up questions based on current context."""
        prompt = (
            "You are a helpful procurement assistant. Based on the user's question and the "
            "retrieved context, suggest three concise follow-up questions that would clarify "
            "the request or gather more details. Return each question on a new line.\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nFOLLOW_UP_QUESTIONS:"
        )
        resp = self.call_ollama(prompt, model=self.settings.extraction_model)
        questions = resp.get("response", "").strip().splitlines()
        return [q for q in questions if q]

    def _load_chat_history(self, user_id: str) -> list:
        history_key = f"chat_history/{user_id}.json"
        try:
            s3_object = self.agent_nick.s3_client.get_object(
                Bucket=self.settings.s3_bucket_name, Key=history_key
            )
            history = json.loads(s3_object["Body"].read().decode("utf-8"))
            print(f"Loaded {len(history)} items from chat history for user '{user_id}'.")
            return history
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(
                    f"No chat history found for user '{user_id}'. Starting new session."
                )
                return []
            else:
                raise

    def _save_chat_history(self, user_id: str, history: list):
        history_key = f"chat_history/{user_id}.json"
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
                f"Saved chat history for user '{user_id}' in bucket '{bucket_name}'."
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

