# ProcWise/agents/rag_agent.py

import json
from botocore.exceptions import ClientError
from .base_agent import BaseAgent


class RAGAgent(BaseAgent):
    def run(self, query: str, user_id: str, top_k: int = 5):
        """Answers questions, loading and saving chat history for the user."""
        print(f"RAGAgent received query: '{query}' for user: '{user_id}'")

        history = self._load_chat_history(user_id)

        query_vector = self.agent_nick.embedding_model.encode(query).tolist()
        search_result = self.agent_nick.qdrant_client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        if not search_result:
            return {"answer": "I could not find any relevant documents to answer your question."}

        context = "".join(
            f"Document ID: {hit.payload.get('record_id', hit.id)}, Score: {hit.score:.2f}\nContent: {hit.payload.get('content', hit.payload.get('summary', ''))}\n---\n"
            for hit in search_result
        )

        history_context = "\n".join([f"Previous Q: {h['query']}\nPrevious A: {h['answer']}" for h in history])

        rag_prompt = f"""You are a helpful procurement assistant.
Considering the chat history and the retrieved documents, answer the user's question concisely.
Cite the Document ID for new information you use.

CHAT HISTORY:
{history_context}

RETRIEVED DOCUMENTS:
{context}
USER QUESTION: {query}
ANSWER:"""

        response = self.call_ollama(rag_prompt, model=self.settings.extraction_model)
        answer = response.get('response', "I am sorry, I could not generate an answer.")

        # Update and save history
        history.append({"query": query, "answer": answer})
        self._save_chat_history(user_id, history)

        return {"answer": answer, "retrieved_documents": [hit.payload for hit in search_result]}

    def _load_chat_history(self, user_id: str) -> list:
        history_key = f"chat_history/{user_id}.json"
        try:
            s3_object = self.agent_nick.s3_client.get_object(Bucket=self.settings.s3_bucket_name, Key=history_key)
            history = json.loads(s3_object['Body'].read().decode('utf-8'))
            print(f"Loaded {len(history)} items from chat history for user '{user_id}'.")
            return history
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"No chat history found for user '{user_id}'. Starting new session.")
                return []
            else:
                raise

    def _save_chat_history(self, user_id: str, history: list):
        history_key = f"chat_history/{user_id}.json"
        bucket_name = getattr(self.settings, "s3_bucket_name", None)
        if not bucket_name:
            print("ERROR: S3 bucket name is not configured. Please set 's3_bucket_name' in settings.")
            return
        try:
            self.agent_nick.s3_client.put_object(
                Bucket=bucket_name,
                Key=history_key,
                Body=json.dumps(history, indent=2)
            )
            print(f"Saved chat history for user '{user_id}' in bucket '{bucket_name}'.")
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                print(f"ERROR: S3 bucket '{bucket_name}' does not exist. Please check configuration.")
            else:
                print(f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}")
        except Exception as e:
            print(f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}")
