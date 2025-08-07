# ProcWise/services/model_selector.py

import json
import logging
import ollama
import pdfplumber
from io import BytesIO
from botocore.exceptions import ClientError
from typing import List, Dict, Optional
from config.settings import settings
from qdrant_client import models

logger = logging.getLogger(__name__)


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
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.history_manager = ChatHistoryManager(agent_nick.s3_client, agent_nick.settings.s3_bucket_name)
        self.default_llm_model = settings.extraction_model

    def _extract_text_from_uploads(self, files: List[tuple[bytes, str]]) -> str:
        """Extracts text from uploaded PDF files."""
        ad_hoc_context = []
        for content_bytes, filename in files:
            try:
                if filename.lower().endswith('.pdf'):
                    with pdfplumber.open(BytesIO(content_bytes)) as pdf:
                        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                        ad_hoc_context.append(f"--- Content from uploaded file: {filename} ---\n{text}")
            except Exception as e:
                logger.error(f"Failed to process uploaded file {filename}: {e}")
        return "\n\n".join(ad_hoc_context)

    def _generate_follow_ups(self, query: str, answer: str, context: str, model: str) -> list:
        """Generates a list of relevant follow-up questions."""
        logger.info("Generating follow-up questions...")
        prompt = f"""Based on the user's question and the provided answer, generate 3 relevant and concise follow-up questions.
**Context:**
- Original Question: "{query}"
- Answer Provided: "{answer}"
- Retrieved Document Summaries: "{context}"
**Instructions:**
- Your entire response MUST be a single, valid JSON array of strings. Example: ["What was the total amount?", "Who was the vendor?"]"""
        try:
            response = ollama.generate(model=model, prompt=prompt, format='json')
            return list(json.loads(response.get('response', '[]')))
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return []

    def answer_question(self, query: str, user_id: str, model_name: Optional[str] = None,
                        files: Optional[List[tuple[bytes, str]]] = None, doc_type: Optional[str] = None,
                        product_type: Optional[str] = None) -> Dict:
        llm_to_use = model_name or self.default_llm_model
        logger.info(
            f"Answering query with model '{llm_to_use}' and filters: doc_type='{doc_type}', product_type='{product_type}'")

        # --- Build Vector DB Filter with multiple conditions ---
        must_conditions = []
        if doc_type:
            must_conditions.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=doc_type)))
        if product_type:
            must_conditions.append(
                models.FieldCondition(key="product_type", match=models.MatchValue(value=product_type)))
        qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

        # --- Retrieve from Vector DB using the unified collection name ---
        query_vector = self.agent_nick.embedding_model.encode(query).tolist()
        search_results = self.agent_nick.qdrant_client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=5
        )
        retrieved_context = "\n---\n".join(
            [
                f"Document ID: {hit.payload.get('record_id', hit.id)}\nSummary: {hit.payload.get('summary')}"
                for hit in search_results
            ]
        ) if search_results else "No relevant documents found."

        # --- Process Uploaded Files & Chat History ---
        ad_hoc_context = self._extract_text_from_uploads(files) if files else ""
        history = self.history_manager.get_history(user_id)
        history_context = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history])

        # --- Build Final Prompt ---
        prompt = f"""You are a helpful assistant. Answer the user's question using the provided context in this order of priority: Uploaded Files, Retrieved Documents, Chat History.

### Ad-hoc Context from Uploaded Files:
{ad_hoc_context if ad_hoc_context else "No files were uploaded for this query."}

### Retrieved Documents from Knowledge Base:
{retrieved_context}

### Chat History:
{history_context if history_context else "No previous conversation history."}

### User's Question: {query}
### Answer:"""

        # --- Generate Answer and Follow-ups ---
        response = ollama.generate(model=llm_to_use, prompt=prompt)
        answer = response.get('response', "Could not generate an answer.")
        follow_ups = self._generate_follow_ups(query, answer, retrieved_context, llm_to_use)

        # --- Save History to S3 ---
        history.append({"query": query, "answer": answer})
        self.history_manager.save_history(user_id, history)

        return {"answer": answer, "follow_ups": follow_ups,
                "retrieved_documents": [hit.payload for hit in search_results]}
