from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Dict, List, Optional

from agents.data_extraction_agent import DataExtractionAgent
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

configure_gpu()

logger = logging.getLogger(__name__)


class DocumentVectorAgent(DataExtractionAgent):
    """Agent dedicated to vectorising documents without structural parsing.

    The agent downloads documents from the configured S3 bucket, extracts the
    raw text and upserts embeddings into the vector store.  No attempt is made
    to parse headers or line items; this agent is solely responsible for
    enabling Retrieval Augmented Generation.
    """

    AGENTIC_PLAN_STEPS = (
        "Collect eligible documents from the configured S3 prefixes for vector indexing.",
        "Extract raw text snippets and generate embeddings using the local model stack.",
        "Upsert embeddings into Qdrant and report indexing status for retrieval workflows.",
    )

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            s3_prefix = context.input_data.get("s3_prefix")
            s3_object_key = context.input_data.get("s3_object_key")
            details = self._process_documents_vector_only(s3_prefix, s3_object_key)
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={"status": "completed", "details": details},
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("DocumentVectorAgent failed: %s", exc)
            return self._with_plan(
                context,
                AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc)),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_documents_vector_only(
        self, s3_prefix: str | None = None, s3_object_key: str | None = None
    ) -> List[Dict[str, str]]:
        """Process documents and store only their embeddings."""
        results: List[Dict[str, str]] = []
        prefixes = [s3_prefix] if s3_prefix else self.settings.s3_prefixes
        keys: List[str] = []
        for prefix in prefixes:
            if s3_object_key and s3_object_key.startswith(prefix):
                keys.append(s3_object_key)
            else:
                with self._borrow_s3_client() as s3_client:
                    resp = s3_client.list_objects_v2(
                        Bucket=self.settings.s3_bucket_name, Prefix=prefix
                    )
                keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        supported_exts = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}
        keys = [k for k in keys if os.path.splitext(k)[1].lower() in supported_exts]

        pool_cap = getattr(self.agent_nick, "s3_pool_size", os.cpu_count() or 4)
        max_workers = max(1, min(len(keys) or 1, pool_cap, os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._vectorize_single_document, k) for k in keys]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)
        return results

    def _vectorize_single_document(self, object_key: str) -> Optional[Dict[str, str]]:
        if not object_key:
            return None
        logger.info("Vectorising %s", object_key)
        try:
            with self._borrow_s3_client() as s3_client:
                obj = s3_client.get_object(
                    Bucket=self.settings.s3_bucket_name, Key=object_key
                )
            body = obj.get("Body")
            if body is None:
                logger.error("No body returned for %s", object_key)
                return None
            try:
                file_bytes = body.read()
            finally:
                try:
                    body.close()
                except Exception:
                    logger.debug("Failed to close streaming body for %s", object_key, exc_info=True)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed downloading %s: %s", object_key, exc)
            return None
        text_bundle = self._extract_text(file_bytes, object_key)
        if isinstance(text_bundle, str):
            text = text_bundle
        else:
            text = getattr(text_bundle, "full_text", "")
        doc_type = self._classify_doc_type(text)
        self._vectorize_document(text, None, doc_type, None, object_key)
        return {
            "object_key": object_key,
            "id": object_key,
            "doc_type": doc_type or "",
            "status": "vectorized",
        }
