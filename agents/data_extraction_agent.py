# ProcWise/agents/data_extraction_agent.py
"""Agent responsible for ingesting procurement documents.

The previous version of this module was heavily truncated which made the
pipeline fragile.  This rewrite focuses on a small but fully functional
implementation that fulfils the repository requirements:

* Read PDF based invoices, purchase orders and quotes from S3.
* Extract a handful of key attributes from the text.
* Persist the structured information in PostgreSQL.
* Store a summary of each document in Qdrant for later retrieval by the
  RAG based ``/ask`` endpoint.

The implementation intentionally keeps the heuristics simple – the goal is
not to perfectly understand every possible document, but to provide a
robust skeleton that can be iteratively improved.  The agent only depends on
``pdfplumber`` for text extraction and ``ollama`` for the lightweight
product category classification.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from io import BytesIO
from typing import Dict, List, Optional

import ollama
import pdfplumber
try:  # PyMuPDF is optional; handled gracefully if unavailable
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None
from qdrant_client import models

from agents.base_agent import (
    BaseAgent,
    AgentContext,
    AgentOutput,
    AgentStatus,
)

logger = logging.getLogger(__name__)

HITL_CONFIDENCE_THRESHOLD = 0.85


def _normalize_point_id(raw_id: str) -> int | str:
    """Qdrant allows integer identifiers; strings are hashed deterministically."""
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    if raw_id.isdigit():
        return int(raw_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


class DataExtractionAgent(BaseAgent):
    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.extraction_model = self.settings.extraction_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        """Entry point used by the orchestrator."""
        try:
            s3_prefix = context.input_data.get("s3_prefix")
            s3_object_key = context.input_data.get("s3_object_key")
            data = self._process_documents(s3_prefix, s3_object_key)
            return AgentOutput(status=AgentStatus.SUCCESS, data=data)
        except Exception as exc:
            logger.error("DataExtractionAgent failed: %s", exc)
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    def _process_documents(self, s3_prefix: str | None = None, s3_object_key: str | None = None) -> Dict:
        """Process documents stored under the given prefix.

        Parameters
        ----------
        s3_prefix: str | None
            Optional folder prefix.  When ``None`` all prefixes configured in
            ``settings.s3_prefixes`` are processed.
        s3_object_key: str | None
            When provided only that object is processed; useful for
            re-processing a single document.
        """
        results: List[Dict[str, str]] = []
        prefixes = [s3_prefix] if s3_prefix else self.settings.s3_prefixes

        for prefix in prefixes:
            if s3_object_key and s3_object_key.startswith(prefix):
                keys = [s3_object_key]
            else:
                resp = self.agent_nick.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name, Prefix=prefix
                )
                keys = [obj["Key"] for obj in resp.get("Contents", [])]

            for object_key in keys:
                if not object_key.lower().endswith(".pdf"):
                    continue
                logger.info("Processing %s", object_key)
                try:
                    obj = self.agent_nick.s3_client.get_object(
                        Bucket=self.settings.s3_bucket_name, Key=object_key
                    )
                    file_bytes = obj["Body"].read()
                except Exception as exc:  # pragma: no cover - network failures
                    logger.error("Failed downloading %s: %s", object_key, exc)
                    continue

                text = self._extract_text_from_pdf(file_bytes)
                header = self._parse_header(text)
                if not header:
                    logger.warning("No header information found in %s", object_key)
                    continue

                line_items: List[Dict] = []  # heuristic line-item extraction could be added later
                doc_type = header.get("doc_type", "Invoice")
                pk_value = (
                    header.get("invoice_id")
                    or header.get("po_id")
                    or header.get("quote_id")
                )
                if not pk_value:
                    logger.error("No primary key found for %s", object_key)
                    continue

                # ensure supplier id always populated
                if not header.get("supplier_id"):
                    header["supplier_id"] = header.get("vendor_name")

                data = {
                    "header_data": header,
                    "line_items": line_items,
                    "validation": {
                        "is_valid": True,
                        "confidence_score": 1.0,
                        "notes": "heuristic parser",
                    },
                }

                product_type = self._classify_product_type(line_items)
                self._upsert_to_qdrant(
                    data,
                    text,
                    pk_value,
                    doc_type,
                    product_type,
                    object_key,
                )

                # Persist structured header data to PostgreSQL for downstream analysis
                self._persist_to_postgres(header, doc_type)

                results.append({
                    "object_key": object_key,
                    "id": pk_value,
                    "doc_type": doc_type,
                    "status": "success",
                })

        return {"status": "completed", "details": results}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text using pdfplumber with a PyMuPDF fallback.

        pdfplumber provides high quality layout aware extraction but can be
        slow or fail on certain documents.  PyMuPDF is considerably faster and
        therefore used as a fallback whenever pdfplumber returns no text or
        raises an exception.
        """
        text = ""
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("pdfplumber failed: %s", exc)

        if not text.strip() and fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("PyMuPDF failed extracting text: %s", exc)
                return ""
        elif not text.strip():
            logger.warning("No text extracted from PDF; PyMuPDF not installed")
            return ""
        return text

    def _parse_header(self, text: str) -> Dict[str, str]:
        header: Dict[str, str] = {}
        inv = re.search(r"Invoice(?:\s*(?:No|Number|ID))?[:#\s]*([A-Za-z0-9-]+)", text, re.I)
        if inv:
            header["invoice_id"] = inv.group(1)
            header["doc_type"] = "Invoice"
        po = re.search(r"Purchase\s+Order(?:\s*(?:No|Number|ID))?[:#\s]*([A-Za-z0-9-]+)", text, re.I)
        if po:
            header["po_id"] = po.group(1)
            header["doc_type"] = "Purchase_Order"
        quote = re.search(r"Quote(?:\s*(?:No|Number|ID))?[:#\s]*([A-Za-z0-9-]+)", text, re.I)
        if quote:
            header["quote_id"] = quote.group(1)
            header["doc_type"] = "Quote"
        vendor = re.search(r"(?:Vendor|From|Supplier)[:#\s]*([\w \-]+)", text, re.I)
        if vendor:
            header["vendor_name"] = vendor.group(1).strip()
        supplier = re.search(r"Supplier\s*ID[:#\s]*([A-Za-z0-9-]+)", text, re.I)
        if supplier:
            header["supplier_id"] = supplier.group(1)
        total = re.search(r"Total(?:\s+Amount)?[:#\s]*([\d.,]+)", text, re.I)
        if total:
            try:
                header["total_amount"] = float(total.group(1).replace(",", ""))
            except ValueError:
                pass

        if "doc_type" not in header:
            header["doc_type"] = self._classify_doc_type(text)
        return header

    def _classify_doc_type(self, text: str) -> str:
        """Use the LLM to infer the document type when heuristics fail."""
        prompt = (
            "Identify the document type from the text. Possible values: "
            "invoice, purchase_order, quote, user_agreement, supplier_data, procurement_insight.\n"
            "Respond with JSON {\"doc_type\": \"<type>\"}.\n"
            f"Text snippet:\n{text[:1000]}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(prompt, model=self.extraction_model, format="json")
            doc_type = json.loads(resp.get("response", "{}")).get("doc_type", "other")
            return doc_type.replace(" ", "_").title()
        except Exception:
            return "Other"

    def _classify_product_type(self, line_items: List[Dict]) -> str:
        descriptions = "\n- ".join(item.get("description", "") for item in line_items if item)
        if not descriptions.strip():
            return "General Goods"
        prompt = (
            "Classify these procurement items into a broad category (e.g., IT Hardware, Office Supplies).\n"
            f"Items:\n- {descriptions}\nRespond with JSON {{\"product_category\": \"<category>\"}}"
        )
        try:  # pragma: no cover - network call
            response = self.call_ollama(prompt, model=self.extraction_model, format="json")
            return json.loads(response.get("response", "{}")).get("product_category", "General Goods")
        except Exception:
            return "General Goods"

    def _upsert_to_qdrant(
        self,
        data: Dict,
        full_text: str,
        pk_value: str,
        doc_type: str,
        product_type: str,
        object_key: str,
    ) -> None:
        """Store document chunks and metadata in the vector database."""
        self.agent_nick._initialize_qdrant_collection()
        summary = self._generate_document_summary(data, pk_value, doc_type)
        chunks = self._chunk_text(full_text)

        # Batch encode the chunks so the GPU can be utilised efficiently instead of
        # invoking the embedding model for every iteration.  ``SentenceTransformer``
        # transparently handles batching and will fall back to the CPU if no GPU is
        # available.
        vectors = self.agent_nick.embedding_model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        )

        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "record_id": pk_value,
                "document_type": doc_type.lower(),
                "product_type": product_type.lower(),
                "s3_key": object_key,
                "chunk_id": idx,
                "content": chunk,
                "summary": summary,
            }
            point_id = _normalize_point_id(f"{pk_value}_{idx}")
            points.append(
                models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )

        if points:
            self.agent_nick.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )

    def _chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        """Whitespace-aware text chunking with configurable overlap.

        A small overlap between chunks helps the retriever maintain context
        around boundaries which in turn improves answer quality.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        step = max_chars - overlap if max_chars > overlap else max_chars
        return [text[i : i + max_chars] for i in range(0, len(text), step)]

    def _generate_document_summary(self, data: Dict, pk_value: str, doc_type: str) -> str:
        header = data.get("header_data", {})
        vendor = header.get("vendor_name", "Unknown Vendor")
        total = header.get("total_amount", "unknown amount")
        return f"{doc_type} {pk_value} from {vendor} for {total}"

    def _persist_to_postgres(self, header: Dict[str, str], doc_type: str) -> None:
        """Write the extracted header information to the appropriate table.

        The function performs a lightweight schema introspection to align the
        extracted keys with existing table columns and uses an upsert so that
        reprocessing a document simply updates the stored record.  Any database
        errors are logged but do not interrupt the extraction pipeline.
        """
        table_map = {
            "Invoice": ("proc", "invoice_agent", "invoice_id"),
            "Purchase_Order": ("proc", "purchase_order_agent", "po_id"),
        }
        target = table_map.get(doc_type)
        if not target:
            return
        schema, table, pk_col = target
        try:
            conn = self.agent_nick.get_db_connection()
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                columns = [r[0] for r in cur.fetchall()]
                payload = {k: v for k, v in header.items() if k in columns}
                if not payload:
                    return
                cols = ", ".join(payload.keys())
                placeholders = ", ".join(["%s"] * len(payload))
                update_cols = ", ".join(
                    f"{c}=EXCLUDED.{c}" for c in payload.keys() if c != pk_col
                )
                sql = (
                    f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_cols}"
                )
                cur.execute(sql, list(payload.values()))
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist %s data: %s", doc_type, exc)

    # ------------------------------------------------------------------
    # End of class
    # ------------------------------------------------------------------


if __name__ == "__main__":
    # Simple manual invocation used during local development. The production
    # system always provides an ``AgentContext`` via the orchestrator.
    from agents.base_agent import AgentNick, AgentContext

    agent_nick = AgentNick()
    logging.basicConfig(level=logging.INFO)
    agent = DataExtractionAgent(agent_nick)  # Replace with actual AgentNick instance

    context = AgentContext(
        workflow_id="manual-test",
        agent_id="data_extraction",
        user_id=agent_nick.settings.script_user,
        input_data={"s3_prefix": "Invoices/"},
    )

    result = agent.run(context)
    print(result)
