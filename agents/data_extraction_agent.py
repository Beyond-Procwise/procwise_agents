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
from typing import Dict, List, Optional, Tuple

import ollama
import pdfplumber
try:  # PyMuPDF is optional; handled gracefully if unavailable
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None
from qdrant_client import models
from sentence_transformers import util
import torch

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
                header, line_items = self._extract_structured_data(text)
                if not header:
                    logger.warning("No header information found in %s", object_key)
                    continue

                doc_type = header.get("doc_type", "Invoice")
                pk_value = (
                    header.get("invoice_id")
                    or header.get("po_id")
                    or header.get("quote_id")
                )
                if not pk_value:
                    logger.error("No primary key found for %s", object_key)
                    continue

                # try to derive vendor/supplier name from document title or header
                if not header.get("vendor_name"):
                    vendor_guess = self._infer_vendor_name(text, object_key)
                    if vendor_guess:
                        header["vendor_name"] = vendor_guess

                # ensure supplier id always populated
                if not header.get("supplier_id"):
                    header["supplier_id"] = header.get("vendor_name")

                data = {
                    "header_data": header,
                    "line_items": line_items,
                    "validation": {
                        "is_valid": True,
                        "confidence_score": 1.0,
                        "notes": "llm parser",
                    },
                }

                product_type = self._classify_product_type(line_items)
                self._vectorize_document(
                    data,
                    text,
                    pk_value,
                    doc_type,
                    product_type,
                    object_key,
                )

                # Persist structured header and line item data to PostgreSQL
                self._persist_to_postgres(header, line_items, doc_type, pk_value)

                results.append(
                    {
                        "object_key": object_key,
                        "id": pk_value,
                        "doc_type": doc_type,
                        "status": "success",
                    }
                )

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
        """Use semantic similarity to locate common header fields."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return {}

        try:
            line_vecs = self.agent_nick.embedding_model.encode(
                lines, convert_to_tensor=True, show_progress_bar=False
            )
        except Exception:
            return {}

        field_synonyms = {
            "invoice_id": ["invoice number", "invoice id", "invoice no"],
            "po_id": ["purchase order", "po number", "po id"],
            "quote_id": ["quote number", "quote id"],
            "vendor_name": ["vendor", "supplier", "from"],
            "supplier_id": ["supplier id", "supplier number"],
            "total_amount": ["total amount", "grand total", "total"],
            "invoice_date": ["invoice date"],
            "due_date": ["due date"],
            "currency": ["currency"],
            "tax_percent": ["tax rate", "tax percent"],
            "tax_amount": ["tax amount", "tax"],
        }
        header: Dict[str, str] = {}
        for field, synonyms in field_synonyms.items():
            try:
                syn_vec = self.agent_nick.embedding_model.encode(
                    synonyms, convert_to_tensor=True, show_progress_bar=False
                )
                field_vec = torch.mean(syn_vec, dim=0, keepdim=True)
                sims = util.cos_sim(field_vec, line_vecs)[0]
                score, idx = torch.max(sims, dim=0)
                idx = int(idx.item())
            except Exception:
                continue
            if score.item() < 0.5:
                continue
            candidate = lines[idx]
            lower_candidate = candidate.lower()
            value = candidate
            for syn in synonyms:
                syn_l = syn.lower()
                if syn_l in lower_candidate:
                    value = candidate[lower_candidate.index(syn_l) + len(syn_l) :]
                    break
            value = value.split(":")[-1].strip(" -#")
            if field in {"total_amount", "tax_percent", "tax_amount"}:
                header[field] = self._clean_numeric(value)
            else:
                header[field] = value

        if "invoice_id" in header:
            header["doc_type"] = "Invoice"
        elif "po_id" in header:
            header["doc_type"] = "Purchase_Order"
        elif "quote_id" in header:
            header["doc_type"] = "Quote"
        else:
            header["doc_type"] = self._classify_doc_type(text)

        prompt = (
            "Extract header details from the following procurement document.\n"
            "Return JSON with any of these keys if present: "
            "invoice_id, po_id, supplier_id, buyer_id, requisition_id, requested_by, "
            "requested_date, invoice_date, due_date, currency, invoice_amount, tax_percent, "
            "tax_amount, invoice_total_incl_tax.\n"
            f"Text:\n{text[:2000]}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(prompt, model=self.extraction_model, format="json")
            llm_header = json.loads(resp.get("response", "{}"))
            header.update({k: v for k, v in llm_header.items() if v})
        except Exception:
            pass

        numeric_fields = {"total_amount", "tax_percent", "tax_amount"}
        numeric_id_fields = {"buyer_id", "supplier_id", "requisition_id"}
        for field in numeric_fields:
            if field in header:
                header[field] = self._clean_numeric(header[field])
        for field in numeric_id_fields:
            if field in header:
                num = self._clean_numeric(header[field])
                header[field] = int(num) if num is not None else None

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

    def _infer_vendor_name(self, text: str, object_key: str | None = None) -> str:
        """Best-effort vendor/supplier name extraction.

        Many procurement documents place the vendor name prominently in the
        title or top header without an explicit label.  This helper searches the
        first few lines for a candidate and falls back to the file name when
        necessary.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:5]:
            if not re.search(r"(invoice|purchase order|quote|bill|statement)", line, re.I):
                return line
        if object_key:
            base = object_key.split("/")[-1].rsplit(".", 1)[0]
            token = re.split(r"[-_]", base)[0]
            if token and not re.search(r"(invoice|po|purchase|quote)", token, re.I):
                return token
        return ""

    def _extract_structured_data(self, text: str) -> Tuple[Dict[str, str], List[Dict]]:
        """Parse header and line items from raw text."""
        header = self._parse_header(text)
        line_items = self._extract_line_items(text, header.get("doc_type", "Invoice"))
        return header, line_items

    def _extract_line_items(self, text: str, doc_type: str) -> List[Dict]:
        """Use the LLM to extract line items from a document."""
        if doc_type == "Invoice":
            prompt = (
                "Extract line items from the following invoice. "
                "Return JSON with a list under the key 'line_items' where each item has "
                "line_no, item_id, quantity, unit_price, tax_percent, line_total, "
                "tax_amount and total_with_tax.\n"
                f"Text:\n{text[:4000]}"
            )
        elif doc_type == "Purchase_Order":
            prompt = (
                "Extract line items from the following purchase order. "
                "Return JSON with a list under the key 'line_items' where each item has "
                "line_number, item_id, item_description, quantity, unit_price, "
                "unit_of_measue, currency, line_total, tax_percent, tax_amount, "
                "total_amount.\n"
                f"Text:\n{text[:4000]}"
            )
        else:
            prompt = (
                f"Extract line items from the following {doc_type}. "
                "Return JSON with a list under the key 'line_items'.\n"
                f"Text:\n{text[:4000]}"
            )
        try:  # pragma: no cover - network call
            response = self.call_ollama(prompt, model=self.extraction_model, format="json")
            data = json.loads(response.get("response", "{}"))
            items = data.get("line_items", [])
            if isinstance(items, list):
                return items
        except Exception:
            pass
        return []

    def _classify_product_type(self, line_items: List[Dict]) -> str:
        descriptions = "\n- ".join(
            item.get("description")
            or item.get("item_description")
            or ""
            for item in line_items
            if item
        )
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

    def _vectorize_document(
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

    def _clean_numeric(self, value: str) -> Optional[float]:
        """Strip non-numeric characters and attempt float casting.

        Parameters
        ----------
        value: str
            Raw numeric string potentially containing currency symbols or
            free-form text.

        Returns
        -------
        Optional[float]
            ``float`` representation if parsing succeeds otherwise ``None``.
        """
        cleaned = re.sub(r"[^0-9.]", "", str(value))
        if not cleaned:
            logger.warning("Unable to parse numeric value '%s'", value)
            return None
        try:
            return float(cleaned)
        except ValueError:
            logger.warning("Unable to parse numeric value '%s'", value)
            return None

    def _sanitize_value(self, value):
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value

    def _persist_to_postgres(
        self,
        header: Dict[str, str],
        line_items: List[Dict],
        doc_type: str,
        pk_value: str,
    ) -> None:
        """Persist extracted header and line items to PostgreSQL."""
        self._persist_header_to_postgres(header, doc_type)
        self._persist_line_items_to_postgres(pk_value, line_items, doc_type, header)

    def _persist_header_to_postgres(self, header: Dict[str, str], doc_type: str) -> None:
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
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                columns = {r[0]: r[1] for r in cur.fetchall()}
                payload = {}
                numeric_types = {
                    "integer",
                    "bigint",
                    "smallint",
                    "numeric",
                    "decimal",
                    "double precision",
                    "real",
                }
                for k, v in header.items():
                    if k not in columns:
                        continue
                    sanitized = self._sanitize_value(v)
                    if sanitized is None:
                        continue
                    col_type = columns[k]
                    if col_type in numeric_types:
                        if isinstance(sanitized, str):
                            sanitized = self._clean_numeric(sanitized)
                        try:
                            if sanitized is None:
                                raise ValueError
                            sanitized = float(sanitized)
                            if col_type in {"integer", "bigint", "smallint"}:
                                sanitized = int(sanitized)
                        except (TypeError, ValueError):
                            logger.warning(
                                "Dropping %s due to type mismatch (%s)", k, v
                            )
                            continue
                    payload[k] = sanitized
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

    def _persist_line_items_to_postgres(
        self,
        pk_value: str,
        line_items: List[Dict],
        doc_type: str,
        header: Dict[str, str],
    ) -> None:
        table_map = {
            "Invoice": ("proc", "invoice_line_items", "invoice_id", "line_no"),
            "Purchase_Order": (
                "proc",
                "po_line_items_agent",
                "po_id",
                "line_number",
            ),
        }
        field_map = {
            "Invoice": [
                "item_id",
                "quantity",
                "unit_price",
                "tax_percent",
                "line_total",
                "tax_amount",
                "total_with_tax",
            ],
            "Purchase_Order": [
                "item_id",
                "item_description",
                "quantity",
                "unit_price",
                "unit_of_measue",
                "currency",
                "line_total",
                "tax_percent",
                "tax_amount",
                "total_amount",
            ],
        }
        target = table_map.get(doc_type)
        fields = field_map.get(doc_type, [])
        if not target or not line_items:
            return
        schema, table, fk_col, line_no_col = target
        try:
            conn = self.agent_nick.get_db_connection()
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                columns = [r[0] for r in cur.fetchall()]
                for idx, item in enumerate(line_items, start=1):
                    line_key = "line_no" if line_no_col == "line_no" else "line_number"
                    payload = {fk_col: pk_value, line_no_col: item.get(line_key, idx)}
                    if doc_type == "Invoice" and "po_id" in columns and header.get("po_id"):
                        payload["po_id"] = header.get("po_id")
                    for key in fields:
                        if key in payload:
                            continue
                        if key in columns and item.get(key) is not None:
                            payload[key] = self._sanitize_value(item[key])
                    cols = ", ".join(payload.keys())
                    placeholders = ", ".join(["%s"] * len(payload))
                    update_cols = ", ".join(
                        f"{c}=EXCLUDED.{c}" for c in payload.keys() if c not in [fk_col, line_no_col]
                    )
                    sql = (
                        f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders}) "
                        f"ON CONFLICT ({fk_col}, {line_no_col}) DO UPDATE SET {update_cols}"
                    )
                    sanitized = {k: self._sanitize_value(v) for k, v in payload.items()}
                    cur.execute(sql, list(sanitized.values()))
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist line items for %s: %s", doc_type, exc)


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
