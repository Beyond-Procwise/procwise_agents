from __future__ import annotations
import logging
import re
import uuid
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import gzip
import concurrent.futures
import pdfplumber
import pandas as pd
import numpy as np
import json

try:
    import fitz  # PyMuPDF (optional)
except Exception:
    fitz = None
try:
    import docx  # DOCX (optional)
except Exception:
    docx = None
try:
    from PIL import Image, UnidentifiedImageError  # OCR (optional)
    import pytesseract
except Exception:
    Image = None
    pytesseract = None
    UnidentifiedImageError = Exception  # type: ignore
try:
    import easyocr  # GPU OCR (optional)
except Exception:
    easyocr = None
_easyocr_reader = None

from qdrant_client import models
from sentence_transformers import util
import torch
from datetime import datetime, timedelta
from dateutil import parser

from agents.document_jsonifier import convert_document_to_json
from utils.nlp import extract_entities
from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from agents.discrepancy_detection_agent import DiscrepancyDetectionAgent
from utils.gpu import configure_gpu
from utils.procurement_schema import extract_structured_content

logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
configure_gpu()

HITL_CONFIDENCE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
INVOICE_SCHEMA = {
    "invoice_id": "text",
    "po_id": "text",
    "supplier_id": "text",
    "buyer_id": "text",
    "requisition_id": "text",
    "requested_by": "text",
    "requested_date": "date",
    "invoice_date": "date",
    "due_date": "date",
    "invoice_paid_date": "date",
    "payment_terms": "text",
    "currency": "character varying(3)",
    "invoice_amount": "numeric(18,2)",
    "tax_percent": "numeric(5,2)",
    "tax_amount": "numeric(18,2)",
    "invoice_total_incl_tax": "numeric(18,2)",
    "exchange_rate_to_usd": "numeric(10,4)",
    "converted_amount_usd": "numeric(18,2)",
    "country": "text",
    "region": "text",
    "invoice_status": "text",
    "ai_flag_required": "text",
    "trigger_type": "text",
    "trigger_context_description": "text",
    "created_date": "timestamp without time zone",
    "vendor_name": "text",
}

INVOICE_LINE_ITEMS_SCHEMA = {
    "invoice_line_id": "text",
    "invoice_id": "text",
    "line_no": "integer",
    "item_id": "text",
    "item_description": "text",
    "quantity": "integer",
    "unit_of_measure": "text",
    "unit_price": "numeric(10,2)",
    "line_amount": "numeric(18,2)",
    "tax_percent": "numeric(5,2)",
    "tax_amount": "numeric(18,2)",
    "total_amount_incl_tax": "numeric(18,2)",
    "po_id": "text",
    "delivery_date": "date",
    "country": "text",
    "region": "text",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
}

PURCHASE_ORDER_SCHEMA = {
    "po_id": "text",
    "supplier_id": "text",
    "buyer_id": "text",
    "requisition_id": "text",
    "requested_by": "text",
    "requested_date": "date",
    "currency": "character varying(3)",
    "order_date": "date",
    "expected_delivery_date": "date",
    "ship_to_country": "text",
    "delivery_region": "text",
    "incoterm": "text",
    "incoterm_responsibility": "text",
    "total_amount": "numeric(18,2)",
    "delivery_address_line1": "text",
    "delivery_address_line2": "text",
    "delivery_city": "text",
    "postal_code": "text",
    "default_currency": "character varying(3)",
    "po_status": "character varying(20)",
    "payment_terms": "character varying(30)",
    "exchange_rate_to_usd": "numeric(18,4)",
    "converted_amount_usd": "numeric(18,4)",
    "ai_flag_required": "character varying(5)",
    "trigger_type": "character varying(30)",
    "trigger_context_description": "text",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
    "contract_id": "text",
}

PO_LINE_ITEMS_SCHEMA = {
    "po_line_id": "text",
    "po_id": "text",
    "line_number": "integer",
    "item_id": "text",
    "item_description": "text",
    "quantity": "integer",
    "unit_price": "numeric(18,2)",
    "unit_of_measue": "text",
    "currency": "character varying(3)",
    "line_total": "numeric(18,2)",
    "tax_percent": "smallint",
    "tax_amount": "numeric(18,2)",
    "total_amount": "numeric(18,2)",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
}

QUOTE_SCHEMA = {
    "quote_id": "text",
    "supplier_id": "text",
    "buyer_id": "text",
    "quote_date": "date",
    "validity_date": "date",
    "currency": "character varying(3)",
    "total_amount": "numeric(18,2)",
    "tax_percent": "numeric(5,2)",
    "tax_amount": "numeric(18,2)",
    "total_amount_incl_tax": "numeric(18,2)",
    "po_id": "text",
    "country": "text",
    "region": "text",
    "ai_flag_required": "character varying(5)",
    "trigger_type": "character varying(30)",
    "trigger_context_description": "text",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
}

QUOTE_LINE_ITEMS_SCHEMA = {
    "quote_line_id": "text",
    "quote_id": "text",
    "line_number": "integer",
    "item_id": "text",
    "item_description": "text",
    "quantity": "integer",
    "unit_of_measure": "text",
    "unit_price": "numeric(18,2)",
    "line_total": "numeric(18,2)",
    "tax_percent": "numeric(5,2)",
    "tax_amount": "numeric(18,2)",
    "total_amount": "numeric(18,2)",
    "currency": "character varying(3)",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
}

CONTRACT_SCHEMA = {
    "contract_id": "text",
    "contract_title": "text",
    "contract_type": "text",
    "supplier_id": "text",
    "buyer_org_id": "text",
    "contract_start_date": "date",
    "contract_end_date": "date",
    "currency": "text",
    "total_contract_value": "numeric(18,2)",
    "spend_category": "text",
    "business_unit_id": "text",
    "cost_centre_id": "text",
    "is_amendment": "text",
    "parent_contract_id": "text",
    "auto_renew_flag": "text",
    "renewal_term": "text",
    "contract_lifecycle_status": "text",
    "jurisdiction": "text",
    "governing_law": "text",
    "contract_signatory_name": "text",
    "contract_signatory_role": "text",
    "payment_terms": "text",
    "risk_assessment_completed": "text",
    "created_date": "timestamp without time zone",
    "created_by": "text",
    "last_modified_by": "text",
    "last_modified_date": "timestamp without time zone",
}


SCHEMA_MAP = {
    "Invoice": {"header": INVOICE_SCHEMA, "line_items": INVOICE_LINE_ITEMS_SCHEMA},
    "Purchase_Order": {"header": PURCHASE_ORDER_SCHEMA, "line_items": PO_LINE_ITEMS_SCHEMA},
    "Quote": {"header": QUOTE_SCHEMA, "line_items": QUOTE_LINE_ITEMS_SCHEMA},
    "Contract": {"header": CONTRACT_SCHEMA, "line_items": {}},  # no line items
}
# ---------------------------------------------------------------------------
# DOC CONTEXT / KEYWORDS
# ---------------------------------------------------------------------------
PRODUCT_KEYWORDS = {
    "it hardware": ["laptop", "desktop", "notebook", "printer", "server", "monitor", "keyboard", "mouse"],
    "it software": ["software", "license", "subscription", "saas"],
    "office supplies": ["paper", "pen", "stapler", "notebook", "folder"],
}

DOC_TYPE_CONTEXT = {
    "Purchase_Order": ("Purchase Order: A buyer sends a purchase order to a vendor to procure goods or services. "
                       "Once the vendor accepts, the PO becomes a binding agreement."),
    "Invoice": ("Invoice: After fulfilling the PO, the vendor sends an invoice to the buyer. The invoice details the "
                "goods/services delivered and requests payment."),
    "Quote": ("A vendor sends a quote to a potential buyer or client offering goods or services at a specific price "
              "and under certain conditions before a purchase is agreed."),
    "Contract": ("A contract is sent by the party proposing the agreement—often the seller, service provider, or their "
                 "representative—for review and signature by the other parties."),
}
DOC_CONTEXT_TEXT = "\n".join(DOC_TYPE_CONTEXT.values())

DOC_TYPE_KEYWORDS = {
    "Invoice": ["invoice", "amount due", "bill"],
    "Purchase_Order": ["purchase order", "po number", "purchase requisition"],
    "Quote": ["quote", "quotation", "estimate"],
    "Contract": ["contract", "agreement", "terms"],
}

UNIQUE_ID_PATTERNS = {
    "Invoice": r"invoice\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Purchase_Order": r"(?:purchase order|po)\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Quote": r"quote\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Contract": r"contract\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
}

# ---------------------------------------------------------------------------
# NEW: robust alias maps + supplier profiles + totals stop
# ---------------------------------------------------------------------------
FIELD_ALIASES = {
    "invoice_id": [r"\bInvoice\s*(?:No\.?|#|Number)\b", r"\bTax\s*Invoice\s*No\b", r"\bBill\s*No\b"],
    "po_id": [r"\bPO\s*(?:No\.?|#|Number)\b", r"\bPurchase\s*Order\s*(?:No|Number)\b"],
    "invoice_date": [r"\bInvoice\s*Date\b", r"\bInv\s*Date\b", r"\bBilling\s*Date\b"],
    "due_date": [r"\bDue\s*Date\b", r"\bPayment\s*Due\b"],
    "invoice_paid_date": [r"\bPaid\s*Date\b", r"\bPayment\s*Date\b"],
    "payment_terms": [r"\bPayment\s*Terms\b", r"\bTerms\b"],
    "currency": [r"\bCurrency\b"],
    "invoice_amount": [r"\bAmount\s*Due\b", r"\bInvoice\s*Amount\b"],
    "tax_percent": [r"\bTax\s*%\b", r"\bVAT\s*%\b", r"\bGST\s*%\b"],
    "tax_amount": [r"\bTax\s*Amount\b", r"\bVAT\b", r"\bGST\b"],
    "invoice_total_incl_tax": [r"\bGrand\s*Total\b", r"\bInvoice\s*Total\b", r"\bTotal\s*\(Incl\.?\s*Tax\)\b", r"\bTotal\s*Amount\b"],
    "requested_date": [r"\bRequested\s*Date\b"],
    "requested_by": [r"\bRequested\s*By\b", r"\bRequester\b"],
    "buyer_id": [r"\bBuyer\s*(?:ID|No)\b"],
    "supplier_id": [r"\bSupplier\s*(?:ID|No)\b"],
    "country": [r"\bCountry\b"],
    "region": [r"\bRegion\b"]
}
LINE_HEADERS_ALIASES = {
    "line_no": [r"\bLine\s*#\b", r"\bLine\b", r"\bItem\s*#\b"],
    "item_id": [r"\bItem\s*(?:Code|ID)\b", r"\bSKU\b", r"\bPart\s*No\b", r"\bProduct\s*Code\b"],
    "item_description": [r"\bDescription\b", r"\bItem\s*Description\b"],
    "quantity": [r"\bQty\b", r"\bQuantity\b"],
    "unit_of_measure": [r"\bUOM\b", r"\bUnit\b", r"\bUnit\s*of\s*Measure\b"],
    "unit_price": [r"\bUnit\s*Price\b", r"\bRate\b", r"\bPrice\b", r"\bUnit\s*Cost\b"],
    "line_amount": [r"\bLine\s*Total\b", r"\bLine\s*Amount\b", r"\bAmount\b"],
    "tax_percent": [r"\bTax\s*%\b", r"\bVAT\s*%\b"],
    "tax_amount": [r"\bTax\s*Amt\b", r"\bTax\b", r"\bVAT\b"],
    "total_amount_incl_tax": [r"\bTotal\b", r"\bTotal\s*Incl\s*Tax\b", r"\bGross\b"],
    "currency": [r"\bCurrency\b"],
    "line_total": [r"\bLine\s*Total\b", r"\bAmount\b", r"\bTotal\b"],
    "total_amount": [r"\bTotal\b", r"\bTotal\s*Amount\b"]
}
TOTALS_STOP_WORDS = re.compile(r"\b(Subtotal|Tax|VAT|GST|Total|Amount\s*Due)\b", re.I)

SUPPLIER_PROFILES = {
    # Example profile – extend as you learn templates
    "ACME-UK": {
        "date_format_hint": "DD/MM/YYYY",
        "currency_hint": "GBP",
        "label_overrides": {
            "invoice_total_incl_tax": [r"\bGrand\s*Total\b", r"\bTotal\s*Invoice\b"]
        },
        "table_header_regex": [r"^Line\s*#\s+Item\s+Description\s+Qty\s+Rate\s+Amount$"]
    }
}

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------
def _get_easyocr_reader():
    """Lazily initialise an EasyOCR reader if the library is installed."""
    global _easyocr_reader
    if easyocr is None:
        return None
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        except Exception:
            _easyocr_reader = None
    return _easyocr_reader
# === QDRANT: Idempotent collection initialization ===
def _initialize_qdrant_collection_idempotent(
    qdrant_client,
    collection_name: str,
    vector_size: int,
    distance: str = "COSINE",
) -> None:
    """
    Create the collection only if it doesn't already exist.
    Works across qdrant-client versions. No-op if exists.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse
    try:
        # Fast existence check:
        try:
            _ = qdrant_client.get_collection(collection_name)
            return  # Already exists
        except Exception:
            # If the get_collection not available/failed, list as fallback
            try:
                coll_list = qdrant_client.get_collections().collections
                if any(getattr(c, "name", None) == collection_name for c in coll_list):
                    return
            except Exception:
                # proceed to attempt creation
                pass

        # Create collection with correct Distance enum (uppercase)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=int(vector_size),
                distance=getattr(models.Distance, distance.upper()),
            ),
        )
    except UnexpectedResponse as e:
        # 409 Conflict (already exists) -> safe to ignore
        if getattr(e, "status_code", None) == 409:
            return
        # Anything else: re-raise to be handled by caller
        raise
    except Exception:
        # Quietly allow caller to attempt upsert (and handle 404 there)
        import logging
        logging.warning("Qdrant init skipped or failed (will handle on upsert).", exc_info=True)

def _normalize_point_id(raw_id: str) -> int | str:
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    if raw_id.isdigit():
        return int(raw_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

def _normalize_label(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value).lower()

def _maybe_decompress(content: bytes) -> bytes:
    try:
        if content[:2] == b"\x1f\x8b":
            return gzip.decompress(content)
    except Exception:
        logger.warning("Failed to decompress gzip content")
    return content

def _dict_to_text(data: Dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in data.items() if v not in (None, ""))

# ---------------------------------------------------------------------------
# NEW: Idempotent Qdrant collection initializer (fixes 409 Conflict)
# ---------------------------------------------------------------------------
def _initialize_qdrant_collection_idempotent(client, collection_name: str, vector_size: int, distance: str = "Cosine"):
    """Create collection only if it doesn't already exist."""
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            return
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=getattr(models.Distance, distance)),
        )
    except Exception as e:
        # Non-fatal; skip creation if race/exists, log for visibility
        logger.warning(f"Qdrant init skipped or failed: {e}")

# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------
class DataExtractionAgent(BaseAgent):
    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.extraction_model = self.settings.extraction_model

    # ============================ PUBLIC API ==============================
    def run(self, context: AgentContext) -> AgentOutput:
        try:
            s3_prefix = context.input_data.get("s3_prefix")
            s3_object_key = context.input_data.get("s3_object_key")
            data = self._process_documents(s3_prefix, s3_object_key)
            docs = data.get("details", [])
            processing_issues = [
                {
                    "doc_type": doc.get("doc_type", "Unknown"),
                    "record_id": doc.get("invoice_id")
                    or doc.get("po_id")
                    or doc.get("id")
                    or doc.get("object_key"),
                    "reason": "; ".join(doc.get("issues", [])) or "document not converted",
                }
                for doc in docs
                if doc.get("status") in {"error", "failed"}
            ]
            if processing_issues:
                discrepancy_result = self._run_discrepancy_detection(
                    docs, context, processing_issues
                )
            else:
                discrepancy_result = self._run_discrepancy_detection(docs, context)

            if discrepancy_result.status != AgentStatus.SUCCESS:
                err = discrepancy_result.error or "discrepancy detection failed"
                data["summary"] = {
                    "documents_provided": len(docs),
                    "documents_valid": 0,
                    "documents_with_discrepancies": len(docs),
                }
                return AgentOutput(status=AgentStatus.FAILED, data=data, error=err)

            mismatches = discrepancy_result.data.get("mismatches", [])
            data["summary"] = {
                "documents_provided": len(docs),
                "documents_valid": len(docs) - len(mismatches),
                "documents_with_discrepancies": len(mismatches)
                + len(processing_issues),
            }
            if mismatches:
                data["mismatches"] = mismatches
            return AgentOutput(status=AgentStatus.SUCCESS, data=data)

        except Exception as exc:
            logger.error("DataExtractionAgent failed: %s", exc)
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    def _run_discrepancy_detection(
        self,
        docs: List[Dict[str, Any]],
        context: AgentContext,
        processing_issues: List[Dict[str, str]] | None = None,
    ) -> AgentOutput:
        if not docs:
            return AgentOutput(status=AgentStatus.SUCCESS, data={"mismatches": []})
        disc_agent = DiscrepancyDetectionAgent(self.agent_nick)
        disc_context = AgentContext(
            workflow_id=context.workflow_id,
            agent_id="discrepancy_detection",
            user_id=context.user_id,
            input_data={
                "extracted_docs": docs,
                "processing_issues": processing_issues or [],
            },
            parent_agent=context.agent_id,
            routing_history=context.routing_history.copy(),
        )
        return disc_agent.execute(disc_context)

    def _process_documents(self, s3_prefix: str | None = None, s3_object_key: str | None = None) -> Dict:
        results: List[Dict[str, str]] = []
        prefixes = [s3_prefix] if s3_prefix else self.settings.s3_prefixes
        keys: List[str] = []
        for prefix in prefixes:
            if s3_object_key and s3_object_key.startswith(prefix):
                keys.append(s3_object_key)
            else:
                resp = self.agent_nick.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name, Prefix=prefix
                )
                keys.extend(obj["Key"] for obj in resp.get("Contents", []))

        supported_exts = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}
        keys = [k for k in keys if os.path.splitext(k)[1].lower() in supported_exts]

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(self._process_single_document, k) for k in keys]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)

        return {"status": "completed", "details": results}

    def _process_single_document(self, object_key: str) -> Optional[Dict[str, str]]:
        if not object_key:
            return None
        logger.info("Processing %s", object_key)
        try:
            obj = self.agent_nick.s3_client.get_object(Bucket=self.settings.s3_bucket_name, Key=object_key)
            file_bytes = obj["Body"].read()
        except Exception as exc:
            logger.error("Failed downloading %s: %s", object_key, exc)
            return None

        text = self._extract_text(file_bytes, object_key)
        doc_type = self._classify_doc_type(text)
        product_type = self._classify_product_type(text)
        unique_id = self._extract_unique_id(text, doc_type)
        vendor_name = self._infer_vendor_name(text, object_key)
        if not unique_id:
            unique_id = uuid.uuid4().hex[:8]

        # Consult vector database for related documents to leverage prior context
        similar_docs = self.vector_search(text, top_k=1)
        if similar_docs:
            logger.debug(
                "Found similar document %s while processing %s",
                similar_docs[0].id,
                object_key,
            )

        # Vectorize raw document content for search regardless of type
        self._vectorize_document(text, unique_id, doc_type, product_type, object_key)

        header: Dict[str, Any] = {"doc_type": doc_type, "product_type": product_type}
        if vendor_name:
            header["vendor_name"] = vendor_name
        line_items: List[Dict[str, Any]] = []
        pk_value = unique_id
        data: Optional[Dict[str, Any]] = None

        if doc_type in {"Purchase_Order", "Invoice", "Quote", "Contract"}:
            header, line_items = self._extract_structured_data(text, file_bytes, doc_type)
            header["doc_type"] = doc_type
            header["product_type"] = product_type

            if doc_type == "Invoice" and not header.get("invoice_id"):
                header["invoice_id"] = unique_id
            elif doc_type == "Purchase_Order" and not header.get("po_id"):
                header["po_id"] = unique_id
            elif doc_type == "Quote" and not header.get("quote_id"):
                header["quote_id"] = unique_id
            elif doc_type == "Contract" and not header.get("contract_id"):
                header["contract_id"] = unique_id

            header = self._sanitize_party_names(header)
            pk_value = (
                header.get("invoice_id")
                or header.get("po_id")
                or header.get("quote_id")
                or header.get("contract_id")
                or unique_id
            )

            if not header.get("supplier_id") and header.get("vendor_name"):
                header["supplier_id"] = header.get("vendor_name")

            ok = header.get("_validation", {}).get("ok", True)
            conf = header.get("_validation", {}).get("confidence", 0.8)
            notes = header.get("_validation", {}).get("notes", [])

            data = {
                "header_data": header,
                "line_items": line_items,
                "validation": {
                    "is_valid": bool(ok),
                    "confidence_score": float(conf),
                    "notes": "; ".join(notes) if notes else "ok",
                },
            }

            self._persist_to_postgres(header, line_items, doc_type, pk_value)
            self._vectorize_structured_data(header, line_items, doc_type, pk_value, product_type)

        result = {"object_key": object_key, "id": pk_value or object_key, "doc_type": doc_type or "", "status": "success"}
        if data:
            result["data"] = data
        return result

    # ============================ EXTRACTION HELPERS ======================
    def _extract_text_from_pdf(self, file_bytes: bytes) -> str:
        file_bytes = _maybe_decompress(file_bytes)
        if not file_bytes.startswith(b"%PDF"):
            logger.warning("Provided bytes do not look like a PDF; attempting image extraction")
            return self._extract_text_from_image(file_bytes, allow_pdf_fallback=False)

        lines: List[str] = []
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words() or []
                    row_dict: Dict[float, Dict[str, List[str]]] = {}
                    for word in words:
                        row_y = round(word["top"], 0)
                        x_center = (word["x0"] + word["x1"]) / 2
                        if row_y not in row_dict:
                            row_dict[row_y] = {"left": [], "right": []}
                        side = "left" if x_center < page.width / 2 else "right"
                        row_dict[row_y][side].append(word["text"])
                    if row_dict:
                        for y in sorted(row_dict.keys()):
                            left_text = " ".join(row_dict[y]["left"]).strip()
                            right_text = " ".join(row_dict[y]["right"]).strip()
                            if right_text:
                                lines.append(f"{left_text} | {right_text}".strip())
                            elif left_text:
                                lines.append(left_text)
                        for table in page.extract_tables() or []:
                            for row in table:
                                row_text = " ".join(cell or "" for cell in row).strip()
                                if row_text:
                                    lines.append(row_text)
                    else:
                        # OCR fallback
                        if easyocr is not None:
                            try:
                                page_image = page.to_image(resolution=300).original
                                img_arr = np.array(page_image)
                                reader = _get_easyocr_reader()
                                ocr_lines = reader.readtext(img_arr, detail=0, paragraph=True)
                                lines.extend(ln.strip() for ln in ocr_lines if ln.strip())
                            except Exception as ocr_exc:
                                logger.warning("EasyOCR failed: %s", ocr_exc)
                        elif Image is not None and pytesseract is not None:
                            try:
                                page_image = page.to_image(resolution=300).original
                                ocr_text = pytesseract.image_to_string(page_image)
                                lines.extend(ln.strip() for ln in ocr_text.splitlines() if ln.strip())
                            except Exception as ocr_exc:
                                logger.warning("OCR failed: %s", ocr_exc)
        except Exception as exc:
            logger.warning("pdfplumber failed: %s", exc)

        if not lines and fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            except Exception as exc:
                logger.error("PyMuPDF failed extracting text: %s", exc)
                img_text = self._extract_text_from_image(file_bytes, allow_pdf_fallback=False)
                return img_text
        elif not lines:
            logger.warning("No text extracted from PDF; PyMuPDF not installed")
            return ""

        return "\n".join(lines)

    def _extract_text_from_docx(self, file_bytes: bytes) -> str:
        if docx is None:
            logger.warning("python-docx not installed; cannot extract DOCX text")
            return ""
        try:
            document = docx.Document(BytesIO(file_bytes))
            return "\n".join(par.text for par in document.paragraphs)
        except Exception as exc:
            logger.error("Failed extracting DOCX text: %s", exc)
            return ""

    def _extract_text_from_image(self, file_bytes: bytes, allow_pdf_fallback: bool = True) -> str:
        if Image is None:
            logger.warning("PIL not installed; cannot OCR image")
            return ""
        try:
            img = Image.open(BytesIO(file_bytes))
        except UnidentifiedImageError:
            if allow_pdf_fallback:
                logger.warning("Not a valid image; attempting PDF extraction fallback")
                return self._extract_text_from_pdf(file_bytes)
            logger.warning("Provided bytes are not a valid image")
            return ""
        except Exception as exc:
            logger.error("Failed OCR on image: %s", exc)
            return ""
        if easyocr is not None:
            try:
                arr = np.array(img)
                reader = _get_easyocr_reader()
                ocr_lines = reader.readtext(arr, detail=0, paragraph=True)
                return "\n".join(ocr_lines)
            except Exception as exc:
                logger.warning("EasyOCR failed: %s", exc)
        if pytesseract is not None:
            try:
                return pytesseract.image_to_string(img)
            except Exception as exc:
                logger.error("Failed OCR on image: %s", exc)
                return ""
        logger.warning("pytesseract not installed; cannot OCR image")
        return ""

    def _extract_text(self, file_bytes: bytes, object_key: str) -> str:
        ext = os.path.splitext(object_key)[1].lower()
        if ext == ".pdf":
            return self._extract_text_from_pdf(file_bytes)
        if ext in {".doc", ".docx"}:
            return self._extract_text_from_docx(file_bytes)
        if ext in {".png", ".jpg", ".jpeg"}:
            return self._extract_text_from_image(file_bytes)
        logger.warning("Unsupported document type '%s' for %s", ext, object_key)
        return ""

    # ============================ NEW LAYOUT HELPERS ======================
    def _extract_layout_blocks(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for pageno, page in enumerate(pdf.pages, start=1):
                    words = page.extract_words() or []
                    for w in words:
                        blocks.append({
                            "text": w.get("text", "").strip(),
                            "x0": w.get("x0", 0.0),
                            "y0": w.get("top", 0.0),
                            "x1": w.get("x1", 0.0),
                            "y1": w.get("bottom", 0.0),
                            "page": pageno,
                            "width": page.width,
                            "height": page.height,
                        })
        except Exception as exc:
            logger.debug("Layout extraction failed: %s", exc)
        return blocks

    def _nearest_value_by_label(self, blocks: List[Dict[str, Any]], label_res: List[str], search_dx: float = 180.0, search_dy: float = 45.0) -> Optional[str]:
        if not blocks:
            return None
        patterns = [re.compile(pat, re.I) for pat in label_res]
        candidates = []
        for b in blocks:
            bt = b["text"]
            if not bt:
                continue
            if any(p.search(bt) for p in patterns):
                x0, y0 = b["x1"], b["y0"] - 4
                x1, y1 = b["x1"] + search_dx, b["y0"] + search_dy
                region_words = [w["text"] for w in blocks if (w["x0"] >= x0 and w["x1"] <= x1 and w["y0"] >= y0 and w["y1"] <= y1 and w["page"] == b["page"])]
                text = " ".join(region_words).strip(" :#-")
                if text:
                    dist = min(abs((w["x0"] - x0)) + abs((w["y0"] - y0)) for w in blocks if w["text"] in region_words) if region_words else 1e9
                    candidates.append((dist, text))
        if not candidates:
            return None
        candidates.sort(key=lambda z: z[0])
        return candidates[0][1]

    def _apply_supplier_profile_overrides(self, header: Dict[str, Any]) -> Dict[str, Any]:
        key = (header.get("supplier_id") or header.get("vendor_name") or "").strip()
        profile = SUPPLIER_PROFILES.get(key) if key else None
        if not profile:
            return header
        overrides = profile.get("label_overrides") or {}
        for fld, res in overrides.items():
            if fld in FIELD_ALIASES:
                FIELD_ALIASES[fld] = list(set(FIELD_ALIASES[fld] + res))
        header.setdefault("_currency_hint", profile.get("currency_hint"))
        header.setdefault("_date_format_hint", profile.get("date_format_hint"))
        return header

    def _confidence_from_method(self, method: str) -> float:
        base = {
            "layout_label_proximity": 0.92,
            "semantic_regex": 0.80,
            "table_detector": 0.85,
            "llm_fill": 0.70,
            "ner": 0.60
        }.get(method, 0.50)
        return base

    def _record_conf(self, conf: Dict[str, float], key: str, method: str, value_present: bool):
        if not value_present:
            return
        conf[key] = max(conf.get(key, 0.0), self._confidence_from_method(method))

    def _validate_business_rules(self, doc_type: str, header: Dict[str, Any], line_items: List[Dict[str, Any]]) -> Tuple[bool, float, List[str]]:
        notes = []
        conf = 0.0
        if doc_type == "Invoice":
            try:
                la = sum(float(self._clean_numeric(li.get("line_amount") or 0) or 0) for li in line_items)
                ta = sum(float(self._clean_numeric(li.get("tax_amount") or 0) or 0) for li in line_items)
                total = self._clean_numeric(header.get("invoice_total_incl_tax"))
                if total is not None:
                    ok = abs((la + ta) - total) <= 0.02
                    if ok:
                        conf += 0.08
                    else:
                        notes.append("Line totals + tax do not match invoice_total_incl_tax (±0.02).")
                else:
                    notes.append("invoice_total_incl_tax missing.")
            except Exception:
                notes.append("Failed total reconciliation.")
        try:
            due = self._clean_date(header.get("due_date"))
            paid = self._clean_date(header.get("invoice_paid_date"))
            if due and not paid and due < datetime.utcnow().date():
                header["invoice_status"] = header.get("invoice_status") or "Overdue"
                conf += 0.02
        except Exception:
            pass
        overall_ok = len([n for n in notes if "do not match" in n]) == 0
        return overall_ok, min(1.0, conf), notes

    # ============================ HEADER PARSING ==========================
    def _parse_header(self, text: str, file_bytes: bytes | None = None) -> Dict[str, Any]:
        header: Dict[str, Any] = {}
        conf: Dict[str, float] = {}

        # 1) Layout: label → proximity (precise)
        blocks = self._extract_layout_blocks(file_bytes) if file_bytes else []
        if blocks:
            for field, regexes in FIELD_ALIASES.items():
                val = self._nearest_value_by_label(blocks, regexes)
                if val:
                    if field in {"invoice_amount", "tax_amount", "tax_percent", "invoice_total_incl_tax"}:
                        header[field] = self._clean_numeric(val)
                    else:
                        header[field] = val
                    self._record_conf(conf, field, "layout_label_proximity", True)

        # 2) Semantic fallback
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            try:
                line_vecs = self.agent_nick.embedding_model.encode(lines, convert_to_tensor=True, show_progress_bar=False)
                field_synonyms = {
                    "invoice_id": ["invoice number", "invoice id", "invoice no"],
                    "po_id": ["purchase order", "po number", "po id"],
                    "requested_by": ["requested by", "requester"],
                    "requested_date": ["requested date", "request date"],
                    "invoice_date": ["invoice date"],
                    "due_date": ["due date"],
                    "invoice_paid_date": ["paid date", "payment date", "invoice paid date"],
                    "payment_terms": ["payment terms", "terms"],
                    "currency": ["currency"],
                    "invoice_amount": ["invoice amount", "amount due"],
                    "tax_percent": ["tax rate", "tax percent"],
                    "tax_amount": ["tax amount", "tax"],
                    "invoice_total_incl_tax": ["total including tax", "grand total", "invoice total", "total amount"],
                    "country": ["country"],
                    "region": ["region"],
                }
                for field, synonyms in field_synonyms.items():
                    if field in header and header[field]:
                        continue
                    try:
                        syn_vec = self.agent_nick.embedding_model.encode(synonyms, convert_to_tensor=True, show_progress_bar=False)
                        field_vec = torch.mean(syn_vec, dim=0, keepdim=True)
                        sims = util.cos_sim(field_vec, line_vecs)[0]
                        score, idx = torch.max(sims, dim=0)
                        idx = int(idx.item())
                        if score.item() >= 0.52:
                            candidate = lines[idx]
                            lower_candidate = candidate.lower()
                            value = candidate
                            for syn in synonyms:
                                syn_l = syn.lower()
                                if syn_l in lower_candidate:
                                    value = candidate[lower_candidate.index(syn_l) + len(syn_l):]
                                    break
                            value = value.split(":")[-1].strip(" -#")
                            if field in {"invoice_amount", "tax_percent", "tax_amount", "invoice_total_incl_tax"}:
                                header[field] = self._clean_numeric(value)
                            else:
                                header[field] = value
                            self._record_conf(conf, field, "semantic_regex", True)
                    except Exception:
                        continue
            except Exception:
                pass

        # 3) NER supplement
        ner_header = self._extract_header_with_ner(text)
        for k, v in ner_header.items():
            header.setdefault(k, v)
            if k in ner_header:
                self._record_conf(conf, k, "ner", True)

        # doc type
        if "invoice_id" in header:
            header["doc_type"] = "Invoice"
        elif "po_id" in header:
            header["doc_type"] = "Purchase_Order"
        else:
            header["doc_type"] = self._classify_doc_type(text)

        # normalize numerics
        for f in {"invoice_amount", "tax_percent", "tax_amount", "invoice_total_incl_tax", "exchange_rate_to_usd", "converted_amount_usd"}:
            if f in header:
                header[f] = self._clean_numeric(header[f])

        # supplier profile
        header = self._apply_supplier_profile_overrides(header)
        header["_field_confidence"] = conf
        return header

    def _extract_header_with_ner(self, text: str) -> Dict[str, Any]:
        entities = extract_entities(text)
        header: Dict[str, Any] = {}
        for ent in entities:
            label = ent.get("entity_group")
            word = ent.get("word", "").strip()
            if not word:
                continue
            if label == "ORG" and "vendor_name" not in header:
                header["vendor_name"] = word
            elif label == "DATE":
                if "invoice_date" not in header:
                    header["invoice_date"] = word
                elif "due_date" not in header:
                    header["due_date"] = word
            elif label == "MONEY" and "invoice_total_incl_tax" not in header:
                header["invoice_total_incl_tax"] = self._clean_numeric(word)
        return header

    # ============================ LINE ITEMS ==============================
    def _extract_line_items_from_pdf_tables(self, file_bytes: bytes, doc_type: str) -> List[Dict]:
        items: List[Dict] = []
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words() or []
                    if not words:
                        continue
                    # find header row
                    header_row = None
                    for y in sorted(set(round(w["top"], 0) for w in words)):
                        row_words = [w for w in words if round(w["top"], 0) == y]
                        header_text = " ".join(w["text"].strip().lower() for w in row_words)
                        hits = 0
                        required = ["item_description", "quantity", "unit_price"]
                        for req in required:
                            aliases = LINE_HEADERS_ALIASES.get(req, [])
                            if any(re.search(p, header_text, re.I) for p in aliases):
                                hits += 1
                        if hits >= 2:
                            header_row = row_words
                            break
                    if not header_row:
                        continue

                    # columns spans
                    header_spans: List[Tuple[float, float, str]] = []
                    for logical_col, aliases in LINE_HEADERS_ALIASES.items():
                        for w in header_row:
                            txt = w["text"].lower()
                            if any(re.search(p, txt, re.I) for p in aliases):
                                header_spans.append((w["x0"], w["x1"], logical_col))
                                break
                    header_spans = sorted(header_spans, key=lambda t: (t[0] + t[1]) / 2.0)
                    if not header_spans:
                        continue

                    # rows below header
                    below_rows = [w for w in words if w["top"] > min(w["top"] for w in header_row) + 2]
                    by_y: Dict[int, List[Dict[str, Any]]] = {}
                    for w in below_rows:
                        y = int(round(w["top"], 0))
                        by_y.setdefault(y, []).append(w)

                    pending_desc = None
                    for y in sorted(by_y.keys()):
                        row = by_y[y]
                        row_text = " ".join(w["text"] for w in row).strip()
                        if TOTALS_STOP_WORDS.search(row_text):
                            break
                        row_obj: Dict[str, Any] = {}
                        for w in row:
                            mid = (w["x0"] + w["x1"]) / 2.0
                            best = None
                            bestd = 1e9
                            for (x0, x1, label) in header_spans:
                                cx = (x0 + x1) / 2.0
                                d = abs(mid - cx)
                                if d < bestd:
                                    bestd = d
                                    best = label
                            if best:
                                row_obj.setdefault(best, [])
                                row_obj[best].append(w["text"])
                        row_obj = {k: " ".join(v).strip() for k, v in row_obj.items()}

                        non_desc_keys = [k for k in row_obj.keys() if k != "item_description" and row_obj[k]]
                        if ("item_description" in row_obj and row_obj.get("item_description") and not non_desc_keys):
                            if items:
                                items[-1]["item_description"] = f'{items[-1].get("item_description","")} {row_obj["item_description"]}'.strip()
                            else:
                                pending_desc = row_obj["item_description"]
                            continue
                        if pending_desc and "item_description" in row_obj:
                            row_obj["item_description"] = f'{pending_desc} {row_obj["item_description"]}'.strip()
                            pending_desc = None

                        row_norm = self._normalize_line_item_fields([row_obj], doc_type)[0]
                        for k in ["quantity", "unit_price", "line_amount", "tax_percent", "tax_amount", "total_amount_incl_tax", "line_total", "total_amount"]:
                            if k in row_norm:
                                row_norm[k] = self._clean_numeric(row_norm[k])
                        items.append(row_norm)
        except Exception as exc:
            logger.warning("Layout line-item extraction failed: %s", exc)
        return items

    # ============================ STRUCTURED FLOW =========================
    def _normalize_header_fields(self, header: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        if isinstance(header, str):
            try:
                header = json.loads(header)
            except Exception:
                header = {}
        elif not isinstance(header, dict):
            header = {}
        alias_map = {
            "Invoice": {
                "invoice_total": "invoice_amount",
                "total_amount": "invoice_amount",
                "total": "invoice_total_incl_tax",
                "grand_total": "invoice_total_incl_tax",
                "vendor": "vendor_name",
                "supplier": "supplier_name",
                "recipient": "receiver_name",
                "bill_to": "buyer_id",
                "to": "supplier_id",
            },
            "Purchase_Order": {
                "po_number": "po_id",
                "po_no": "po_id",
                "purchase_order_id": "po_id",
                "order_number": "po_id",
                "vendor": "vendor_name",
                "supplier": "supplier_name",
                "recipient": "receiver_name",
                "ship_to": "delivery_address_line1",
                "bill_to": "buyer_id",
                "to": "supplier_id",
            },
            "Quote": {
                "quote_number": "quote_id",
                "quote_no": "quote_id",
                "quote#": "quote_id",
                "quotation": "quote_id",
                "supplier": "supplier_id",
                "vendor": "supplier_id",
                "vendor_name": "supplier_id",
                "bill_to": "buyer_id",
                "buyer": "buyer_id",
                "valid_until": "validity_date",
                "validity": "validity_date",
                "expiry_date": "validity_date",
                "total": "total_amount",
                "grand_total": "total_amount_incl_tax",
                "amount_due": "total_amount_incl_tax",
            },
            "Contract": {
                "title": "contract_title",
                "contract name": "contract_title",
                "start date": "contract_start_date",
                "effective date": "contract_start_date",
                "end date": "contract_end_date",
                "expiry date": "contract_end_date",
                "expiration date": "contract_end_date",
                "supplier": "supplier_id",
                "vendor": "supplier_id",
                "category": "spend_category",
                "business unit": "business_unit_id",
                "cost center": "cost_centre_id",
                "signatory": "contract_signatory_name",
                "signatory name": "contract_signatory_name",
                "signatory role": "contract_signatory_role",
                "payment terms": "payment_terms",
            },
        }
        mapping_raw = alias_map.get(doc_type, {})
        def _norm_alias(name: str) -> str:
            return re.sub(r"[^a-z0-9]", "", name.lower())

        mapping: Dict[str, str] = {}
        for alias, target in mapping_raw.items():
            mapping[_norm_alias(alias)] = target
            mapping.setdefault(_norm_alias(target), target)

        normalised: Dict[str, Any] = {}
        for key, value in header.items():
            norm_key = _norm_alias(key)
            target = mapping.get(norm_key, key)
            normalised[target] = value
        if normalised.get("vendor_name") and not normalised.get("supplier_id"):
            normalised["supplier_id"] = normalised["vendor_name"]
        return normalised

    def _sanitize_party_names(self, header: Dict[str, Any]) -> Dict[str, Any]:
        for field in ("vendor_name", "supplier_id", "supplier_name"):
            val = header.get(field)
            if val and "purchase" in str(val).lower():
                header[field] = ""
        if header.get("vendor_name") and not header.get("supplier_id"):
            header["supplier_id"] = header["vendor_name"]
        return header

    def _reconcile_header_from_lines(
        self, header: Dict[str, Any], line_items: List[Dict[str, Any]], doc_type: str
    ) -> Dict[str, Any]:
        if not line_items:
            return header

        reconciled = dict(header)

        def _sum_field(fields: List[str]) -> Optional[float]:
            values: List[float] = []
            for field in fields:
                for item in line_items:
                    raw = item.get(field)
                    if raw in (None, ""):
                        continue
                    num = self._clean_numeric(raw)
                    if num is not None:
                        values.append(float(num))
                if values:
                    break
            if values:
                return float(sum(values))
            return None

        def _maybe_round(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return round(value, 2)

        currency_values = {
            str(item.get("currency")).strip()
            for item in line_items
            if item.get("currency") not in (None, "")
        }
        if len(currency_values) == 1 and not reconciled.get("currency"):
            reconciled["currency"] = currency_values.pop()

        if doc_type == "Invoice":
            subtotal = _sum_field(["line_amount", "line_total"])
            tax_total = _sum_field(["tax_amount"])
            gross = _sum_field(["total_amount_incl_tax", "total_amount"])
            if gross is None and subtotal is not None and tax_total is not None:
                gross = subtotal + tax_total
            if subtotal is not None and not reconciled.get("invoice_amount"):
                reconciled["invoice_amount"] = _maybe_round(subtotal)
            if tax_total is not None and not reconciled.get("tax_amount"):
                reconciled["tax_amount"] = _maybe_round(tax_total)
            if gross is not None and not reconciled.get("invoice_total_incl_tax"):
                reconciled["invoice_total_incl_tax"] = _maybe_round(gross)
            if (
                subtotal not in (None, 0)
                and tax_total is not None
                and not reconciled.get("tax_percent")
            ):
                try:
                    reconciled["tax_percent"] = _maybe_round((tax_total / subtotal) * 100.0)
                except ZeroDivisionError:
                    pass
        elif doc_type == "Purchase_Order":
            total = _sum_field(["total_amount", "line_total"])
            if total is not None and not reconciled.get("total_amount"):
                reconciled["total_amount"] = _maybe_round(total)
        elif doc_type == "Quote":
            line_total = _sum_field(["line_total", "total_amount"])
            if line_total is not None and not reconciled.get("total_amount"):
                reconciled["total_amount"] = _maybe_round(line_total)
            tax_total = _sum_field(["tax_amount"])
            if tax_total is not None and not reconciled.get("tax_amount"):
                reconciled["tax_amount"] = _maybe_round(tax_total)
            if (
                line_total not in (None, 0)
                and tax_total is not None
                and not reconciled.get("tax_percent")
            ):
                try:
                    reconciled["tax_percent"] = _maybe_round((tax_total / line_total) * 100.0)
                except ZeroDivisionError:
                    pass
            gross = _sum_field(["total_amount_incl_tax"])
            if gross is None and line_total is not None and tax_total is not None:
                gross = line_total + tax_total
            if gross is not None and not reconciled.get("total_amount_incl_tax"):
                reconciled["total_amount_incl_tax"] = _maybe_round(gross)

        return reconciled

    def _normalize_line_item_fields(self, items: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        alias_map = {
            "Invoice": {
                "line": "line_no",
                "lineno": "line_no",
                "linenumber": "line_no",
                "item": "item_description",
                "description": "item_description",
                "itemdescription": "item_description",
                "product": "item_description",
                "part": "item_id",
                "partno": "item_id",
                "itemcode": "item_id",
                "sku": "item_id",
                "qty": "quantity",
                "quantity": "quantity",
                "units": "quantity",
                "uom": "unit_of_measure",
                "unit": "unit_of_measure",
                "unitofmeasure": "unit_of_measure",
                "unitprice": "unit_price",
                "price": "unit_price",
                "rate": "unit_price",
                "amount": "line_amount",
                "lineamount": "line_amount",
                "extendedprice": "line_amount",
                "subtotal": "line_amount",
                "tax": "tax_amount",
                "taxamount": "tax_amount",
                "taxamt": "tax_amount",
                "vat": "tax_amount",
                "gst": "tax_amount",
                "taxpercent": "tax_percent",
                "taxrate": "tax_percent",
                "taxpercentage": "tax_percent",
                "total": "total_amount_incl_tax",
                "totalamount": "total_amount_incl_tax",
                "totalincltax": "total_amount_incl_tax",
                "totalwithtax": "total_amount_incl_tax",
                "totalamountincltax": "total_amount_incl_tax",
                "currency": "currency",
            },
            "Purchase_Order": {
                "line": "line_number",
                "lineno": "line_number",
                "linenumber": "line_number",
                "item": "item_description",
                "description": "item_description",
                "itemdescription": "item_description",
                "product": "item_description",
                "part": "item_id",
                "partno": "item_id",
                "itemcode": "item_id",
                "sku": "item_id",
                "qty": "quantity",
                "quantity": "quantity",
                "units": "quantity",
                "unitprice": "unit_price",
                "price": "unit_price",
                "rate": "unit_price",
                "uom": "unit_of_measue",
                "unit": "unit_of_measue",
                "unitofmeasure": "unit_of_measue",
                "amount": "line_total",
                "lineamount": "line_total",
                "linetotal": "line_total",
                "total": "total_amount",
                "totalamount": "total_amount",
                "currency": "currency",
                "quote": "quote_number",
                "quoteno": "quote_number",
            },
            "Quote": {
                "line": "line_number",
                "lineno": "line_number",
                "linenumber": "line_number",
                "item": "item_description",
                "description": "item_description",
                "itemdescription": "item_description",
                "product": "item_description",
                "part": "item_id",
                "partno": "item_id",
                "itemcode": "item_id",
                "sku": "item_id",
                "qty": "quantity",
                "quantity": "quantity",
                "units": "quantity",
                "uom": "unit_of_measure",
                "unit": "unit_of_measure",
                "unitofmeasure": "unit_of_measure",
                "unitprice": "unit_price",
                "price": "unit_price",
                "rate": "unit_price",
                "amount": "line_total",
                "lineamount": "line_total",
                "linetotal": "line_total",
                "total": "total_amount",
                "totalamount": "total_amount",
                "totalincltax": "total_amount",
                "currency": "currency",
            },
        }
        mapping_raw = alias_map.get(doc_type, {})

        def _norm_key(key: str) -> str:
            return re.sub(r"[^a-z0-9]", "", key.lower())

        mapping: Dict[str, str] = {}
        for alias, target in mapping_raw.items():
            mapping[_norm_key(alias)] = target
            mapping.setdefault(_norm_key(target), target)

        normalised_items: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    continue
            if not isinstance(item, dict):
                continue
            normalised: Dict[str, Any] = {}
            for key, value in item.items():
                norm_key = _norm_key(str(key))
                target = mapping.get(norm_key)
                lower_key = str(key).lower()
                if not target and "tax" in lower_key:
                    if "%" in lower_key or "percent" in lower_key or "rate" in lower_key:
                        target = mapping.get(_norm_key("tax_percent"), "tax_percent")
                    else:
                        target = mapping.get(_norm_key("tax_amount"), "tax_amount")
                if not target and "total" in lower_key:
                    if "incl" in lower_key or "with" in lower_key:
                        target = mapping.get(_norm_key("total_amount_incl_tax"), "total_amount_incl_tax")
                    elif doc_type in {"Purchase_Order", "Quote"}:
                        target = mapping.get(_norm_key("total_amount"), "total_amount")
                    else:
                        target = mapping.get(_norm_key("line_amount"), "line_amount")
                if not target and "amount" in lower_key and doc_type == "Invoice":
                    target = mapping.get(_norm_key("line_amount"), "line_amount")
                normalised[target or key] = value
            normalised_items.append(normalised)
        return normalised_items

    def _enrich_contract_fields(self, text: str, header: Dict[str, Any]) -> Dict[str, Any]:
        # Best-effort patterns for common contract fields
        pairs = [
            (
                "contract_start_date",
                r"\b(?:Effective|Start)\s+Date[:\s]+([A-Za-z0-9,/\- ]{4,})",
            ),
            (
                "contract_end_date",
                r"(?:\bExpiry|Expiration|End)\s+Date[:\s]+([A-Za-z0-9,/\- ]{4,})",
            ),
            ("governing_law", r"\bGoverning\s+Law[:\s]+([A-Za-z ,&]{3,})"),
            ("jurisdiction", r"\bJurisdiction[:\s]+([A-Za-z ,&]{3,})"),
            (
                "contract_signatory_name",
                r"\bSigned\s+by[:\s]+([A-Za-z ,.'-]{3,})",
            ),
            ("contract_signatory_role", r"\bTitle[:\s]+([A-Za-z ,.'-]{3,})"),
            (
                "payment_terms",
                r"\bPayment\s+Terms[:\s]+([A-Za-z0-9 ,.'-]{3,})",
            ),
        ]
        low = text
        for key, pat in pairs:
            if header.get(key):
                continue
            m = re.search(pat, low, re.I | re.S)
            if m:
                group_idx = 1 if m.lastindex else 0
                header[key] = m.group(group_idx).strip()
        return header

    def _extract_structured_data(self, text: str, file_bytes: bytes, doc_type: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Step 1: schema-aware extraction seeded from procurement reference
        structured = extract_structured_content(text, doc_type)
        header_seed = self._normalize_header_fields(structured.get("header", {}), doc_type)
        line_items_seed = self._normalize_line_item_fields(structured.get("line_items", []), doc_type)

        # Step 2: lightweight JSONifier to catch explicit key:value pairs
        data = convert_document_to_json(text, doc_type)
        header_seed.update(self._normalize_header_fields(data.get("header_data", {}), doc_type))

        # Step 3: layout-driven extraction for headers and line items
        header_layout = self._parse_header(text, file_bytes)
        header = {**header_layout, **header_seed}
        line_items = line_items_seed or self._extract_line_items_from_pdf_tables(file_bytes, doc_type)
        header = self._reconcile_header_from_lines(header, line_items, doc_type)

        # Step 4: LLM strict fill for any remaining gaps
        header, line_items = self._fill_missing_fields_with_llm(text, doc_type, header, line_items)
        header = self._reconcile_header_from_lines(header, line_items, doc_type)
        header = self._sanitize_party_names(header)
        if doc_type == "Contract":
            header = self._enrich_contract_fields(text, header)
        header, line_items = self._validate_and_cast(header, line_items, doc_type)

        ok, conf_boost, notes = self._validate_business_rules(doc_type, header, line_items)
        field_conf = header.pop("_field_confidence", {})
        base = np.mean(list(field_conf.values())) if field_conf else 0.75
        overall_conf = float(min(1.0, base + conf_boost))
        header["_validation"] = {"ok": ok, "notes": notes, "confidence": overall_conf, "field_confidence": field_conf}
        return header, line_items

    def _fill_missing_fields_with_llm(self, text: str, doc_type: str, header: Dict[str, Any], line_items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        schema = SCHEMA_MAP.get(doc_type, {})
        header_fields = list(schema.get("header", {}).keys())
        missing_header = [f for f in header_fields if not header.get(f)]
        need_items = (not line_items) and bool(schema.get("line_items"))

        if not missing_header and not need_items:
            return header, line_items

        context_hint = DOC_TYPE_CONTEXT.get(doc_type, "")
        prompt = (
            f"You are an information extraction engine for {doc_type}. {context_hint} "
            "Return ONLY valid JSON. Do not include explanations.\n"
            "Rules:\n"
            "- Fill ONLY the missing header fields listed below; leave others as provided.\n"
            "- For dates, use YYYY-MM-DD.\n"
            "- Currency must be a 3-letter ISO code.\n"
            "- Numbers must be plain decimals without commas or symbols.\n"
            "- If unknown, set null (do not guess).\n\n"
            f"Missing header fields: {', '.join(missing_header) if missing_header else 'none'}\n"
        )
        if need_items:
            prompt += "Also extract 'line_items' as a list of objects. Include description/qty/price amounts when available.\n"
        prompt += "\nExisting data:\n" + json.dumps({"header_data": header, "line_items": line_items}) + "\n\nDocument:\n" + text

        payload = {}
        for _ in range(2):
            try:
                resp = self.call_ollama(prompt=prompt, model=self.extraction_model, format="json")
                payload = json.loads(resp.get("response", "{}"))
                if isinstance(payload, dict) and "header_data" in payload and "line_items" in payload:
                    break
            except Exception:
                continue

        llm_header = self._normalize_header_fields(payload.get("header_data", {}), doc_type)
        for key in missing_header:
            if key in llm_header and llm_header[key] not in (None, "", []):
                header[key] = llm_header[key]
                header.setdefault("_field_confidence", {})[key] = self._confidence_from_method("llm_fill")

        if need_items and isinstance(payload.get("line_items"), list) and payload["line_items"]:
            llm_items = self._normalize_line_item_fields(payload["line_items"], doc_type)
            if llm_items:
                line_items = llm_items

        return header, line_items

    # ============================ VECTORIZING =============================
    def _ensure_qdrant_collection(self) -> None:
        """
        Determine vector size reliably and ensure the Qdrant collection exists.
        """
        # Get target collection name
        collection = self.settings.qdrant_collection_name

        # Determine embedding dimension safely
        dim = None
        emb = getattr(self.agent_nick, "embedding_model", None)
        if emb is not None:
            # sentence-transformers usually has this:
            get_dim = getattr(emb, "get_sentence_embedding_dimension", None)
            if callable(get_dim):
                try:
                    dim = int(get_dim())
                except Exception:
                    dim = None
            if dim is None:
                # Fallback: encode a token and inspect shape
                try:
                    vec = emb.encode(["__dim_probe__"], normalize_embeddings=True, show_progress_bar=False)
                    # vec shape: (1, D)
                    dim = int(getattr(vec, "shape", [None, None])[1]) if hasattr(vec, "shape") else int(len(vec[0]))
                except Exception:
                    pass
        if not dim:
            # Sensible default if all else fails
            dim = 768

        # Initialize idempotently
        _initialize_qdrant_collection_idempotent(
            self.agent_nick.qdrant_client,
            collection_name=collection,
            vector_size=dim,
            distance="COSINE",
        )

    def _vectorize_document(
            self,
            full_text: str,
            pk_value: str | None,
            doc_type: Any,
            product_type: Any,
            object_key: str,
    ) -> None:
        """Store document chunks and metadata in the vector database."""
        if not full_text:
            return

        # Ensure collection exists (idempotent)
        try:
            self._ensure_qdrant_collection()
        except Exception:
            logger.warning("Qdrant init skipped or failed (will attempt upsert-retry).", exc_info=True)

        chunks = self._chunk_text(full_text)
        if not chunks:
            return

        vectors = self.agent_nick.embedding_model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        )

        summary = full_text[:200]
        record_id = pk_value or object_key

        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "record_id": record_id,
                "document_type": _normalize_label(doc_type),
                "product_type": _normalize_label(product_type),
                "s3_key": object_key,
                "chunk_id": idx,
                "content": chunk,
                "summary": summary,
            }
            point_id = _normalize_point_id(f"{record_id}_{idx}")
            points.append(
                models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )

        if not points:
            return

        # Upsert with create-on-404 retry
        try:
            self.agent_nick.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
        except Exception as e:
            # If collection missing, create then retry once
            msg = str(e)
            if "doesn't exist" in msg or "does not exist" in msg or "Not found" in msg:
                logger.info("Qdrant collection missing — creating and retrying upsert...")
                self._ensure_qdrant_collection()
                self.agent_nick.qdrant_client.upsert(
                    collection_name=self.settings.qdrant_collection_name,
                    points=points,
                    wait=True,
                )
            else:
                raise

    def _vectorize_structured_data(
            self,
            header: Dict[str, Any],
            line_items: List[Dict[str, Any]],
            doc_type: str,
            pk_value: str,
            product_type: Any,
    ) -> None:
        """Embed header and line-item content for retrieval."""
        if not pk_value:
            return

        # Ensure collection exists (idempotent)
        try:
            self._ensure_qdrant_collection()
        except Exception:
            logger.warning("Qdrant init skipped or failed (will attempt upsert-retry).", exc_info=True)

        texts: List[str] = []
        meta: List[Tuple[str, Dict[str, Any]]] = []

        # Header payload
        header_text = _dict_to_text(header)
        if header_text:
            texts.append(header_text)
            meta.append(
                (
                    _normalize_point_id(f"{pk_value}_header"),
                    {
                        "record_id": pk_value,
                        "document_type": _normalize_label(doc_type),
                        "product_type": _normalize_label(product_type),
                        "data_type": "header",
                        "content": header_text,
                    },
                )
            )

        # Line item payloads
        for idx, item in enumerate(line_items, start=1):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    continue
            if not isinstance(item, dict):
                continue
            item_text = _dict_to_text(item)
            if not item_text:
                continue
            texts.append(item_text)
            meta.append(
                (
                    _normalize_point_id(f"{pk_value}_line_{idx}"),
                    {
                        "record_id": pk_value,
                        "document_type": _normalize_label(doc_type),
                        "product_type": _normalize_label(product_type),
                        "data_type": "line_item",
                        "line_number": idx,
                        "content": item_text,
                    },
                )
            )

        if not texts:
            return

        vectors = self.agent_nick.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        points = [
            models.PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for (pid, payload), vec in zip(meta, vectors)
        ]

        # Upsert with create-on-404 retry
        try:
            self.agent_nick.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
        except Exception as e:
            msg = str(e)
            if "doesn't exist" in msg or "does not exist" in msg or "Not found" in msg:
                logger.info("Qdrant collection missing — creating and retrying upsert...")
                self._ensure_qdrant_collection()
                self.agent_nick.qdrant_client.upsert(
                    collection_name=self.settings.qdrant_collection_name,
                    points=points,
                    wait=True,
                )
            else:
                raise

    # ============================ TEXT CHUNKING ===========================
    def _chunk_text(self, text: str, max_tokens: int = 256, overlap: int = 20) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            step = max_tokens - overlap if max_tokens > overlap else max_tokens
            chunks = []
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i: i + max_tokens]
                chunks.append(enc.decode(chunk_tokens))
            return chunks
        except Exception:
            max_chars = max_tokens
            step = max_chars - overlap if max_chars > overlap else max_chars
            return [text[i: i + max_chars] for i in range(0, len(text), step)]

    # ============================ NORMALIZE / VALIDATE ====================
    def _clean_numeric(self, value: str | int | float) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).strip()
        if not value_str:
            return None
        if not any(ch.isdigit() for ch in value_str):
            return None
        trimmed = value_str.strip()
        is_negative = trimmed.startswith("(") and trimmed.endswith(")")
        value_str = value_str.replace(",", "")
        numbers = re.findall(r"\d*\.\d+|\d+", value_str)
        if not numbers:
            logger.debug("Unable to parse numeric value '%s'", value)
            return None
        num_str = numbers[0] if "%" in value_str else numbers[-1]
        try:
            num = float(num_str)
            return -num if is_negative else num
        except ValueError:
            logger.debug("Unable to parse numeric value '%s'", value)
            return None

    def _clean_date(self, value: str) -> Optional[datetime.date]:
        try:
            value_str = str(value).strip()
            if not value_str:
                return None
            value_str = re.sub(r"[^\w\s:/\-.]", " ", value_str)
            if not any(ch.isdigit() for ch in value_str) and not re.search(
                r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", value_str, re.I,
            ):
                return None
            match_sub = re.search(r"(\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}|\d{4}-\d{2}-\d{2})", value_str)
            if match_sub:
                value_str = match_sub.group(1)
            value_str = re.sub(r"([A-Za-z])\.\b", r"\1", value_str)
            match = re.match(r"(.+?)\s*\+\s*(\d+)\s*days", value_str, re.I)
            if match:
                base = parser.parse(match.group(1), fuzzy=True)
                offset = int(match.group(2))
                return (base + timedelta(days=offset)).date()
            return parser.parse(value_str, fuzzy=True).date()
        except Exception:
            logger.debug("Unable to parse date value '%s'", value)
            return None

    def _clean_text(self, value: str) -> str:
        if value is None:
            return ""
        cleaned = re.sub(r"[^\w\s\-.,:/@]", "", str(value))
        return re.sub(r"\s+", " ", cleaned).strip()

    def _sanitize_value(self, value, key: Optional[str] = None):
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        numeric_fields = {
            "quantity", "unit_price", "tax_percent", "tax_amount", "line_total", "line_amount",
            "total_with_tax", "total_amount", "total_amount_incl_tax", "total",
            "invoice_amount", "invoice_total_incl_tax", "exchange_rate_to_usd", "converted_amount_usd",
        }
        date_fields = {
            "invoice_date", "due_date", "po_date", "requested_date", "invoice_paid_date", "delivery_date",
            "order_date", "expected_delivery_date", "created_date", "last_modified_date",
        }
        currency_fields = {"currency", "default_currency"}
        if key:
            lower = key.lower()
            if lower in numeric_fields:
                return self._clean_numeric(value)
            if lower in date_fields:
                return self._clean_date(value)
            if lower in currency_fields:
                if isinstance(value, str):
                    val = value.strip().upper()
                    symbol_map = {"$": "USD", "£": "GBP", "€": "EUR"}
                    if val in symbol_map:
                        return symbol_map[val]
                    val = re.sub(r"[^A-Z]", "", val)
                    return val[:3] if val else None
                return None
        if isinstance(value, str):
            cleaned = self._clean_text(value)
            return cleaned or None
        return value

    def _cast_sql_type(self, value: Any, sql_type: str):
        if value is None:
            return None
        sql_type = sql_type.lower()
        try:
            if "numeric" in sql_type or sql_type in {"integer", "smallint"}:
                num = self._clean_numeric(value)
                if num is None:
                    return None
                # Handle precision/scale limits like numeric(5,2)
                m = re.match(r"numeric\((\d+),(\d+)\)", sql_type)
                if m:
                    precision, scale = map(int, m.groups())
                    limit = 10 ** (precision - scale)
                    if abs(num) >= limit:
                        logger.warning(
                            "Numeric overflow for value %s with type %s", value, sql_type
                        )
                        return None
                    num = round(num, scale)
                if sql_type in {"integer", "smallint"}:
                    return int(num)
                return float(num)
            if sql_type == "integer":
                return int(self._clean_numeric(value) or 0)
            if "date" in sql_type and "timestamp" not in sql_type:
                dt = self._clean_date(value)
                return dt.isoformat() if dt else None
            if "timestamp" in sql_type:
                dt = parser.parse(str(value))
                return dt.isoformat()
        except Exception:
            return None
        return value

    def _validate_and_cast(self, header: Dict[str, Any], line_items: List[Dict[str, Any]], doc_type: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        schemas = SCHEMA_MAP.get(doc_type, {})
        header_schema = schemas.get("header", {})
        line_schema = schemas.get("line_items", {})

        cast_header: Dict[str, Any] = {}
        for k, v in header.items():
            if k in header_schema:
                val = self._sanitize_value(v, k)
                cast_header[k] = self._cast_sql_type(val, header_schema[k])
            else:
                cast_header[k] = v

        cast_lines: List[Dict[str, Any]] = []
        for item in line_items:
            cast_item: Dict[str, Any] = {}
            for k, v in item.items():
                if k in line_schema:
                    val = self._sanitize_value(v, k)
                    cast_item[k] = self._cast_sql_type(val, line_schema[k])
                else:
                    cast_item[k] = v
            if cast_item:
                cast_lines.append(cast_item)

        return cast_header, cast_lines

    # ============================ PERSISTENCE =============================
    def _persist_to_postgres(self, header: Dict[str, str], line_items: List[Dict], doc_type: str, pk_value: str) -> None:
        pk_map = {
            "Invoice": "invoice_id",
            "Purchase_Order": "po_id",
            "Quote": "quote_id",
            "Contract": "contract_id",
        }
        if isinstance(pk_value, str):
            pk_value = self._clean_text(pk_value)
        pk_col = pk_map.get(doc_type)
        if pk_col and pk_value:
            header.setdefault(pk_col, pk_value)
        header, line_items = self._validate_and_cast(header, line_items, doc_type)
        pk_value = header.get(pk_col, pk_value) if pk_col else pk_value

        try:
            conn = self.agent_nick.get_db_connection()
            with conn:
                # Persist the header first; if it fails we do not attempt line items
                if not self._persist_header_to_postgres(header, doc_type, conn):
                    conn.rollback()
                    return
                self._persist_line_items_to_postgres(pk_value, line_items, doc_type, header, conn)
        except Exception as exc:
            logger.error("Failed to persist %s data: %s", doc_type, exc)

    def _has_unique_constraint(self, cur, schema: str, table: str, columns: List[str]) -> bool:
        if not columns:
            return False
        try:
            cur.execute(
                """
                SELECT tc.constraint_name, array_agg(ccu.column_name ORDER BY ccu.column_name)
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema=%s AND tc.table_name=%s
                  AND tc.constraint_type IN ('UNIQUE','PRIMARY KEY')
                GROUP BY tc.constraint_name
                """,
                (schema, table),
            )
            target = sorted(columns)
            for _name, cols in cur.fetchall():
                if sorted(cols) == target:
                    return True
        except Exception:
            return False
        return False

    def _persist_header_to_postgres(self, header: Dict[str, str], doc_type: str, conn=None) -> bool:
        table_map = {
            "Invoice": ("proc", "invoice_agent", "invoice_id"),
            "Purchase_Order": ("proc", "purchase_order_agent", "po_id"),
            "Quote": ("proc", "quote_agent", "quote_id"),
            "Contract": ("proc", "contracts", "contract_id"),
        }
        target = table_map.get(doc_type)
        if not target:
            return False
        schema, table, pk_col = target
        close_conn = False
        if conn is None:
            conn = self.agent_nick.get_db_connection()
            close_conn = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                columns = {r[0]: r[1] for r in cur.fetchall()}
                payload = {}
                numeric_types = {"integer", "bigint", "smallint", "numeric", "decimal", "double precision", "real"}
                for k, v in header.items():
                    if k not in columns:
                        continue
                    sanitized = self._sanitize_value(v, k)
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
                            logger.warning("Dropping %s due to type mismatch (%s)", k, v)
                            continue
                    payload[k] = sanitized
                if not payload:
                    return False
                cols = ", ".join(payload.keys())
                placeholders = ", ".join(["%s"] * len(payload))
                update_cols = ", ".join(f"{c}=EXCLUDED.{c}" for c in payload.keys() if c != pk_col)
                sql_base = f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders}) "
                if self._has_unique_constraint(cur, schema, table, [pk_col]):
                    if update_cols:
                        sql = sql_base + f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_cols}"
                    else:
                        sql = sql_base + f"ON CONFLICT ({pk_col}) DO NOTHING"
                else:
                    sql = sql_base + "ON CONFLICT DO NOTHING"
                cur.execute(sql, list(payload.values()))
            if close_conn:
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to persist %s data: %s", doc_type, exc)
            if close_conn:
                conn.rollback()
            return False
        finally:
            if close_conn:
                conn.close()

    def _persist_line_items_to_postgres(self, pk_value: str, line_items: List[Dict], doc_type: str, header: Dict[str, str], conn=None) -> None:
        table_map = {
            "Invoice": ("proc", "invoice_line_items_agent", "invoice_id", "line_no"),
            "Purchase_Order": ("proc", "po_line_items_agent", "po_id", "line_number"),
            "Quote": ("proc", "quote_line_items_agent", "quote_id", "line_number"),
        }
        field_map = {
            "Invoice": {
                "item_id": "item_id",
                "item_description": "item_description",
                "quantity": "quantity",
                "unit_of_measure": "unit_of_measure",
                "unit_price": "unit_price",
                "line_amount": "line_amount",
                "tax_percent": "tax_percent",
                "tax_amount": "tax_amount",
                "total_amount_incl_tax": "total_amount_incl_tax",
            },
            "Purchase_Order": {
                "item_id": "item_id",
                "item_description": "item_description",
                "quantity": "quantity",
                "unit_price": "unit_price",
                "unit_of_measue": "unit_of_measure",
                "currency": "currency",
                "line_total": "line_total",
                "tax_percent": "tax_percent",
                "tax_amount": "tax_amount",
                "total_amount": "total_amount",
            },
            "Quote": {
                "item_id": "item_id",
                "item_description": "item_description",
                "quantity": "quantity",
                "unit_of_measure": "unit_of_measure",
                "unit_price": "unit_price",
                "line_total": "line_total",
                "tax_percent": "tax_percent",
                "tax_amount": "tax_amount",
                "total_amount": "total_amount",
                "currency": "currency",
            },
        }

        target = table_map.get(doc_type)
        field_map = field_map.get(doc_type, {})
        if not target or not line_items:
            return
        schema, table, fk_col, line_no_col = target
        close_conn = False
        if conn is None:
            conn = self.agent_nick.get_db_connection()
            close_conn = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                columns = [r[0] for r in cur.fetchall()]
                numeric_fields = {
                    "quantity", "unit_price", "tax_percent", "tax_amount", "line_total", "line_amount",
                    "total_with_tax", "total_amount_incl_tax", "total_amount",
                }
                for idx, item in enumerate(line_items, start=1):
                    line_key = "line_no" if line_no_col == "line_no" else "line_number"
                    raw_line = item.get(line_key)
                    if raw_line in (None, ""):
                        line_value = idx
                    else:
                        cleaned = self._clean_numeric(raw_line)
                        line_value = int(cleaned) if cleaned is not None else idx
                    payload = {fk_col: pk_value, line_no_col: line_value}
                    if doc_type == "Invoice":
                        if "po_id" in columns and header.get("po_id"):
                            payload["po_id"] = header.get("po_id")
                        for extra in ["delivery_date", "country", "region"]:
                            if extra in columns and header.get(extra):
                                payload[extra] = header.get(extra)
                    if doc_type == "Purchase_Order" and "po_line_id" in columns:
                        payload.setdefault("po_line_id", f"{pk_value}-{line_value}")
                    if doc_type == "Invoice" and "invoice_line_id" in columns:
                        payload.setdefault("invoice_line_id", f"{pk_value}-{line_value}")
                    if doc_type == "Quote" and "quote_line_id" in columns:
                        payload.setdefault("quote_line_id", f"{pk_value}-{line_value}")
                    for col, source in field_map.items():
                        if col in payload:
                            continue
                        if col in columns and item.get(source) is not None:
                            payload[col] = item[source]
                    sanitized = {}
                    for k, v in payload.items():
                        val = self._sanitize_value(v, k)
                        if k in numeric_fields:
                            if val in (None, ""):
                                continue
                            if not isinstance(val, (int, float)):
                                val = self._clean_numeric(val)
                            if val in (None, "") or not isinstance(val, (int, float)):
                                logger.warning("Dropping field %s due to non-numeric value. Payload: %s", k, payload)
                                continue
                            val = float(val)
                        sanitized[k] = val

                    cols = ", ".join(sanitized.keys())
                    placeholders = ", ".join(["%s"] * len(sanitized))
                    update_cols = ", ".join(f"{c}=EXCLUDED.{c}" for c in sanitized.keys() if c not in {fk_col, line_no_col})
                    if doc_type == "Invoice" and "invoice_line_id" in columns:
                        conflict_cols = ["invoice_line_id"]
                    elif doc_type == "Purchase_Order" and "po_line_id" in columns:
                        conflict_cols = ["po_line_id"]
                    else:
                        conflict_cols = [fk_col, line_no_col]
                    if not self._has_unique_constraint(cur, schema, table, conflict_cols):
                        conflict_cols = []
                    sql = f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders})"
                    if conflict_cols:
                        target_cols = ", ".join(conflict_cols)
                        if update_cols:
                            sql += f" ON CONFLICT ({target_cols}) DO UPDATE SET {update_cols}"
                        else:
                            sql += f" ON CONFLICT ({target_cols}) DO NOTHING"
                    else:
                        sql += " ON CONFLICT DO NOTHING"
                    cur.execute(sql, list(sanitized.values()))
            if close_conn:
                conn.commit()
        except Exception as exc:
            logger.error("Failed to persist line items for %s: %s", doc_type, exc)
            if close_conn:
                conn.rollback()
        finally:
            if close_conn:
                conn.close()

    # ============================ CLASSIFICATION ==========================
    def _classify_doc_type(self, text: str) -> str:
        snippet = text[:2000].lower()
        scores = {dtype: sum(snippet.count(kw) for kw in kws) for dtype, kws in DOC_TYPE_KEYWORDS.items()}
        best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 0:
            return best_type
        prompt = ("Classify the following document as Invoice, Purchase_Order, Quote, Contract, or Other. "
                  "Respond with only the label.\n\nContext:\n" + DOC_CONTEXT_TEXT + "\n\nDocument:\n" + snippet)
        try:
            resp = self.call_ollama(prompt=prompt, model=self.extraction_model)
            label = resp.get("response", "").strip().lower()
            for canonical in DOC_TYPE_KEYWORDS:
                if canonical.replace("_", " ").lower() in label:
                    return canonical
            if "invoice" in label:
                return "Invoice"
            if "purchase" in label or "po" in label:
                return "Purchase_Order"
            if "quote" in label:
                return "Quote"
            if "contract" in label:
                return "Contract"
        except Exception:
            pass
        return "Other"

    def _classify_product_type(self, text: str) -> str:
        snippet = text.lower()
        for category, keywords in PRODUCT_KEYWORDS.items():
            for kw in keywords:
                if kw in snippet:
                    return category
        return "other"

    def _extract_unique_id(self, text: str, doc_type: str) -> str:
        pattern = UNIQUE_ID_PATTERNS.get(doc_type)
        if not pattern:
            return ""
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return ""
        candidate = re.sub(r"[^A-Za-z0-9-]", "", match.group(1))
        return candidate[:32]

    def _infer_vendor_name(self, text: str, object_key: str | None = None) -> str:
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

    # ============================ TRAIN (optional) =======================
    def train_extraction_model(
        self, training_data: List[Tuple[str, str]], epochs: int = 1
    ) -> Optional[str]:
        """Fine-tune the embedding model using provided training data.

        Parameters
        ----------
        training_data: List[Tuple[str, str]]
            Pairs of text segments that should be close in embedding space.
        epochs: int
            Number of training epochs.

        Returns
        -------
        Optional[str]
            Path to the directory containing the fine-tuned model.
        """
        if not training_data:
            logger.info("No training data provided; skipping fine-tuning.")
            return None
        try:
            from sentence_transformers import InputExample, losses
            from torch.utils.data import DataLoader
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("sentence-transformers training dependencies missing: %s", exc)
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.agent_nick.embedding_model
        examples = [InputExample(texts=[t, l]) for t, l in training_data]
        train_loader = DataLoader(examples, batch_size=8, shuffle=True)
        train_loss = losses.CosineSimilarityLoss(model)
        output_dir = os.path.join("/tmp", f"fine_tuned_{uuid.uuid4().hex[:8]}")
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=epochs,
            output_path=output_dir,
            device=device,
        )
        logger.info("Fine-tuned extraction model saved to %s", output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # End of class
    # ------------------------------------------------------------------

#
# if __name__ == "__main__":
#     # Simple manual invocation used during local development. The production
#     # system always provides an ``AgentContext`` via the orchestrator.
#     from agents.base_agent import AgentNick, AgentContext
#
#     agent_nick = AgentNick()
#     logging.basicConfig(level=logging.INFO)
#     agent = DataExtractionAgent(agent_nick)  # Replace with actual AgentNick instance
#
#     context = AgentContext(
#         workflow_id="manual-test",
#         agent_id="data_extraction",
#         user_id=agent_nick.settings.script_user,
#         input_data={"s3_prefix": "Invoices/"},
#     )

#     result = agent.run(context)
#     print(result)
