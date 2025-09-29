from __future__ import annotations
import logging
import re
import uuid
import os
import tempfile
import textwrap
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
try:
    import camelot  # type: ignore
except Exception:
    camelot = None
_easyocr_reader = None

from qdrant_client import models
from sentence_transformers import util
import torch
from datetime import datetime, timedelta
from dateutil import parser

from utils.nlp import extract_entities
from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from agents.discrepancy_detection_agent import DiscrepancyDetectionAgent
from utils.gpu import configure_gpu
from utils.procurement_schema import (
    DOC_TYPE_TO_TABLE,
    PROCUREMENT_SCHEMAS,
    TableSchema,
    extract_structured_content,
)


logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
configure_gpu()

HITL_CONFIDENCE_THRESHOLD = 0.85

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs" / "procurement_table_reference.md"
_TABLE_SECTION_PATTERN = re.compile(r"### `([^`]+)`'?[\r\n]+```(.*?)```", re.DOTALL)


@lru_cache(maxsize=1)
def _table_reference_sections() -> Dict[str, str]:
    if not REFERENCE_PATH.exists():
        return {}
    try:
        text = REFERENCE_PATH.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Unable to read procurement table reference at %s", REFERENCE_PATH)
        return {}

    sections: Dict[str, str] = {}
    for match in _TABLE_SECTION_PATTERN.finditer(text):
        table_name = match.group(1).strip().strip("'")
        block = textwrap.dedent(match.group(2)).strip()
        if table_name and block:
            sections[table_name] = block
    return sections


@dataclass
class PageExtractionResult:
    page_number: int
    route: str
    digital_text: str = ""
    ocr_text: str = ""
    char_count: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        return (self.digital_text or "").strip() or (self.ocr_text or "").strip()


@dataclass
class DocumentTextBundle:
    full_text: str
    page_results: List[PageExtractionResult] = field(default_factory=list)
    raw_text: str = ""
    ocr_text: str = ""
    routing_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StructuredExtractionResult:
    header: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    header_df: pd.DataFrame
    line_df: pd.DataFrame
    report: Dict[str, Any]

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

DOC_TYPE_EXTRA_KEYWORDS = {
    "Invoice": ["supplier", "vendor", "remit to", "from", "sold by", "ship from"],
    "Purchase_Order": ["supplier", "vendor", "sold by", "ship from", "ship to"],
    "Quote": ["supplier", "vendor", "prepared by"],
    "Contract": [
        "supplier",
        "vendor",
        "party",
        "parties",
        "effective date",
        "term",
        "duration",
        "payment terms",
        "renewal",
        "governing law",
        "jurisdiction",
        "signature",
    ],
}

SUPPLIER_EXTRACTION_GUIDANCE = (
    "Always identify the supplier/vendor/contracting party names by reading the header, footer, and signature blocks. "
    "If a clear supplier or company name appears anywhere in those sections, populate supplier_name and supplier_id with "
    "the most complete legal name rather than returning null."
)

DOC_TYPE_EXTRA_INSTRUCTIONS = {
    "Invoice": (
        "Invoices typically show the supplier in header or footer panels such as 'From', 'Vendor', or 'Supplier'. "
        "Use those cues along with remit-to details when extracting supplier fields."
    ),
    "Purchase_Order": (
        "Purchase orders display the selling supplier or vendor in the top banner and sometimes in the closing footer. "
        "Capture that identity even if it only appears once."
    ),
    "Contract": (
        "Contracts describe the supplier party, contract title, term, total value, payment terms, renewal language, and "
        "governing law within the introductory clauses and signature pages. Read those sections carefully so the "
        "extracted fields reflect the actual agreement context."
    ),
}

UNIQUE_ID_PATTERNS = {
    "Invoice": r"invoice\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Purchase_Order": r"(?:purchase order|po)\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Quote": r"quote\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
    "Contract": r"contract\s*(?:number|no\.|#)?\s*[:#-]?\s*([A-Za-z0-9-]+)",
}

PATTERN_MAPPING = {
    "invoice_no": [
        r"\binvoice\s*(?:no\.?|number|#)\b",
        r"\bTax\s*Invoice\s*(?:No\.?|Number)\b",
        r"\bInvoice\s*#\b",
    ],
    "po_no": [
        r"\bpo\s*(?:no\.?|number|#)\b",
        r"\bpurchase\s*order\s*(?:no\.?|number)\b",
    ],
    "invoice_date": [
        r"\binvoice\s*date\b",
        r"\bbilling\s*date\b",
        r"\bdate\s*of\s*invoice\b",
    ],
    "due_date": [r"\bdue\s*date\b", r"\bpayment\s*due\b"],
    "supplier_name": [
        r"\bsupplier\s*(?:name)?\b",
        r"\bvendor\s*(?:name)?\b",
        r"\bfrom\b",
    ],
    "buyer": [r"\bbuyer\b", r"\bbill\s*to\b"],
    "currency": [r"\bcurrency\b", r"\bcur\.\b"],
    "invoice_total": [
        r"\bgrand\s*total\b",
        r"\binvoice\s*total\b",
        r"\bamount\s*due\b",
    ],
    "tax_amount": [r"\btax\s*amount\b", r"\bvat\b", r"\bgst\b"],
    "po_total": [r"\border\s*total\b", r"\btotal\s*amount\b"],
    "quote_total": [r"\bquote\s*total\b", r"\bnet\s*total\b"],
}

PATTERN_TARGETS = {
    "invoice_no": "invoice_id",
    "po_no": "po_id",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "supplier_name": "vendor_name",
    "buyer": "buyer_id",
    "currency": "currency",
    "invoice_total": "invoice_total_incl_tax",
    "tax_amount": "tax_amount",
    "po_total": "total_amount",
    "quote_total": "total_amount",
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
    "quantity": [r"\bQty\b", r"\bQuantity\b", r"\bHours\b", r"\bDays\b"],
    "unit_of_measure": [
        r"\bUOM\b",
        r"\bUnit\b",
        r"\bUnit\s*of\s*Measure\b",
        r"\bMeasure\b",
    ],
    "unit_price": [
        r"\bUnit\s*Price\b",
        r"\bUnit\s*Cost\b",
        r"\bUnit\s*Rate\b",
        r"\bRate\b",
        r"\bPrice\b",
        r"\bPrice\s*Each\b",
        r"\bCost\s*Each\b",
        r"\bPrice\s*Per\s*Unit\b",
        r"\bNet\s*Price\b",
        r"\bUnit\s*Net\b",
    ],
    "line_amount": [
        r"\bLine\s*Total\b",
        r"\bLine\s*Amount\b",
        r"\bAmount\b",
        r"\bNet\s*Amount\b",
        r"\bSubtotal\b",
    ],
    "tax_percent": [r"\bTax\s*%\b", r"\bVAT\s*%\b", r"\bGST\s*%\b"],
    "tax_amount": [
        r"\bTax\s*Amt\b",
        r"\bTax\b",
        r"\bVAT\b",
        r"\bVAT\s*Amount\b",
        r"\bGST\s*Amount\b",
    ],
    "total_amount_incl_tax": [
        r"\bTotal\b",
        r"\bTotal\s*Incl\s*Tax\b",
        r"\bGross\b",
        r"\bTotal\s*Payable\b",
        r"\bTotal\s*Incl\s*VAT\b",
        r"\bTotal\s*Due\b",
    ],
    "currency": [r"\bCurrency\b"],
    "line_total": [r"\bLine\s*Total\b", r"\bAmount\b", r"\bTotal\b"],
    "total_amount": [r"\bTotal\b", r"\bTotal\s*Amount\b", r"\bOrder\s*Total\b"],
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
        self.extraction_model = getattr(
            self.settings,
            "document_extraction_model",
            self.settings.extraction_model,
        )

    # --------------------------------------------------------------------
    # Regex-based header extraction
    # --------------------------------------------------------------------
    def _extract_header_regex(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Best-effort extraction of key header fields using simple regular expressions.
        This is designed as a complementary heuristic to the schema-aware and
        LLM-based extraction methods.  It searches the raw text for common
        procurement identifiers and dates and returns any matches.  If a value
        cannot be parsed into a standard format (e.g. a date), the raw string
        is returned as-is.

        Parameters
        ----------
        text : str
            The complete document text.
        doc_type : str
            The canonical document type: "Invoice", "Purchase_Order", "Quote", or "Contract".

        Returns
        -------
        Dict[str, Any]
            A dictionary of extracted header fields keyed by canonical column names.
        """
        header: Dict[str, Any] = {}
        # Normalize whitespace for easier pattern matching
        # We keep a copy of the original text for case-sensitive matches
        clean_text = " ".join(text.split())
        # Helper to parse dates if possible
        def _parse_date(raw: str) -> str:
            raw = raw.strip()
            try:
                # parser.parse is tolerant of various date formats
                dt = parser.parse(raw, dayfirst=False, fuzzy=True)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return raw

        for pattern_key, regexes in PATTERN_MAPPING.items():
            target_field = PATTERN_TARGETS.get(pattern_key)
            if not target_field or header.get(target_field):
                continue
            for pattern in regexes:
                m = re.search(pattern, text, re.I)
                if not m:
                    continue
                start = m.end()
                snippet = text[start : start + 160]
                snippet_line = snippet.splitlines()[0] if snippet else ""
                candidate = snippet_line.lstrip(" :#-").strip()
                candidate = re.split(r"\s{2,}", candidate)[0].strip()
                candidate = re.split(r"\b(?:date|number|no|#)\b", candidate, 1)[0].strip()
                if not candidate:
                    continue
                header.setdefault(target_field, candidate)
                logger.info(
                    "header_source=pattern field=%s pattern=%s value=%s",
                    target_field,
                    pattern,
                    candidate,
                )
                break

        if doc_type == "Invoice":
            # Invoice ID: look for strings like INV123456 or "Invoice No: 12345"
            m = re.search(r"\bINV[-\w]*\d+\b", clean_text, re.I)
            if m:
                header.setdefault("invoice_id", m.group(0).strip())
            # PO reference on invoice: e.g. "PO12345" or "PO 12345"
            m = re.search(r"\bPO\s*[-]?\s*\d+\b", clean_text, re.I)
            if m:
                header.setdefault("po_id", m.group(0).replace(" ", "").upper())
            # Invoice date
            m = re.search(r"(?:invoice\s+date|date\s+of\s+invoice)[:\-\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("invoice_date", _parse_date(m.group(1)))
            # Due date
            m = re.search(r"(?:due\s+date|payment\s+due)[:\-\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("due_date", _parse_date(m.group(1)))
            # Currency: look for currency symbols and map to ISO codes
            if "$" in text:
                header.setdefault("currency", "USD")
            if "£" in text or "GBP" in text:
                header.setdefault("currency", "GBP")
            if "€" in text or "EUR" in text:
                header.setdefault("currency", "EUR")
            # Capture totals by matching patterns like "Total 1,404.77" or "Grand Total: 1404.77"
            m = re.search(r"(?:grand\s+total|invoice\s+total|total\s+including\s+tax|amount\s+due)[:\s]*([\d,\.]+)", text, re.I)
            if m:
                header.setdefault("invoice_total_incl_tax", m.group(1).replace(",", "").strip())
            # Subtotal / invoice amount (pre-tax)
            m = re.search(r"(?:subtotal|invoice\s+amount)[:\s]*([\d,\.]+)", text, re.I)
            if m:
                header.setdefault("invoice_amount", m.group(1).replace(",", "").strip())
            # Tax amount
            m = re.search(r"(?:tax\s+amount|tax)[:\s]*([\d,\.]+)", text, re.I)
            if m:
                header.setdefault("tax_amount", m.group(1).replace(",", "").strip())
        elif doc_type == "Purchase_Order":
            # PO ID
            m = re.search(r"\bPO\s*[-]?\s*\d+\b", clean_text, re.I)
            if m:
                header.setdefault("po_id", m.group(0).replace(" ", "").upper())
            # Order date
            m = re.search(r"(?:order\s+date|po\s+date|date)[:\-\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("order_date", _parse_date(m.group(1)))
            # Expected delivery date
            m = re.search(r"(?:expected\s+delivery\s+date|delivery\s+date|ship\s+date)[:\-\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("expected_delivery_date", _parse_date(m.group(1)))
            # Quote reference on PO: QUT123456
            m = re.search(r"\bQUT\s*[-]?\s*\d+\b", clean_text, re.I)
            if m:
                header.setdefault("contract_id", m.group(0).replace(" ", "").upper())
            # Total amount
            m = re.search(r"(?:total\s+amount|grand\s+total|order\s+total)[:\s]*([\d,\.]+)", text, re.I)
            if m:
                header.setdefault("total_amount", m.group(1).replace(",", "").strip())
            # Currency
            if "£" in text or "GBP" in text:
                header.setdefault("currency", "GBP")
            if "€" in text or "EUR" in text:
                header.setdefault("currency", "EUR")
            if "$" in text and header.get("currency", "") != "USD":
                header.setdefault("currency", "USD")
        elif doc_type == "Quote":
            # Quote ID
            m = re.search(r"\bQUT\s*[-]?\s*\d+\b", clean_text, re.I)
            if m:
                header.setdefault("quote_id", m.group(0).replace(" ", "").upper())
            # Quote date
            m = re.search(r"(?:quote\s+date|date)[:\-\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("quote_date", _parse_date(m.group(1)))
            # Validity / expiry date
            m = re.search(r"(?:valid\s+until|validity\s+date|expiry\s+date|expiring\s+on)[:\s]*([A-Za-z0-9,./\\-]+)", text, re.I)
            if m:
                header.setdefault("validity_date", _parse_date(m.group(1)))
            # Currency
            if "£" in text or "GBP" in text:
                header.setdefault("currency", "GBP")
            if "€" in text or "EUR" in text:
                header.setdefault("currency", "EUR")
            if "$" in text and header.get("currency", "") != "USD":
                header.setdefault("currency", "USD")
            # Total amount in quote
            m = re.search(r"(?:total\s+amount|grand\s+total|amount\s+due)[:\s]*([\d,\.]+)", text, re.I)
            if m:
                header.setdefault("total_amount", m.group(1).replace(",", "").strip())
        return header

    # --------------------------------------------------------------------
    # Regex-based line item extraction
    # --------------------------------------------------------------------
    def _extract_line_items_regex(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Extract line items from raw text using heuristic patterns.
        This fallback method searches for lines that contain a description followed by
        numeric values (quantity, price, totals).  It stops scanning when it
        encounters common subtotal/total keywords defined in TOTALS_STOP_WORDS.

        Parameters
        ----------
        text : str
            The complete document text.
        doc_type : str
            The canonical document type: "Invoice", "Purchase_Order", or "Quote".

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries representing line items.
        """
        items: List[Dict[str, Any]] = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Patterns for 4-col lines: description, quantity, unit_price, total
        pat4 = re.compile(r"^(.+?)\s+(\d+)\s+([\d,\.]+)\s+([\d,\.]+)\s*$")
        # Patterns for 3-col lines: description, quantity, total (no explicit unit price)
        pat3 = re.compile(r"^(.+?)\s+(\d+)\s+([\d,\.]+)\s*$")
        for raw in lines:
            # Stop if this line is a subtotal or total row
            if TOTALS_STOP_WORDS.search(raw):
                break
            m4 = pat4.match(raw)
            m3 = pat3.match(raw) if not m4 else None
            if m4:
                desc, qty, unit, total = m4.groups()
                item: Dict[str, Any] = {}
                item["item_description"] = desc.strip()
                try:
                    item["quantity"] = int(qty)
                except Exception:
                    item["quantity"] = qty.strip()
                # Normalize numeric strings: remove commas and currency symbols
                def _norm_num(val: str) -> str:
                    return re.sub(r"[^0-9.]", "", val)
                unit_price = _norm_num(unit)
                total_amount = _norm_num(total)
                if doc_type == "Invoice":
                    item["unit_price"] = unit_price
                    item["line_amount"] = total_amount
                    item["total_amount_incl_tax"] = total_amount
                elif doc_type == "Purchase_Order":
                    # In POs, treat unit price as both unit_price and line_total
                    item["unit_price"] = unit_price
                    item["line_total"] = total_amount
                    item["total_amount"] = total_amount
                elif doc_type == "Quote":
                    item["unit_price"] = unit_price
                    item["total_amount"] = total_amount
                items.append(item)
            elif m3:
                desc, qty, amount = m3.groups()
                item = {}
                item["item_description"] = desc.strip()
                try:
                    item["quantity"] = int(qty)
                except Exception:
                    item["quantity"] = qty.strip()
                def _norm_num(val: str) -> str:
                    return re.sub(r"[^0-9.]", "", val)
                amount_clean = _norm_num(amount)
                if doc_type == "Invoice":
                    # Without unit price, treat value as line_amount and total
                    item["line_amount"] = amount_clean
                    item["total_amount_incl_tax"] = amount_clean
                elif doc_type == "Purchase_Order":
                    item["line_total"] = amount_clean
                    item["total_amount"] = amount_clean
                    item["unit_price"] = amount_clean
                elif doc_type == "Quote":
                    item["total_amount"] = amount_clean
                items.append(item)
        return items

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
                with self._borrow_s3_client() as s3_client:
                    resp = s3_client.list_objects_v2(
                        Bucket=self.settings.s3_bucket_name, Prefix=prefix
                    )
                keys.extend(obj["Key"] for obj in resp.get("Contents", []))

        supported_exts = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}
        keys = [k for k in keys if os.path.splitext(k)[1].lower() in supported_exts]

        max_workers_setting = int(getattr(self.settings, "data_extraction_max_workers", os.cpu_count() or 4))
        pool_cap = getattr(self.agent_nick, "s3_pool_size", max_workers_setting)
        max_workers = max(1, min(len(keys) or 1, max_workers_setting, pool_cap))
        logger.info(
            "Processing %d documents using %d worker threads (configured max=%d, s3_pool=%d)",
            len(keys),
            max_workers,
            max_workers_setting,
            pool_cap,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        except Exception as exc:
            logger.error("Failed downloading %s: %s", object_key, exc)
            return None

        force_ocr_vendors = set(
            vendor.lower() for vendor in getattr(self.settings, "force_ocr_vendors", [])
        )
        force_ocr = any(
            vendor and vendor in object_key.lower() for vendor in force_ocr_vendors
        )
        text_bundle = self._extract_text(file_bytes, object_key, force_ocr=force_ocr)
        text = text_bundle.full_text
        doc_type = self._classify_doc_type(text)
        product_type = self._classify_product_type(text)
        unique_id = self._extract_unique_id(text, doc_type)
        vendor_name = self._infer_vendor_name(text, object_key)
        if not unique_id:
            unique_id = uuid.uuid4().hex[:8]

        # Vectorize raw document content for search regardless of type
        self._vectorize_document(text, unique_id, doc_type, product_type, object_key)

        header: Dict[str, Any] = {"doc_type": doc_type, "product_type": product_type}
        if vendor_name:
            header["vendor_name"] = vendor_name
        line_items: List[Dict[str, Any]] = []
        table_report: Dict[str, Any] = {}
        pk_value = unique_id
        data: Optional[Dict[str, Any]] = None

        if doc_type in {"Purchase_Order", "Invoice", "Quote", "Contract"}:
            structured = self._extract_structured_data(
                text, file_bytes, doc_type
            )
            header = structured.header
            line_items = structured.line_items
            table_report = structured.report
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
                "header_df": structured.header_df.replace({pd.NA: None}).to_dict("records"),
                "lines_df": structured.line_df.replace({pd.NA: None}).to_dict("records"),
                "validation": {
                    "is_valid": bool(ok),
                    "confidence_score": float(conf),
                    "notes": "; ".join(notes) if notes else "ok",
                },
                "report": table_report,
            }

            self._persist_to_postgres(header, line_items, doc_type, pk_value)
            self._vectorize_structured_data(header, line_items, doc_type, pk_value, product_type)

        result = {"object_key": object_key, "id": pk_value or object_key, "doc_type": doc_type or "", "status": "success"}
        if data:
            data["raw_text"] = text_bundle.raw_text
            data["ocr_text"] = text_bundle.ocr_text
            data["page_routes"] = text_bundle.routing_log
            result["data"] = data
        return result

    # ============================ EXTRACTION HELPERS ======================
    def _extract_pdf_text_bundle(
        self, file_bytes: bytes, force_ocr: bool = False
    ) -> DocumentTextBundle:
        file_bytes = _maybe_decompress(file_bytes)
        if not file_bytes.startswith(b"%PDF"):
            logger.warning(
                "Provided bytes do not look like a PDF; attempting image extraction"
            )
            ocr_text = self._extract_text_from_image(
                file_bytes, allow_pdf_fallback=False
            )
            page_result = PageExtractionResult(
                page_number=1, route="ocr", ocr_text=ocr_text, char_count=len(ocr_text)
            )
            return DocumentTextBundle(
                full_text=ocr_text,
                page_results=[page_result],
                raw_text="",
                ocr_text=ocr_text,
                routing_log=[{"page": 1, "route": "ocr", "chars": len(ocr_text)}],
            )

        page_results: List[PageExtractionResult] = []
        routing_log: List[Dict[str, Any]] = []
        pdf = None
        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf") if fitz else None
        except Exception as exc:
            logger.debug("PyMuPDF open failed: %s", exc)
            pdf = None

        try:
            with pdfplumber.open(BytesIO(file_bytes)) as plumber_doc:
                for idx, page in enumerate(plumber_doc.pages, start=1):
                    warnings_local: List[str] = []
                    digital_text = page.extract_text() or ""
                    char_count = len(digital_text.strip())
                    route = "digital"
                    ocr_text = ""
                    if not digital_text.strip():
                        digital_text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                        char_count = len(digital_text.strip())

                    if force_ocr or char_count < 24:
                        route = "ocr"
                    if route == "digital" and not digital_text.strip():
                        route = "ocr"

                    if route == "ocr":
                        image = self._render_pdf_page(pdf, page, idx)
                        if image is None:
                            warnings_local.append("render_failed")
                            logger.warning(
                                "Page %s could not be rendered for OCR; falling back to digital text",
                                idx,
                            )
                            route = "digital"
                        else:
                            processed = self._prepare_ocr_image(image)
                            ocr_text = self._perform_ocr(processed)
                            if not ocr_text.strip() and digital_text.strip():
                                warnings_local.append("ocr_empty")
                                route = "digital"
                            elif not ocr_text.strip():
                                warnings_local.append("ocr_empty")
                    routing_log.append(
                        {
                            "page": idx,
                            "route": route,
                            "digital_chars": char_count,
                            "ocr_chars": len(ocr_text.strip()),
                        }
                    )
                    logger.info(
                        "page_route decision=%s page=%d digital_chars=%d ocr_chars=%d",
                        route,
                        idx,
                        char_count,
                        len(ocr_text.strip()),
                    )
                    page_results.append(
                        PageExtractionResult(
                            page_number=idx,
                            route=route,
                            digital_text=digital_text,
                            ocr_text=ocr_text,
                            char_count=char_count,
                            warnings=warnings_local,
                        )
                    )
        except Exception as exc:
            logger.warning("pdfplumber failed during page routing: %s", exc)

        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass

        raw_text = "\n".join(
            result.digital_text.strip() for result in page_results if result.digital_text
        )
        ocr_text = "\n".join(
            result.ocr_text.strip() for result in page_results if result.ocr_text
        )
        full_text_parts: List[str] = []
        for result in page_results:
            primary = result.digital_text.strip() if result.route == "digital" else ""
            if not primary:
                primary = result.combined_text
            if primary:
                full_text_parts.append(primary)
        full_text = "\n".join(part for part in full_text_parts if part)
        if not full_text and raw_text:
            full_text = raw_text
        if not full_text and ocr_text:
            full_text = ocr_text
        return DocumentTextBundle(
            full_text=full_text,
            page_results=page_results,
            raw_text=raw_text,
            ocr_text=ocr_text,
            routing_log=routing_log,
        )

    def _render_pdf_page(
        self,
        fitz_doc,
        plumber_page,
        page_number: int,
        dpi: int = 360,
    ) -> Optional[Image.Image]:
        if Image is None:
            return None
        try:
            if fitz_doc is not None:
                page = fitz_doc.load_page(page_number - 1)
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                return img
            if hasattr(plumber_page, "to_image"):
                rendered = plumber_page.to_image(resolution=dpi).original
                if rendered.mode != "RGB":
                    rendered = rendered.convert("RGB")
                return rendered
        except Exception as exc:
            logger.debug("Failed to render page %d: %s", page_number, exc)
        return None

    def _prepare_ocr_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        upscale_factor = max(1, int(360 / 72))
        new_size = (width * upscale_factor, height * upscale_factor)
        if new_size != image.size:
            image = image.resize(new_size, Image.BICUBIC)
        if pytesseract is not None:
            try:
                osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
                rotate = osd.get("rotate", 0)
                if rotate:
                    image = image.rotate(-rotate, expand=True)
            except Exception as exc:
                logger.debug("Deskew failed: %s", exc)
        return image

    def _perform_ocr(self, image: Image.Image) -> str:
        if pytesseract is not None:
            try:
                return pytesseract.image_to_string(image)
            except Exception as exc:
                logger.error("pytesseract OCR failed: %s", exc)
        if easyocr is not None:
            try:
                reader = _get_easyocr_reader()
                if reader is not None:
                    lines = reader.readtext(np.array(image), detail=0, paragraph=True)
                    return "\n".join(lines)
            except Exception as exc:
                logger.warning("EasyOCR failed: %s", exc)
        return ""

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
                bundle = self._extract_pdf_text_bundle(file_bytes)
                return bundle.full_text
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

    def _extract_text(
        self, file_bytes: bytes, object_key: str, force_ocr: bool = False
    ) -> DocumentTextBundle:
        ext = os.path.splitext(object_key)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf_text_bundle(file_bytes, force_ocr=force_ocr)
        if ext in {".doc", ".docx"}:
            text = self._extract_text_from_docx(file_bytes)
            page_result = PageExtractionResult(
                page_number=1,
                route="digital",
                digital_text=text,
                char_count=len(text or ""),
            )
            return DocumentTextBundle(
                full_text=text,
                page_results=[page_result],
                raw_text=text,
                ocr_text="",
                routing_log=[{"page": 1, "route": "digital", "digital_chars": len(text)}],
            )
        if ext in {".png", ".jpg", ".jpeg"}:
            text = self._extract_text_from_image(file_bytes)
            page_result = PageExtractionResult(
                page_number=1,
                route="ocr",
                ocr_text=text,
                char_count=len(text or ""),
            )
            return DocumentTextBundle(
                full_text=text,
                page_results=[page_result],
                raw_text="",
                ocr_text=text,
                routing_log=[{"page": 1, "route": "ocr", "ocr_chars": len(text)}],
            )
        logger.warning("Unsupported document type '%s' for %s", ext, object_key)
        return DocumentTextBundle(full_text="", page_results=[], raw_text="", ocr_text="")

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
            "llm_structured": 0.75,
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
                        msg = "Line totals + tax do not match invoice_total_incl_tax (±0.02)."
                        notes.append(msg)
                        logger.warning("validation_diff doc_type=%s detail=%s", doc_type, msg)
                else:
                    notes.append("invoice_total_incl_tax missing.")
            except Exception:
                notes.append("Failed total reconciliation.")
        elif doc_type == "Purchase_Order":
            try:
                line_total = sum(
                    float(
                        self._clean_numeric(
                            li.get("line_total")
                            or li.get("total_amount")
                            or li.get("line_amount")
                            or 0
                        )
                        or 0
                    )
                    for li in line_items
                )
                header_total = self._clean_numeric(
                    header.get("total_amount") or header.get("po_total_value")
                )
                if header_total is not None:
                    if abs(line_total - header_total) <= 0.02:
                        conf += 0.06
                    else:
                        msg = "Line totals do not match purchase order total_amount (±0.02)."
                        notes.append(msg)
                        logger.warning("validation_diff doc_type=%s detail=%s", doc_type, msg)
            except Exception:
                notes.append("Failed purchase order total reconciliation.")
        elif doc_type == "Quote":
            try:
                line_total = sum(
                    float(
                        self._clean_numeric(
                            li.get("total_amount")
                            or li.get("line_total")
                            or li.get("line_amount")
                            or 0
                        )
                        or 0
                    )
                    for li in line_items
                )
                header_total = self._clean_numeric(header.get("total_amount"))
                if header_total is not None:
                    if abs(line_total - header_total) <= 0.02:
                        conf += 0.05
                    else:
                        msg = "Quote line totals do not match total_amount (±0.02)."
                        notes.append(msg)
                        logger.warning("validation_diff doc_type=%s detail=%s", doc_type, msg)
            except Exception:
                notes.append("Failed quote total reconciliation.")
        try:
            due = self._clean_date(header.get("due_date"))
            paid = self._clean_date(header.get("invoice_paid_date"))
            if due and not paid and due < datetime.utcnow().date():
                header["invoice_status"] = header.get("invoice_status") or "Overdue"
                conf += 0.02
        except Exception:
            pass
        overall_ok = len([n for n in notes if "do not match" in n or "Failed" in n]) == 0
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
    def _extract_line_items_from_pdf_tables(
        self, file_bytes: bytes, doc_type: str
    ) -> Tuple[List[Dict], Optional[str], List[str]]:
        warnings: List[str] = []
        if camelot is not None:
            for flavor in ("lattice", "stream"):
                try:
                    tables = self._run_camelot_tables(file_bytes, flavor)
                    if not tables:
                        continue
                    items = self._camelot_tables_to_items(tables, doc_type)
                    if items:
                        logger.info(
                            "table_extractor=%s rows=%d", f"camelot-{flavor}", len(items)
                        )
                        return items, f"camelot-{flavor}", warnings
                except Exception as exc:
                    warnings.append(f"camelot-{flavor}-error")
                    logger.debug("Camelot %s extraction failed: %s", flavor, exc)
        try:
            pdf_items = self._pdfplumber_tables_to_items(file_bytes, doc_type)
            if pdf_items:
                logger.info(
                    "table_extractor=pdfplumber rows=%d", len(pdf_items)
                )
                return pdf_items, "pdfplumber", warnings
        except Exception as exc:
            warnings.append("pdfplumber-table-error")
            logger.debug("pdfplumber table parsing failed: %s", exc)

        ocr_items, ocr_warnings = self._extract_line_items_ocr_grid(file_bytes, doc_type)
        warnings.extend(ocr_warnings)
        if ocr_items:
            logger.info("table_extractor=ocr-grid rows=%d", len(ocr_items))
            return ocr_items, "ocr-grid", warnings
        return [], None, warnings

    def _run_camelot_tables(self, file_bytes: bytes, flavor: str):
        if camelot is None:
            return []
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return camelot.read_pdf(tmp.name, flavor=flavor, pages="all")

    def _select_table_header(self, df: pd.DataFrame) -> Tuple[int, Dict[int, str]]:
        """Pick the most likely header row within the first few rows of a table."""
        if df is None or df.empty:
            return -1, {}
        best_idx = -1
        best_map: Dict[int, str] = {}
        best_score = 0
        for idx in range(min(5, len(df))):
            row = df.iloc[idx].tolist()
            header_map = self._map_table_headers(row)
            score = len(header_map)
            if score > best_score:
                best_idx = idx
                best_map = header_map
                best_score = score
            if score >= 3:
                break
        return best_idx, best_map

    def _camelot_tables_to_items(self, tables, doc_type: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for table in tables or []:
            df = getattr(table, "df", None)
            if df is None or df.empty:
                continue
            header_idx, header_map = self._select_table_header(df)
            if len(header_map) < 2:
                continue
            pending_desc: Optional[str] = None
            for row_idx, row in df.iloc[header_idx + 1 :].iterrows():
                row_text = " ".join(str(cell).strip() for cell in row.tolist())
                if TOTALS_STOP_WORDS.search(row_text):
                    break
                record: Dict[str, Any] = {}
                for idx, label in header_map.items():
                    if idx >= len(row):
                        continue
                    value = str(row.iloc[idx]).strip()
                    if not value or value.lower() in {"nan", "none"}:
                        continue
                    record[label] = value
                if not record:
                    continue
                if (
                    "item_description" in record
                    and not any(k for k in record if k != "item_description")
                ):
                    pending_desc = (
                        f"{pending_desc} {record['item_description']}".strip()
                        if pending_desc
                        else record["item_description"]
                    )
                    continue
                if pending_desc and "item_description" in record:
                    record["item_description"] = (
                        f"{pending_desc} {record['item_description']}".strip()
                    )
                    pending_desc = None
                normalised = self._normalize_line_item_fields([record], doc_type)[0]
                normalised = self._repair_invoice_line_values(
                    normalised, row_text, doc_type
                )
                for key in [
                    "quantity",
                    "unit_price",
                    "line_amount",
                    "tax_percent",
                    "tax_amount",
                    "total_amount_incl_tax",
                    "line_total",
                    "total_amount",
                ]:
                    if key in normalised:
                        normalised[key] = self._clean_numeric(normalised[key])
                items.append(normalised)
        return items

    def _pdfplumber_tables_to_items(
        self, file_bytes: bytes, doc_type: str
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                words = page.extract_words() or []
                if not words:
                    continue
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
                below_rows = [
                    w for w in words if w["top"] > min(w["top"] for w in header_row) + 2
                ]
                by_y: Dict[int, List[Dict[str, Any]]] = {}
                for w in below_rows:
                    y = int(round(w["top"], 0))
                    by_y.setdefault(y, []).append(w)

                pending_desc: Optional[str] = None
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

                    non_desc_keys = [
                        k for k in row_obj.keys() if k != "item_description" and row_obj[k]
                    ]
                    if (
                        "item_description" in row_obj
                        and row_obj.get("item_description")
                        and not non_desc_keys
                    ):
                        if items:
                            items[-1]["item_description"] = (
                                f"{items[-1].get('item_description','')} {row_obj['item_description']}"
                            ).strip()
                        else:
                            pending_desc = row_obj["item_description"]
                        continue
                if pending_desc and "item_description" in row_obj:
                    row_obj["item_description"] = (
                        f"{pending_desc} {row_obj['item_description']}"
                    ).strip()
                    pending_desc = None

                    row_norm = self._normalize_line_item_fields([row_obj], doc_type)[0]
                    row_text = " ".join(w["text"] for w in row).strip()
                    row_norm = self._repair_invoice_line_values(
                        row_norm, row_text, doc_type
                    )
                    for k in [
                        "quantity",
                        "unit_price",
                        "line_amount",
                        "tax_percent",
                        "tax_amount",
                        "total_amount_incl_tax",
                        "line_total",
                        "total_amount",
                    ]:
                        if k in row_norm:
                            row_norm[k] = self._clean_numeric(row_norm[k])
                    items.append(row_norm)
        return items

    def _extract_line_items_ocr_grid(
        self, file_bytes: bytes, doc_type: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        items: List[Dict[str, Any]] = []
        if pytesseract is None or Image is None:
            warnings.append("ocr-unavailable")
            return items, warnings
        if fitz is None:
            warnings.append("pymupdf-missing")
            return items, warnings
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            warnings.append("ocr-open-failed")
            logger.debug("PyMuPDF open failed for OCR grid: %s", exc)
            return items, warnings

        try:
            whitespace_positions: List[float] = []
            page_lines: List[List[str]] = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                mat = fitz.Matrix(4, 4)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes(
                    "RGB" if pix.n < 4 else "RGBA",
                    [pix.width, pix.height],
                    pix.samples,
                )
                if img.mode != "RGB":
                    img = img.convert("RGB")
                text = pytesseract.image_to_string(img)
                line_candidates = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
                page_lines.append(line_candidates)
                for line in line_candidates:
                    for match in re.finditer(r"\s{2,}", line):
                        pos = match.start() / max(1, len(line))
                        whitespace_positions.append(pos)

            boundaries: List[float] = []
            if whitespace_positions:
                hist, edges = np.histogram(
                    whitespace_positions, bins=min(12, len(whitespace_positions))
                )
                peak_indices = np.argsort(hist)[-4:]
                boundaries = sorted(
                    {
                        (edges[i] + edges[i + 1]) / 2
                        for i in peak_indices
                        if i < len(edges) - 1
                    }
                )

            column_targets = [
                "item_description",
                "quantity",
                "unit_price",
                "line_amount",
                "total_amount",
            ]
            for lines in page_lines:
                for line in lines:
                    if TOTALS_STOP_WORDS.search(line):
                        continue
                    splits = re.split(r"\s{2,}", line)
                    if len(splits) == 1 and boundaries:
                        indices = [int(round(b * len(line))) for b in boundaries]
                        pieces: List[str] = []
                        start = 0
                        for boundary in indices:
                            boundary = max(boundary, start)
                            pieces.append(line[start:boundary].strip())
                            start = boundary
                        pieces.append(line[start:].strip())
                    else:
                        pieces = [segment.strip() for segment in splits if segment.strip()]
                    if not pieces:
                        continue
                    record: Dict[str, Any] = {}
                    for idx, piece in enumerate(pieces):
                        if idx >= len(column_targets):
                            target = column_targets[-1]
                        else:
                            target = column_targets[idx]
                        record.setdefault(target, piece)
                    if "item_description" not in record:
                        continue
                    items.append(record)
        except Exception as exc:
            warnings.append("ocr-grid-error")
            logger.debug("OCR grid extraction failed: %s", exc)
        finally:
            try:
                doc.close()
            except Exception:
                pass
        if items:
            items = self._normalize_line_item_fields(items, doc_type)
        return items, warnings

    def _map_table_headers(self, headers: List[Any]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for idx, raw in enumerate(headers):
            text = str(raw).strip().lower()
            if not text:
                continue
            for label, aliases in LINE_HEADERS_ALIASES.items():
                if any(re.search(pattern, text, re.I) for pattern in aliases):
                    mapping.setdefault(idx, label)
                    break
        return mapping

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
                "balance_due": "invoice_total_incl_tax",
                "amount_due": "invoice_total_incl_tax",
                "total_due": "invoice_total_incl_tax",
                "total_amount_gbp": "invoice_total_incl_tax",
                "total_amount_usd": "invoice_total_incl_tax",
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
                "order_total": "total_amount",
                "total_amount_gbp": "total_amount",
                "total_value": "total_amount",
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
                "quote_total": "total_amount",
                "net_total": "total_amount",
                "total_amount_gbp": "total_amount",
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

    def _repair_invoice_line_values(
        self, item: Dict[str, Any], row_text: str, doc_type: str
    ) -> Dict[str, Any]:
        if doc_type != "Invoice":
            return item

        value_fields = ["unit_price", "line_amount", "tax_amount", "total_amount_incl_tax"]
        if not any(item.get(field) in (None, "") for field in value_fields):
            return item

        numbers = []
        for match in re.finditer(r"[-+]?\d[\d,]*\.?\d*", row_text):
            num = self._clean_numeric(match.group())
            if num is None:
                continue
            numbers.append(num)

        if not numbers:
            return item

        qty = self._clean_numeric(item.get("quantity")) if item.get("quantity") not in (None, "") else None
        deduped: List[float] = []
        for num in numbers:
            if qty is not None and abs(num - qty) < 1e-6:
                continue
            if not any(abs(num - existing) < 1e-6 for existing in deduped):
                deduped.append(num)

        if not deduped:
            return item

        sorted_vals = sorted(deduped)
        total = sorted_vals[-1]
        if item.get("total_amount_incl_tax") in (None, ""):
            item["total_amount_incl_tax"] = total

        remaining = [n for n in sorted_vals[:-1] if abs(n - total) > 1e-6]

        def _is_close(a: float, b: float) -> bool:
            return abs(a - b) <= max(0.02, 0.01 * max(abs(a), abs(b), 1.0))

        unit_candidate: Optional[float] = None
        if qty not in (None, 0):
            for num in list(remaining):
                if _is_close(num * qty, total):
                    unit_candidate = num
                    remaining.remove(num)
                    break

        line_candidate: Optional[float] = None
        if remaining:
            line_candidate = max(remaining)
            remaining.remove(line_candidate)

        if item.get("line_amount") in (None, ""):
            item["line_amount"] = line_candidate if line_candidate is not None else total

        if item.get("tax_amount") in (None, ""):
            diff = item["total_amount_incl_tax"] - item["line_amount"]
            tax_candidate = None
            for num in list(remaining):
                if _is_close(num, diff) and diff > 0:
                    tax_candidate = num
                    remaining.remove(num)
                    break
            if tax_candidate is not None:
                item["tax_amount"] = tax_candidate

            elif diff > 0.02:
                item["tax_amount"] = round(diff, 2)

        if item.get("unit_price") in (None, ""):
            if unit_candidate is not None:
                item["unit_price"] = unit_candidate
            elif (
                item.get("line_amount") not in (None, "")
                and qty not in (None, 0)
            ):
                item["unit_price"] = round(item["line_amount"] / qty, 2)

        if (
            item.get("total_amount_incl_tax") in (None, "")
            and item.get("line_amount") not in (None, "")
            and item.get("tax_amount") not in (None, "")
        ):
            item["total_amount_incl_tax"] = round(
                item["line_amount"] + item["tax_amount"], 2
            )

        return item

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

    def _get_table_schema(self, doc_type: str, kind: str) -> TableSchema | None:
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
        table = header_table if kind == "header" else line_table
        if not table:
            return None
        return PROCUREMENT_SCHEMAS.get(table)

    def _build_dataframe_from_records(
        self,
        data: Dict[str, Any] | List[Dict[str, Any]],
        doc_type: str,
        kind: str,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        schema = self._get_table_schema(doc_type, kind)
        schema_cols = list(schema.columns) if schema else []
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = [row for row in data if isinstance(row, dict)]
        else:
            records = []

        df = pd.DataFrame(records)
        if df.empty and schema_cols:
            df = pd.DataFrame(columns=schema_cols)
        elif not df.empty:
            df.columns = [str(col).strip() for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]

        extra_cols: List[str] = []
        missing_cols: List[str] = []
        if schema_cols:
            extra_cols = [col for col in df.columns if col not in schema_cols]
            missing_cols = [col for col in schema_cols if col not in df.columns]
            if extra_cols:
                df = df.drop(columns=[col for col in extra_cols if col in df.columns])
            for col in missing_cols:
                df[col] = pd.NA
            df = df[schema_cols]
        return df, missing_cols, extra_cols

    def _dataframe_to_header(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        normalized = df.replace({pd.NA: None})
        record = normalized.iloc[0].to_dict()
        return {k: v for k, v in record.items() if v not in ({}, [])}

    def _dataframe_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        normalized = df.replace({pd.NA: None})
        return normalized.to_dict("records")

    def _is_substantial_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        if isinstance(value, (int, float)):
            return True
        return value is not None

    def _merge_record_fields(
        self, base: Dict[str, Any] | None, override: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = dict(base or {})
        if not override:
            return {k: v for k, v in result.items() if self._is_substantial_value(v)}
        for key, value in override.items():
            if not self._is_substantial_value(value):
                continue
            if not self._is_substantial_value(result.get(key)):
                result[key] = value
        return {k: v for k, v in result.items() if self._is_substantial_value(v)}

    def _merge_line_items(
        self, base: List[Dict[str, Any]] | None, override: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]]:
        if not base and override:
            return [row for row in override if any(self._is_substantial_value(v) for v in row.values())]
        if base and not override:
            return [row for row in base if any(self._is_substantial_value(v) for v in row.values())]
        base = base or []
        override = override or []
        merged: List[Dict[str, Any]] = []
        max_len = max(len(base), len(override))
        for idx in range(max_len):
            base_row = base[idx] if idx < len(base) else {}
            override_row = override[idx] if idx < len(override) else {}
            merged_row = self._merge_record_fields(base_row, override_row)
            if merged_row:
                merged.append(merged_row)
        return merged

    def _infer_currency(self, text: str, header: Dict[str, Any]) -> Optional[str]:
        candidate = header.get("currency") or header.get("_currency_hint")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().upper()
        text_upper = text.upper()
        if "GBP" in text_upper or "£" in text:
            return "GBP"
        if "EUR" in text_upper or "€" in text:
            return "EUR"
        if "USD" in text_upper or "$" in text:
            return "USD"
        return None

    def _schema_verification_notes(
        self,
        doc_type: str,
        header_missing: List[str],
        header_extra: List[str],
        line_missing: List[str],
        line_extra: List[str],
    ) -> List[str]:
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
        notes: List[str] = []
        if header_missing:
            notes.append(
                f"Missing mapped columns for {header_table or 'header'}: {', '.join(sorted(header_missing))}"
            )
        if header_extra:
            notes.append(
                f"Unmapped values ignored for {header_table or 'header'}: {', '.join(sorted(header_extra))}"
            )
        if line_table and line_missing:
            notes.append(
                f"Missing mapped columns for {line_table}: {', '.join(sorted(line_missing))}"
            )
        if line_table and line_extra:
            notes.append(
                f"Unmapped values ignored for {line_table}: {', '.join(sorted(line_extra))}"
            )
        return notes

    @staticmethod
    @lru_cache(maxsize=16)
    def _schema_keywords(doc_type: str) -> Tuple[str, ...]:
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
        keywords: set[str] = set()
        for table_name in (header_table, line_table):
            if not table_name:
                continue
            schema = PROCUREMENT_SCHEMAS.get(table_name)
            if not schema:
                continue
            for column in schema.columns:
                column = column.strip()
                if column:
                    keywords.add(column.lower())
            for alias_list in schema.synonyms.values():
                for alias in alias_list:
                    alias = alias.strip()
                    if alias:
                        keywords.add(alias.lower())
        return tuple(sorted({kw for kw in keywords if len(kw) > 2}))

    def _schema_llm_context(self, doc_type: str) -> str:
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
        sections: List[str] = []
        reference_sections = _table_reference_sections()

        for table_name, label in (
            (header_table, "header"),
            (line_table, "line items"),
        ):
            if not table_name:
                continue
            schema = PROCUREMENT_SCHEMAS.get(table_name)
            details: List[str] = []
            if schema:
                required = ", ".join(schema.required) if schema.required else "None"
                details.append(f"Required fields: {required}")
                if schema.columns:
                    displayed = ", ".join(schema.columns[:40])
                    if len(schema.columns) > 40:
                        displayed += " ..."
                    details.append("Columns: " + displayed)
            snippet = reference_sections.get(table_name)
            if snippet:
                lines = snippet.splitlines()[:24]
                formatted = "\n".join(
                    f"    {line}" for line in lines if line.strip()
                )
                if formatted:
                    details.append(
                        "Definition excerpt from procurement_table_reference.md:\n"
                        + formatted
                    )
            if details:
                sections.append(
                    f"Table `{table_name}` ({label}) guidance:\n" + "\n".join(details)
                )
        return "\n\n".join(sections)

    def _prepare_llm_document(self, text: str, doc_type: str, max_chars: int = 7000) -> str:
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not lines:
            return collapsed[:max_chars]

        snippets: List[Tuple[str, str]] = []
        seen_segments: set[str] = set()
        consumed = 0

        def _append(label: str, segment: str) -> None:
            nonlocal consumed
            cleaned_lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
            if not cleaned_lines:
                return
            cleaned = "\n".join(cleaned_lines)
            if cleaned in seen_segments:
                return
            remaining = max_chars - consumed
            if remaining <= 0:
                return
            if len(cleaned) > remaining:
                cleaned = cleaned[:remaining]
            snippets.append((label, cleaned))
            seen_segments.add(cleaned)
            consumed += len(cleaned) + len(label) + 4

        header_limit = max(1, max_chars // 3)
        header_lines = lines[: min(len(lines), 40)]
        if header_lines:
            header_text = "\n".join(header_lines)
            _append("HEADER SECTION", header_text[:header_limit])

        lowered = text.lower()
        keyword_candidates: List[str] = list(self._schema_keywords(doc_type))
        keyword_candidates.extend(DOC_TYPE_KEYWORDS.get(doc_type, []))
        keyword_candidates.extend(DOC_TYPE_EXTRA_KEYWORDS.get(doc_type, []))
        keyword_candidates.extend(["supplier", "vendor"])

        keywords: List[str] = []
        seen_kw: set[str] = set()
        for candidate in keyword_candidates:
            candidate_lower = candidate.lower()
            if len(candidate_lower) <= 2 or candidate_lower in seen_kw:
                continue
            seen_kw.add(candidate_lower)
            keywords.append(candidate_lower)

        for keyword in keywords:
            idx = lowered.find(keyword)
            if idx == -1:
                continue
            start = max(0, idx - 240)
            end = min(len(text), idx + 360)
            label_keyword = keyword.replace("_", " ")[:32].upper()
            _append(f"CONTEXT NEAR '{label_keyword}'", text[start:end])
            if consumed >= int(max_chars * 0.8):
                break

        if consumed < int(max_chars * 0.85):
            midpoint = len(text) // 2
            span = max_chars // 6
            body_start = max(0, midpoint - span)
            body_end = min(len(text), midpoint + span)
            _append("BODY SECTION", text[body_start:body_end])

        footer_limit = max(1, max_chars // 3)
        footer_lines = lines[-min(len(lines), 40) :]
        if footer_lines:
            footer_text = "\n".join(footer_lines)
            _append("FOOTER SECTION", footer_text[-footer_limit:])

        if not snippets:
            return collapsed[:max_chars]

        merged = "\n---\n".join(f"{label}:\n{segment}" for label, segment in snippets)
        if len(merged) > max_chars:
            merged = merged[:max_chars]
        return merged

    def _llm_structured_dataframes(
        self,
        text: str,
        doc_type: str,
        header_seed: Dict[str, Any],
        line_items_seed: List[Dict[str, Any]],
        llm_text: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        llm_input = llm_text or self._prepare_llm_document(text, doc_type)
        schema = self._get_table_schema(doc_type, "header")
        line_schema = self._get_table_schema(doc_type, "line_items")
        header_columns = schema.columns if schema else []
        header_required = schema.required if schema else []
        line_columns = line_schema.columns if line_schema else []
        line_required = line_schema.required if line_schema else []

        if not header_columns and not line_columns:
            return pd.DataFrame(), pd.DataFrame(), []

        instructions = {
            "document_type": doc_type,
            "header_columns": header_columns,
            "header_required": header_required,
            "line_item_columns": line_columns,
            "line_item_required": line_required,
            "existing_header": header_seed,
            "existing_line_items": line_items_seed,
        }
        schema_context = self._schema_llm_context(doc_type)
        if schema_context:
            instructions["schema_context"] = schema_context

        prompt = (
            "You are a procurement data expert. Convert the provided document text into structured tabular data. "
            "Output strict JSON with keys 'header_rows', 'line_item_rows', and 'discrepancies'. "
            "Use the column lists exactly as provided. Fill missing values with null instead of guessing. "
            "Include at most one header row. Preserve numeric precision (no currency symbols)."
        )
        if schema_context:
            prompt += (
                " Schema guidance from procurement_table_reference.md is provided in the instructions block;"
                " respect the required columns and data types."
            )

        payload: Dict[str, Any] = {}
        for _ in range(3):
            try:
                response = self.call_ollama(
                    prompt=(
                        prompt
                        + "\n\nInstructions:\n"
                        + json.dumps(instructions)
                        + "\n\nDocument:\n"
                        + llm_input
                    ),
                    model=self.extraction_model,
                    format="json",
                )
                candidate = response.get("response", "{}")
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    break
            except Exception:
                continue

        header_rows = payload.get("header_rows", [])
        if isinstance(header_rows, dict):
            header_rows = [header_rows]
        header_df = pd.DataFrame(header_rows)
        if header_columns:
            for column in header_columns:
                if column not in header_df.columns:
                    header_df[column] = pd.NA
            header_df = header_df[header_columns]

        line_rows = payload.get("line_item_rows", [])
        if isinstance(line_rows, dict):
            line_rows = [line_rows]
        line_df = pd.DataFrame(line_rows)
        if line_columns:
            for column in line_columns:
                if column not in line_df.columns:
                    line_df[column] = pd.NA
            line_df = line_df[line_columns]

        notes = payload.get("discrepancies") or payload.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        if not isinstance(notes, list):
            notes = []
        cleaned_notes = [str(n) for n in notes if str(n).strip()]

        return header_df, line_df, cleaned_notes

    def _extract_structured_data(
        self, text: str, file_bytes: bytes, doc_type: str
    ) -> StructuredExtractionResult:
        """
        Custom structured data extraction that prioritises deterministic methods and
        avoids heavy LLM calls for faster and more consistent results.

        This implementation performs the following steps:
        1. Use regex and simple heuristics to extract header fields.
        2. Parse tables using camelot/pdfplumber for line items; fall back to regex line parsing.
        3. Normalise extracted fields and reconcile totals from line items.
        4. Infer currency and vendor names, sanitise names, and enrich contract details.
        5. Cast values to appropriate SQL types and compute a simple validation score.

        Parameters
        ----------
        text : str
            Raw document text extracted from the file.
        file_bytes : bytes
            Raw file contents; used for table parsing and layout heuristics.
        doc_type : str
            Canonical document type: "Invoice", "Purchase_Order", "Quote", or "Contract".

        Returns
        -------
        Tuple[Dict[str, Any], List[Dict[str, Any]]]
            Normalised header record and a list of normalised line item records.
        """
        # 1) LLM-guided structural extraction to capture contextual values first
        header: Dict[str, Any] = {}
        line_items: List[Dict[str, Any]] = []
        table_method: Optional[str] = None
        table_warnings: List[str] = []
        field_sources: Dict[str, str] = {}
        llm_structured = self._llm_structured_extraction(text, doc_type)
        llm_header = self._normalize_header_fields(
            (llm_structured.get("header_data") if isinstance(llm_structured, dict) else {})
            or {},
            doc_type,
        )
        if llm_header:
            before_keys = set(header.keys())
            header = self._merge_record_fields(header, llm_header)
            confidence = header.setdefault("_field_confidence", {})
            llm_conf = self._confidence_from_method("llm_structured")
            for key in llm_header:
                if key.startswith("_"):
                    continue
                confidence[key] = max(confidence.get(key, 0.0), llm_conf)
            gained = set(header.keys()) - before_keys
            for key in gained:
                field_sources.setdefault(key, "llm_structured")
        llm_line_items = []
        if isinstance(llm_structured, dict):
            raw_items = llm_structured.get("line_items") or []
            if raw_items:
                try:
                    llm_line_items = self._normalize_line_item_fields(raw_items, doc_type)
                except Exception:
                    llm_line_items = []
            raw_context = llm_structured.get("field_contexts")
            if isinstance(raw_context, dict):
                context_map = {
                    key: value
                    for key, value in raw_context.items()
                    if isinstance(value, str) and value.strip()
                }
                if context_map:
                    header["_field_context"] = context_map
        if llm_line_items:
            line_items = self._merge_line_items(line_items, llm_line_items)

        # 2) Heuristic header extraction layered on top of LLM results
        try:
            header_regex = self._extract_header_regex(text, doc_type) or {}
            normalized = self._normalize_header_fields(header_regex, doc_type)
            before_keys = set(header.keys())
            header = self._merge_record_fields(header, normalized)
            gained = set(header.keys()) - before_keys
            if gained:
                field_sources.update({key: "regex" for key in gained})
                logger.info("header_source=regex fields=%s", sorted(gained))
        except Exception:
            logger.debug("Regex header extraction failed", exc_info=True)
        try:
            header_layout = self._parse_header(text, file_bytes)
            before_keys = set(header.keys())
            header = self._merge_record_fields(header, header_layout)
            gained = set(header.keys()) - before_keys
            if gained:
                field_sources.update({key: "layout" for key in gained})
                logger.info("header_source=layout fields=%s", sorted(gained))
        except Exception:
            logger.debug("Layout header extraction failed", exc_info=True)
        try:
            ner_header = self._extract_header_with_ner(text)
            if ner_header:
                before_keys = set(header.keys())
                header = self._merge_record_fields(header, ner_header)
                gained = set(header.keys()) - before_keys
                if gained:
                    field_sources.update({key: "ner" for key in gained})
                    logger.info("header_source=ner fields=%s", sorted(gained))
        except Exception:
            logger.debug("NER header extraction failed", exc_info=True)

        try:
            structured = extract_structured_content(text, doc_type)
            schema_header = structured.get("header") or {}
            schema_lines = structured.get("line_items") or []
            if schema_header:
                header = self._merge_record_fields(
                    header,
                    self._normalize_header_fields(schema_header, doc_type),
                )
            if schema_lines:
                line_items = self._merge_line_items(line_items, schema_lines)
        except Exception:
            logger.debug("Schema-guided extraction fallback failed", exc_info=True)

        if not header.get("vendor_name"):
            from_match = re.search(r"\bfrom\s+([A-Za-z0-9&.,' ]{2,80})", text, re.I)
            if from_match:
                candidate = from_match.group(1)
                candidate = re.split(r"\bfor\b", candidate, 1)[0]
                candidate = re.sub(r"[^A-Za-z0-9&.,' ]", "", candidate).strip()
                if candidate:
                    header["vendor_name"] = candidate
        if not header.get("vendor_name"):
            vendor_guess = self._infer_vendor_name(text)
            if vendor_guess:
                header["vendor_name"] = vendor_guess

        # 3) Line item extraction: table detection then regex fallback
        try:
            if file_bytes:
                extracted, method, warnings = self._extract_line_items_from_pdf_tables(
                    file_bytes, doc_type
                )
                if extracted:
                    line_items = extracted
                table_method = method or table_method
                table_warnings.extend(warnings)
        except Exception as exc:
            table_warnings.append("table_extraction_exception")
            logger.debug("Table extraction failed: %s", exc)
        if not line_items:
            try:
                regex_items = self._extract_line_items_regex(text, doc_type) or []
                if regex_items:
                    line_items = regex_items
                    table_method = table_method or "regex-lines"
            except Exception as exc:
                table_warnings.append("regex_line_extract_exception")
                logger.debug("Regex line extraction failed: %s", exc)

        # Normalise and clean numeric fields
        if line_items:
            line_items = self._normalize_line_item_fields(line_items, doc_type)
            for item in line_items:
                for num_field in [
                    "quantity",
                    "unit_price",
                    "line_amount",
                    "tax_percent",
                    "tax_amount",
                    "total_amount_incl_tax",
                    "line_total",
                    "total_amount",
                ]:
                    if num_field in item:
                        item[num_field] = self._clean_numeric(item[num_field])

        # Preserve LLM-derived metadata before schema alignment
        meta_confidence = dict(header.get("_field_confidence", {}))
        meta_context = dict(header.get("_field_context", {}))

        # 3) Reconcile header using line totals
        header = self._reconcile_header_from_lines(header, line_items, doc_type)

        # 4) Infer currency if missing
        inferred_curr = self._infer_currency(text, header)
        if inferred_curr:
            header.setdefault("currency", inferred_curr)

        # 5) Sanitise names and enrich contract fields
        header = self._sanitize_party_names(header)
        if doc_type == "Contract":
            header = self._enrich_contract_fields(text, header)

        # 6) Align with schema and cast values
        header_df, header_missing, header_extra = self._build_dataframe_from_records(
            header, doc_type, "header"
        )
        header_df_out = header_df.copy()
        header = self._dataframe_to_header(header_df)
        line_df, line_missing, line_extra = self._build_dataframe_from_records(
            line_items, doc_type, "line_items"
        )
        line_df_out = line_df.copy()
        line_items = self._dataframe_to_records(line_df)
        header, line_items = self._validate_and_cast(header, line_items, doc_type)

        if meta_confidence:
            existing_conf = header.get("_field_confidence", {}) or {}
            if not isinstance(existing_conf, dict):
                existing_conf = {}
            merged_conf = {**meta_confidence, **existing_conf}
            header["_field_confidence"] = merged_conf
        if meta_context:
            header["_field_context"] = meta_context

        # 7) Compute validation and confidence
        ok, conf_boost, notes = self._validate_business_rules(doc_type, header, line_items)
        total_missing = len(header_missing) + len(line_missing)
        base_conf = 0.9 if total_missing == 0 else max(0.5, 0.9 - 0.05 * total_missing)
        confidence = float(min(1.0, max(0.0, base_conf + conf_boost)))
        if total_missing:
            notes.append("Missing required fields")
            ok = False
        header["_validation"] = {
            "ok": ok,
            "notes": notes,
            "confidence": confidence,
        }
        schema_notes = self._schema_verification_notes(
            doc_type, header_missing, header_extra, line_missing, line_extra
        )
        if schema_notes:
            notes.extend(schema_notes)
        self._log_structured_analysis(doc_type, header, line_items)
        report = {
            "table_method": table_method or "none",
            "table_warnings": table_warnings,
            "header_missing": header_missing,
            "line_missing": line_missing,
            "validation_notes": notes,
            "quality_score": confidence,
            "field_confidence": header.get("_field_confidence", {}),
            "field_sources": field_sources,
            "line_items_detected": len(line_items),
            "fields_failed": sorted(set(header_missing + line_missing)),
        }
        if schema_notes:
            report["manual_review"] = schema_notes
        return StructuredExtractionResult(
            header=header,
            line_items=line_items,
            header_df=header_df_out,
            line_df=line_df_out,
            report=report,
        )

    def _llm_structured_extraction(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Run a single LLM pass to capture field-level context for the document."""

        schema = SCHEMA_MAP.get(doc_type, {})
        header_fields = list(schema.get("header", {}).keys())
        line_schema = schema.get("line_items", {})
        line_fields = list(line_schema.keys())

        context_hint = DOC_TYPE_CONTEXT.get(doc_type, "")
        llm_input = self._prepare_llm_document(text, doc_type)
        schema_context = self._schema_llm_context(doc_type)

        header_field_str = ", ".join(header_fields) if header_fields else "(none)"
        line_field_str = ", ".join(line_fields) if line_fields else "(none)"

        prompt_parts = [
            f"You are an information extraction engine for {doc_type}. {context_hint}",
            "Return ONLY valid JSON in the following structure:",
            "{\n  \"header_data\": { ... },\n  \"line_items\": [ ... ],\n  \"field_contexts\": { ... }\n}",
            "Rules:",
            "- header_data MUST include only the canonical fields listed below and use null when a value cannot be located.",
            "- line_items MUST be a list of objects containing only the permitted line item fields in the order supplied.",
            "- field_contexts MUST map each header field to a short snippet (<=160 chars) of the supporting text.",
            "- Dates must use YYYY-MM-DD, currency must be a 3-letter ISO code, and numeric values must be plain decimals.",
            f"Header fields: {header_field_str}",
            f"Line item fields: {line_field_str}",
        ]
        prompt_parts.append(SUPPLIER_EXTRACTION_GUIDANCE)
        extra_instruction = DOC_TYPE_EXTRA_INSTRUCTIONS.get(doc_type)
        if extra_instruction:
            prompt_parts.append(extra_instruction)
        prompt_parts.append(
            "If a supplier or contracting party name appears anywhere in the header, footer, or signature blocks, "
            "capture it instead of leaving supplier fields null."
        )
        if schema_context:
            prompt_parts.append(
                "Schema guidance sourced from procurement_table_reference.md:\n" + schema_context
            )
        prompt_parts.append("Document:\n" + llm_input)
        prompt = "\n".join(part for part in prompt_parts if part)

        try:
            response = self.call_ollama(
                prompt=prompt,
                model=self.extraction_model,
                format="json",
            )
            payload = json.loads(response.get("response", "{}"))
            if not isinstance(payload, dict):
                raise ValueError("LLM structured payload is not a JSON object")
            return payload
        except Exception:
            logger.debug("LLM structured extraction failed", exc_info=True)
            return {}

    def _structured_output_metrics(
        self,
        doc_type: str,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))

        if header_table:
            schema = PROCUREMENT_SCHEMAS.get(header_table)
            if schema and schema.required:
                missing = [col for col in schema.required if not header.get(col)]
                metrics.append(
                    {
                        "table": header_table,
                        "segment": "header",
                        "filled": len(schema.required) - len(missing),
                        "total": len(schema.required),
                        "missing": missing,
                    }
                )
        if line_table and line_items:
            schema = PROCUREMENT_SCHEMAS.get(line_table)
            if schema and schema.required:
                filled_rows = 0
                for row in line_items:
                    if all(row.get(col) for col in schema.required):
                        filled_rows += 1
                metrics.append(
                    {
                        "table": line_table,
                        "segment": "line_items",
                        "filled": filled_rows,
                        "total": len(line_items),
                        "missing": [],
                    }
                )
        return metrics

    def _log_structured_analysis(
        self,
        doc_type: str,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
    ) -> None:
        metrics = self._structured_output_metrics(doc_type, header, line_items)
        for metric in metrics:
            total = metric.get("total", 0) or 0
            filled = metric.get("filled", 0)
            if not total:
                continue
            coverage = filled / total if total else 1.0
            table = metric.get("table", "unknown")
            segment = metric.get("segment", "header")
            missing = metric.get("missing", [])
            if missing:
                logger.warning(
                    "Document extraction coverage %.0f%% for %s (%s); missing required fields: %s",
                    coverage * 100,
                    table,
                    segment,
                    ", ".join(missing),
                )
            else:
                logger.info(
                    "Document extraction coverage %.0f%% for %s (%s)",
                    coverage * 100,
                    table,
                    segment,
                )

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
            char_match = re.match(r"(?:character varying|varchar|char)\((\d+)\)", sql_type)
            if char_match:
                max_len = int(char_match.group(1))
                text_value = str(value)
                if len(text_value) > max_len:
                    logger.debug(
                        "Truncating value for SQL type %s from %d to %d characters",
                        sql_type,
                        len(text_value),
                        max_len,
                    )
                    text_value = text_value[:max_len]
                return text_value
            if sql_type in {"character varying", "varchar", "char", "text"}:
                return str(value)
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
