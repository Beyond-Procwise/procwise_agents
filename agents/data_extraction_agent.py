from __future__ import annotations
import json
import logging
import re
import uuid
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import gzip
import concurrent.futures
import ollama
import pdfplumber
import pandas as pd
import numpy as np
try:  # PyMuPDF is optional; handled gracefully if unavailable
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None
try:  # Optional dependency for DOCX extraction
    import docx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    docx = None
try:  # Optional dependencies for image OCR
    from PIL import Image, UnidentifiedImageError  # type: ignore
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None
    pytesseract = None
    UnidentifiedImageError = Exception  # type: ignore
try:  # Optional GPU-accelerated OCR
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    easyocr = None
_easyocr_reader = None
from qdrant_client import models
from sentence_transformers import util
import torch
from datetime import datetime, timedelta
from dateutil import parser


def _get_easyocr_reader():
    """Lazily initialise an EasyOCR reader if the library is installed."""
    global _easyocr_reader
    if easyocr is None:
        return None
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        except Exception:  # pragma: no cover - optional dependency
            _easyocr_reader = None
    return _easyocr_reader

from agents.base_agent import (
    BaseAgent,
    AgentContext,
    AgentOutput,
    AgentStatus,
)
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

configure_gpu()

HITL_CONFIDENCE_THRESHOLD = 0.85


def _normalize_point_id(raw_id: str) -> int | str:
    """Qdrant allows integer identifiers; strings are hashed deterministically."""
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    if raw_id.isdigit():
        return int(raw_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


def _normalize_label(value: Any) -> str:
    """Return a lowercase label for ``doc_type``/``product_type`` fields.

    Upstream LLM responses may occasionally return lists or other unexpected
    types.  This helper safely converts those to a deterministic lowercase
    string so downstream processing is robust.
    """
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value).lower()


def _maybe_decompress(content: bytes) -> bytes:
    """Decompress gzip-compressed payloads if necessary."""
    try:
        if content[:2] == b"\x1f\x8b":
            return gzip.decompress(content)
    except Exception:
        logger.warning("Failed to decompress gzip content")
    return content


def _dict_to_text(data: Dict[str, Any]) -> str:
    """Convert a dictionary into a simple ``key: value`` text block.

    Using plain text instead of JSON improves semantic retrieval accuracy
    because embedding models typically perform better on natural language
    compared to structured representations.
    """
    return "\n".join(
        f"{k}: {v}" for k, v in data.items() if v not in (None, "")
    )


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
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                next_agents=["discrepancy_detection"],
                pass_fields={"extracted_docs": data.get("details", [])},
            )
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

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4
        ) as executor:
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
            obj = self.agent_nick.s3_client.get_object(
                Bucket=self.settings.s3_bucket_name, Key=object_key
            )
            file_bytes = obj["Body"].read()
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed downloading %s: %s", object_key, exc)
            return None

        text = self._extract_text(file_bytes, object_key)

        # Determine the document type up-front so downstream logic can be
        # content-aware.  Only purchase orders and invoices are structured; all
        # other documents are merely vectorized for retrieval.
        doc_type = self._classify_doc_type(text)

        if doc_type in {"Purchase_Order", "Invoice"}:
            header, line_items = self._extract_structured_data(text, doc_type)
            header["doc_type"] = doc_type
            pk_value = (
                header.get("invoice_id")
                or header.get("po_id")
                or header.get("quote_id")
            )
            if not header.get("vendor_name"):
                vendor_guess = self._infer_vendor_name(text, object_key)
                if vendor_guess:
                    header["vendor_name"] = vendor_guess
            if not header.get("supplier_id"):
                header["supplier_id"] = header.get("vendor_name")
        else:
            header, line_items, pk_value = {"doc_type": doc_type}, [], None

        product_type = self._classify_product_type(line_items) if line_items else None

        # Always vectorize the raw document text, even when structured parsing fails
        self._vectorize_document(
            text,
            pk_value,
            doc_type,
            product_type,
            object_key,
        )

        if header and pk_value:
            data = {
                "header_data": header,
                "line_items": line_items,
                "validation": {
                    "is_valid": True,
                    "confidence_score": 1.0,
                    "notes": "llm parser",
                },
            }
            self._persist_to_postgres(header, line_items, doc_type, pk_value)
            self._vectorize_structured_data(header, line_items, doc_type, pk_value)
            doc_id = pk_value
        else:
            data = None
            doc_id = pk_value or object_key

        result = {
            "object_key": object_key,
            "id": doc_id,
            "doc_type": doc_type or "",
            "status": "success",
        }
        if data:
            result["data"] = data
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF files with OCR and table awareness.

        The routine first attempts layout aware extraction using
        :mod:`pdfplumber`.  When a page contains no embedded text (e.g. a
        scanned image) an OCR fallback via ``pytesseract`` is used.  Word level
        coordinates are leveraged to separate left/right columns so that the
        reading order is preserved for common two column layouts.  Basic table
        extraction is also performed and appended line-wise to the output.
        """

        file_bytes = _maybe_decompress(file_bytes)
        if not file_bytes.startswith(b"%PDF"):
            logger.warning(
                "Provided bytes do not look like a PDF; attempting image extraction"
            )
            return self._extract_text_from_image(file_bytes, allow_pdf_fallback=False)

        lines: List[str] = []

        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    # Group words by their vertical position and side of page
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

                        # Append any detected tables as additional lines
                        for table in page.extract_tables() or []:
                            for row in table:
                                row_text = " ".join(cell or "" for cell in row).strip()
                                if row_text:
                                    lines.append(row_text)
                    else:
                        # OCR fallback for scanned pages
                        if easyocr is not None:
                            try:
                                page_image = page.to_image(resolution=300).original
                                img_arr = np.array(page_image)
                                reader = _get_easyocr_reader()
                                ocr_lines = reader.readtext(img_arr, detail=0, paragraph=True)
                                lines.extend(
                                    ln.strip() for ln in ocr_lines if ln.strip()
                                )
                            except Exception as ocr_exc:  # pragma: no cover - defensive
                                logger.warning("EasyOCR failed: %s", ocr_exc)
                        elif Image is not None and pytesseract is not None:
                            try:
                                page_image = page.to_image(resolution=300).original
                                ocr_text = pytesseract.image_to_string(page_image)
                                lines.extend(
                                    ln.strip() for ln in ocr_text.splitlines() if ln.strip()
                                )
                            except Exception as ocr_exc:  # pragma: no cover - defensive
                                logger.warning("OCR failed: %s", ocr_exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("pdfplumber failed: %s", exc)

        # Fallback to PyMuPDF if pdfplumber produced nothing
        if not lines and fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("PyMuPDF failed extracting text: %s", exc)
                img_text = self._extract_text_from_image(
                    file_bytes, allow_pdf_fallback=False
                )
                return img_text
        elif not lines:
            logger.warning("No text extracted from PDF; PyMuPDF not installed")
            return ""

        return "\n".join(lines)

    def _extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX files using python-docx when available."""
        if docx is None:
            logger.warning("python-docx not installed; cannot extract DOCX text")
            return ""
        try:
            document = docx.Document(BytesIO(file_bytes))
            return "\n".join(par.text for par in document.paragraphs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed extracting DOCX text: %s", exc)
            return ""

    def _extract_text_from_image(self, file_bytes: bytes, allow_pdf_fallback: bool = True) -> str:
        """OCR text from image files using EasyOCR or pytesseract."""
        if Image is None:
            logger.warning("PIL not installed; cannot OCR image")
            return ""
        try:
            img = Image.open(BytesIO(file_bytes))
        except UnidentifiedImageError:
            if allow_pdf_fallback:
                logger.warning(
                    "Provided bytes are not a valid image; attempting PDF extraction fallback",
                )
                return self._extract_text_from_pdf(file_bytes)
            logger.warning("Provided bytes are not a valid image")
            return ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed OCR on image: %s", exc)
            return ""
        if easyocr is not None:
            try:
                arr = np.array(img)
                reader = _get_easyocr_reader()
                ocr_lines = reader.readtext(arr, detail=0, paragraph=True)
                return "\n".join(ocr_lines)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("EasyOCR failed: %s", exc)
        if pytesseract is not None:
            try:
                return pytesseract.image_to_string(img)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed OCR on image: %s", exc)
                return ""
        logger.warning("pytesseract not installed; cannot OCR image")
        return ""


    def _extract_text(self, file_bytes: bytes, object_key: str) -> str:
        """Route to appropriate extractor based on file extension."""
        ext = os.path.splitext(object_key)[1].lower()
        if ext == ".pdf":
            return self._extract_text_from_pdf(file_bytes)
        if ext in {".doc", ".docx"}:
            return self._extract_text_from_docx(file_bytes)
        if ext in {".png", ".jpg", ".jpeg"}:
            return self._extract_text_from_image(file_bytes)
        logger.warning("Unsupported document type '%s' for %s", ext, object_key)
        return ""


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
            "buyer_id": ["buyer id", "buyer number"],
            "requisition_id": ["requisition id", "requisition number"],
            "requested_by": ["requested by", "requester"],
            "requested_date": ["requested date", "request date"],
            "invoice_date": ["invoice date"],
            "due_date": ["due date"],
            "invoice_paid_date": ["paid date", "payment date", "invoice paid date"],
            "payment_terms": ["payment terms", "terms"],
            "currency": ["currency"],
            "invoice_amount": ["invoice amount", "amount due"],
            "total_amount": ["total amount", "grand total", "total"],
            "tax_percent": ["tax rate", "tax percent"],
            "tax_amount": ["tax amount", "tax"],
            "invoice_total_incl_tax": [
                "total including tax",
                "total incl tax",
                "total amount incl tax",
            ],
            "exchange_rate_to_usd": ["exchange rate", "exchange rate to usd"],
            "converted_amount_usd": ["usd amount", "amount usd", "converted amount usd"],
            "country": ["country"],
            "region": ["region"],
            "invoice_status": ["invoice status", "status"],
            "ai_flag_required": ["ai flag required", "ai flag"],
            "trigger_type": ["trigger type"],
            "trigger_context_description": ["trigger context", "context description"],
            "created_date": ["created date", "creation date"],
            "order_date": ["order date", "po date"],
            "expected_delivery_date": ["expected delivery date", "delivery date"],
            "ship_to_country": ["ship to country", "shipping country"],
            "delivery_region": ["delivery region", "region"],
            "incoterm": ["incoterm"],
            "incoterm_responsibility": ["incoterm responsibility", "incoterm resp"],
            "delivery_address_line1": ["delivery address line1", "address line 1"],
            "delivery_address_line2": ["delivery address line2", "address line 2"],
            "delivery_city": ["delivery city", "city"],
            "postal_code": ["postal code", "postcode", "zip"],
            "default_currency": ["default currency"],
            "po_status": ["po status", "status"],
            "contract_id": ["contract id", "contract number"],
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
            "Return JSON with any of these keys if matching or present: "
            "invoice_id, po_id, supplier_id, buyer_id, requisition_id, requested_by, "
            "requested_date, invoice_date, due_date, invoice_paid_date, payment_terms, "
            "currency, invoice_amount, tax_percent, tax_amount, invoice_total_incl_tax, "
            "exchange_rate_to_usd, converted_amount_usd, country, region, invoice_status, "
            "ai_flag_required, trigger_type, trigger_context_description, created_date, "
            "order_date, expected_delivery_date, ship_to_country, delivery_region, incoterm, "
            "incoterm_responsibility, total_amount, delivery_address_line1, delivery_address_line2, "
            "delivery_city, postal_code, default_currency, po_status, contract_id.\n"
            f"Text:\n{text[:2000]}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(prompt, model=self.extraction_model, format="json")
            llm_header = json.loads(resp.get("response", "{}"))
            header.update({k: v for k, v in llm_header.items() if v})
        except Exception:
            pass

        numeric_fields = {
            "total_amount",
            "tax_percent",
            "tax_amount",
            "invoice_amount",
            "invoice_total_incl_tax",
            "exchange_rate_to_usd",
            "converted_amount_usd",
        }
        numeric_id_fields = {"buyer_id", "supplier_id", "requisition_id"}
        for field in numeric_fields:
            if field in header and header[field] is not None:
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

    def _extract_line_items_from_pdf_tables(
        self, file_bytes: bytes, doc_type: str
    ) -> List[Dict]:
        """Extract line items by parsing tabular data directly from the PDF."""
        if pdfplumber is None:
            return []
        line_items: List[Dict] = []
        if doc_type == "Invoice":
            column_synonyms = {
                "item_id": ["item", "sku", "part", "product code"],
                "item_description": ["description", "item description"],
                "quantity": ["qty", "quantity"],
                "unit_price": ["unit price", "price", "rate", "unit cost"],
                "unit_of_measure": ["unit", "uom"],
                "line_amount": ["line total", "line amount", "amount", "total"],
                "tax_percent": ["tax %", "tax percent", "tax rate"],
                "tax_amount": ["tax", "tax amount"],
                "total_amount_incl_tax": ["total amount incl tax", "total with tax"],
            }
        else:  # Purchase orders and others
            column_synonyms = {
                "item_id": ["item", "sku", "part", "product code"],
                "item_description": ["description", "item description"],
                "quantity": ["qty", "quantity"],
                "unit_price": ["unit price", "price", "rate", "unit cost"],
                "unit_of_measure": ["unit", "uom"],
                "currency": ["currency"],
                "line_total": ["line total", "amount", "total"],
                "tax_percent": ["tax %", "tax percent", "tax rate"],
                "tax_amount": ["tax", "tax amount"],
                "total_amount": ["total", "total amount", "total with tax"],
            }
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables() or []:
                        df = pd.DataFrame(table).dropna(how="all")
                        if df.empty:
                            continue
                        header = (
                            df.iloc[0]
                            .fillna("")
                            .astype(str)
                            .str.lower()
                            .str.strip()
                            .tolist()
                        )
                        col_map: Dict[int, str] = {}
                        for idx, col in enumerate(header):
                            for field, syns in column_synonyms.items():
                                if any(syn in col for syn in syns):
                                    col_map[idx] = field
                                    break
                        if len(col_map) < 2:
                            continue
                        for _, row in df.iloc[1:].iterrows():
                            item: Dict[str, Any] = {}
                            for idx, field in col_map.items():
                                val = row.iloc[idx]
                                if isinstance(val, str):
                                    val = val.strip()
                                if val not in (None, ""):
                                    item[field] = val
                            if item:
                                line_items.append(item)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Table extraction failed: %s", exc)
        return line_items

    def _normalize_header_fields(self, header: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """Map LLM-extracted header keys to canonical column names."""
        alias_map = {
            "Invoice": {
                "invoice_total": "invoice_amount",
                "total_amount": "invoice_amount",
                "vendor": "vendor_name",
                "supplier": "vendor_name",
                "recipient": "vendor_name",
                "to": "vendor_name",
                "supplier_name": "vendor_name",
            },
            "Purchase_Order": {
                "po_number": "po_id",
                "purchase_order_id": "po_id",
                "vendor": "vendor_name",
                "supplier": "vendor_name",
                "recipient": "vendor_name",
                "to": "vendor_name",
                "supplier_name": "vendor_name",
            },
        }
        mapping = alias_map.get(doc_type, {})
        normalised: Dict[str, Any] = {}
        for key, value in header.items():
            normalised[mapping.get(key, key)] = value
        if normalised.get("vendor_name") and not normalised.get("supplier_id"):
            normalised["supplier_id"] = normalised["vendor_name"]
        return normalised

    def _normalize_line_item_fields(
        self, items: List[Dict[str, Any]], doc_type: str
    ) -> List[Dict[str, Any]]:
        """Rename common line-item field variants based on document type."""
        alias_map = {
            "Invoice": {
                "description": "item_description",
                "qty": "quantity",
                "price": "unit_price",
                "amount": "line_amount",
                "tax": "tax_amount",
            },
            "Purchase_Order": {
                "description": "item_description",
                "qty": "quantity",
                "price": "unit_price",
                "amount": "line_total",
                "uom": "unit_of_measure",
            },
        }
        mapping = alias_map.get(doc_type, {})
        normalised_items: List[Dict[str, Any]] = []
        for item in items:
            normalised: Dict[str, Any] = {}
            for key, value in item.items():
                normalised[mapping.get(key, key)] = value
            normalised_items.append(normalised)
        return normalised_items

    def _extract_structured_data(
        self, text: str, doc_type: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse header and line items using context-aware LLM extraction."""
        header, line_items = self._context_based_extraction(text, doc_type)
        header = self._normalize_header_fields(header or {}, doc_type)
        line_items = self._normalize_line_item_fields(line_items or [], doc_type)
        return header, line_items

    def _context_based_extraction(
        self, text: str, doc_type: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Use the LLM to extract header fields and line items purely from text."""
        schema = {
            "Invoice": {
                "header": [
                    "invoice_id",
                    "po_id",
                    "supplier_id",
                    "buyer_id",
                    "requisition_id",
                    "requested_by",
                    "requested_date",
                    "invoice_date",
                    "due_date",
                    "invoice_paid_date",
                    "payment_terms",
                    "currency",
                    "invoice_amount",
                    "tax_percent",
                    "tax_amount",
                    "invoice_total_incl_tax",
                    "exchange_rate_to_usd",
                    "converted_amount_usd",
                    "country",
                    "region",
                    "invoice_status",
                    "ai_flag_required",
                    "trigger_type",
                    "trigger_context_description",
                    "created_date",
                ],
                "line_items": [
                    "invoice_line_id",
                    "invoice_id",
                    "line_no",
                    "item_id",
                    "item_description",
                    "quantity",
                    "unit_of_measure",
                    "unit_price",
                    "line_amount",
                    "tax_percent",
                    "tax_amount",
                    "total_amount_incl_tax",
                    "po_id",
                    "delivery_date",
                    "country",
                    "region",
                    "created_date",
                    "created_by",
                    "last_modified_by",
                    "last_modified_date",
                ],
            },
            "Purchase_Order": {
                "header": [
                    "po_id",
                    "supplier_id",
                    "buyer_id",
                    "requisition_id",
                    "requested_by",
                    "requested_date",
                    "currency",
                    "order_date",
                    "expected_delivery_date",
                    "ship_to_country",
                    "delivery_region",
                    "incoterm",
                    "incoterm_responsibility",
                    "total_amount",
                    "delivery_address_line1",
                    "delivery_address_line2",
                    "delivery_city",
                    "postal_code",
                    "default_currency",
                    "po_status",
                    "payment_terms",
                    "exchange_rate_to_usd",
                    "converted_amount_usd",
                    "ai_flag_required",
                    "trigger_type",
                    "trigger_context_description",
                    "created_date",
                    "created_by",
                    "last_modified_by",
                    "last_modified_date",
                    "contract_id",
                ],
                "line_items": [
                    "po_line_id",
                    "po_id",
                    "line_number",
                    "item_id",
                    "item_description",
                    "quantity",
                    "unit_price",
                    "unit_of_measure",
                    "currency",
                    "line_total",
                    "tax_percent",
                    "tax_amount",
                    "total_amount",
                    "created_date",
                    "created_by",
                    "last_modified_by",
                    "last_modified_date",
                ],
            },
        }
        fields = schema.get(doc_type)
        if not fields:
            return {}, []
        prompt = (
            f"You are a procurement data extraction agent. "
            f"The supplier name may appear as vendor, recipient or after 'TO'. "
            f"Extract data from this {doc_type.replace('_', ' ').lower()} and respond in JSON. "
            "Use two keys: 'header' and 'line_items'. "
            f"'header' must contain: {fields['header']}. "
            f"'line_items' is a list of objects with: {fields['line_items']}. "
            "Use null when a value is missing.\n"
            f"Text:\n{text[:6000]}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(prompt, model=self.extraction_model, format="json")
            data = json.loads(resp.get("response", "{}")) or {}
            header = data.get("header", {}) or {}
            line_items = data.get("line_items", []) or []
        except Exception:
            header, line_items = {}, []
        return header, line_items

    def _extract_line_items(self, text: str, doc_type: str) -> List[Dict]:
        """Use the LLM to extract line items from a document."""
        if doc_type == "Invoice":
            prompt = (
                "Extract line items from the following invoice. "
                "Return JSON with a list under the key 'line_items' where each item has "
                "line_no, item_id, item_description, quantity, unit_of_measure, unit_price, "
                "line_amount, tax_percent, tax_amount, total_amount_incl_tax.\n"
                f"Text:\n{text[:4000]}"
            )
        elif doc_type == "Purchase_Order":
            prompt = (
                "Extract line items from the following purchase order. "
                "Return JSON with a list under the key 'line_items' where each item has "
                "line_number, item_id, item_description, quantity, unit_price, "
                "unit_of_measure, currency, line_total, tax_percent, tax_amount, "
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
        full_text: str,
        pk_value: str | None,
        doc_type: Any,
        product_type: Any,
        object_key: str,
    ) -> None:
        """Store document chunks and metadata in the vector database."""
        self.agent_nick._initialize_qdrant_collection()
        chunks = self._chunk_text(full_text)

        # Batch encode the chunks so the GPU can be utilised efficiently.  The
        # embedding model automatically uses the GPU when available via
        # :func:`utils.gpu.configure_gpu`.
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

        if points:
            self.agent_nick.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )

    def _vectorize_structured_data(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
        pk_value: str,
    ) -> None:
        """Embed header and line items for fine-grained retrieval.

        Each header and line item is converted to a compact text form and
        embedded individually. This allows downstream retrieval to surface
        specific fields (e.g. a single line item) rather than only the full
        document chunk. Embeddings are stored in the same Qdrant collection
        as the full document vectors, tagged with a ``data_type`` payload so
        callers can filter as needed.
        """

        if not pk_value:
            return

        self.agent_nick._initialize_qdrant_collection()
        points: List[models.PointStruct] = []

        # Header payload -------------------------------------------------
        if header:
            header_text = _dict_to_text(header)
            vec = self.agent_nick.embedding_model.encode(
                [header_text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            points.append(
                models.PointStruct(
                    id=_normalize_point_id(f"{pk_value}_header"),
                    vector=vec.tolist(),
                    payload={
                        "record_id": pk_value,
                        "document_type": _normalize_label(doc_type),
                        "data_type": "header",
                        "content": header_text,
                    },
                )
            )

        # Line item payloads --------------------------------------------
        for idx, item in enumerate(line_items, start=1):
            item_text = _dict_to_text(item)
            vec = self.agent_nick.embedding_model.encode(
                [item_text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            points.append(
                models.PointStruct(
                    id=_normalize_point_id(f"{pk_value}_line_{idx}"),
                    vector=vec.tolist(),
                    payload={
                        "record_id": pk_value,
                        "document_type": _normalize_label(doc_type),
                        "data_type": "line_item",
                        "line_number": idx,
                        "content": item_text,
                    },
                )
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

    def _clean_numeric(self, value: str | int | float) -> Optional[float]:
        """Best-effort parsing of free-form numeric strings.

        The helper is tolerant to common human formats such as ``20%``,
        ``£1,200.50`` or ``(100)`` for negative numbers.  Any non-numeric
        characters, including currency symbols and thousand separators, are
        stripped before casting.

        Parameters
        ----------
        value: str | int | float
            Raw numeric value or string potentially containing decorations.

        Returns
        -------
        Optional[float]
            ``float`` representation if parsing succeeds otherwise ``None``.
        """
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
        # When a percent sign is present the first number usually represents
        # the percentage, otherwise the last number tends to be the amount.
        num_str = numbers[0] if "%" in value_str else numbers[-1]
        try:
            num = float(num_str)
            return -num if is_negative else num
        except ValueError:
            logger.debug("Unable to parse numeric value '%s'", value)
            return None

    def _clean_date(self, value: str) -> Optional[datetime.date]:
        """Parse fuzzy or relative date strings.

        Supports expressions like "Jan 15, 2024 + 30 days" by applying the
        specified offset. Returns ``datetime.date`` objects or ``None`` when
        parsing fails.
        """
        try:
            value_str = str(value)
            # Some documents include a trailing dot after the month name,
            # e.g. ``30 MARCH. 2024`` which confuses the parser.  Strip dots
            # that directly follow a word token.
            value_str = re.sub(r"([A-Za-z])\.", r"\1", value_str)
            match = re.match(r"(.+?)\s*\+\s*(\d+)\s*days", value_str, re.I)
            if match:
                base = parser.parse(match.group(1), fuzzy=True)
                offset = int(match.group(2))
                return (base + timedelta(days=offset)).date()
            return parser.parse(value_str, fuzzy=True).date()
        except Exception:
            logger.warning("Unable to parse date value '%s'", value)
            return None

    def _sanitize_value(self, value, key: Optional[str] = None):
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        numeric_fields = {
            "quantity",
            "unit_price",
            "tax_percent",
            "tax_amount",
            "line_total",
            "line_amount",
            "total_with_tax",
            "total_amount",
            "total_amount_incl_tax",
            "total",
            "invoice_amount",
            "invoice_total_incl_tax",
            "exchange_rate_to_usd",
            "converted_amount_usd",
        }
        date_fields = {
            "invoice_date",
            "due_date",
            "po_date",
            "requested_date",
            "invoice_paid_date",
            "delivery_date",
            "order_date",
            "expected_delivery_date",
            "created_date",
            "last_modified_date",
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
        return value

    def _persist_to_postgres(
        self,
        header: Dict[str, str],
        line_items: List[Dict],
        doc_type: str,
        pk_value: str,
    ) -> None:
        """Persist extracted header and line items to PostgreSQL."""
        try:
            conn = self.agent_nick.get_db_connection()
            with conn:
                self._persist_header_to_postgres(header, doc_type, conn)
                self._persist_line_items_to_postgres(
                    pk_value, line_items, doc_type, header, conn
                )
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist %s data: %s", doc_type, exc)

    def _persist_header_to_postgres(
        self, header: Dict[str, str], doc_type: str, conn=None
    ) -> None:
        table_map = {
            "Invoice": ("proc", "invoice_agent", "invoice_id"),
            "Purchase_Order": ("proc", "purchase_order_agent", "po_id"),
        }
        target = table_map.get(doc_type)
        if not target:
            return
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
                sql_base = f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders}) "
                if update_cols:
                    sql = sql_base + f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_cols}"
                else:
                    sql = sql_base + f"ON CONFLICT ({pk_col}) DO NOTHING"
                cur.execute(sql, list(payload.values()))
            if close_conn:
                conn.commit()
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist %s data: %s", doc_type, exc)
            if close_conn:
                conn.rollback()
        finally:
            if close_conn:
                conn.close()

    def _persist_line_items_to_postgres(
        self,
        pk_value: str,
        line_items: List[Dict],
        doc_type: str,
        header: Dict[str, str],
        conn=None,
    ) -> None:
        table_map = {
            "Invoice": ("proc", "invoice_line_items_agent", "invoice_id", "line_no"),
            "Purchase_Order": (
                "proc",
                "po_line_items_agent",
                "po_id",
                "line_number",
            ),
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
                # Remove existing rows to avoid stale line items lingering
                cur.execute(
                    f"DELETE FROM {schema}.{table} WHERE {fk_col} = %s",
                    (pk_value,),
                )
                numeric_fields = {
                    "quantity",
                    "unit_price",
                    "tax_percent",
                    "tax_amount",
                    "line_total",
                    "line_amount",
                    "total_with_tax",
                    "total_amount_incl_tax",
                    "total_amount",
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
                    # generate synthetic line identifiers when supported by the schema
                    if doc_type == "Purchase_Order" and "po_line_id" in columns:
                        payload.setdefault("po_line_id", f"{pk_value}-{line_value}")
                    if doc_type == "Invoice" and "invoice_line_id" in columns:
                        payload.setdefault("invoice_line_id", f"{pk_value}-{line_value}")
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
                                logger.warning(
                                    "Dropping field %s due to non-numeric value. Payload: %s", k, payload
                                )
                                continue
                            val = float(val)
                        sanitized[k] = val
                    cols = ", ".join(sanitized.keys())
                    placeholders = ", ".join(["%s"] * len(sanitized))
                    sql = f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders})"
                    cur.execute(sql, list(sanitized.values()))
            if close_conn:
                conn.commit()
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist line items for %s: %s", doc_type, exc)
            if close_conn:
                conn.rollback()
        finally:
            if close_conn:
                conn.close()

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
