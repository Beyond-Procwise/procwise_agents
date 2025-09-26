from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TableSchema:
    columns: List[str]
    required: List[str] = field(default_factory=list)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)


PROCUREMENT_SCHEMAS: Dict[str, TableSchema] = {
    "proc.invoice_agent": TableSchema(
        columns=[
            "invoice_id",
            "po_id",
            "supplier_name",
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
            "created_by",
            "last_modified_by",
            "last_modified_date",
            "vendor_name",
        ],
        required=["invoice_id", "supplier_name", "invoice_total_incl_tax"],
        synonyms={
            "invoice_id": ["invoice number", "invoice no", "bill number"],
            "supplier_name": ["vendor", "vendor name"],
            "invoice_total_incl_tax": ["total", "total amount", "amount due", "grand total"],
            "invoice_amount": ["subtotal"],
            "tax_amount": ["vat", "gst", "tax"],
        },
    ),
    "proc.invoice_line_items_agent": TableSchema(
        columns=[
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
        required=["invoice_id", "line_no", "item_description"],
        synonyms={
            "line_no": ["line", "line #", "item #"],
            "item_description": ["description", "details"],
            "unit_of_measure": ["uom", "unit"],
            "line_amount": ["amount", "line total"],
        },
    ),
    "proc.purchase_order_agent": TableSchema(
        columns=[
            "po_id",
            "supplier_name",
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
        required=["po_id", "supplier_name", "total_amount"],
        synonyms={
            "po_id": ["purchase order number", "po number", "po #"],
            "supplier_name": ["vendor", "vendor name"],
            "total_amount": ["order total", "grand total", "total"],
        },
    ),
    "proc.po_line_items_agent": TableSchema(
        columns=[
            "po_id",
            "po_line_id",
            "line_number",
            "item_id",
            "item_description",
            "quote_number",
            "quantity",
            "unit_price",
            "unit_of_measue",
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
        required=["po_id", "line_number", "item_description"],
        synonyms={
            "line_number": ["line", "line #", "item #"],
            "unit_of_measue": ["unit", "uom"],
            "line_total": ["amount", "total"],
        },
    ),
}

DOC_TYPE_TO_TABLE = {
    "Invoice": (
        "proc.invoice_agent",
        "proc.invoice_line_items_agent",
    ),
    "Purchase_Order": (
        "proc.purchase_order_agent",
        "proc.po_line_items_agent",
    ),
}

_KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-/]{2,40})\s*[:\-]\s*(.+)$")
_TOTAL_STOP_WORDS = re.compile(r"\b(Subtotal|Tax|VAT|GST|Total|Amount Due)\b", re.IGNORECASE)


def _normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def extract_structured_content(text: str, doc_type: str) -> Dict[str, Any]:
    header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
    header_schema = PROCUREMENT_SCHEMAS.get(header_table) if header_table else None
    line_schema = PROCUREMENT_SCHEMAS.get(line_table) if line_table else None

    header_pairs = _extract_key_value_pairs(text)
    header = _map_pairs_to_schema(header_pairs, header_schema) if header_schema else {}
    line_items = _extract_line_items(text, line_schema) if line_schema else []

    return {"header": header, "line_items": line_items}


def _extract_key_value_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for line in text.splitlines():
        match = _KEY_VALUE_PATTERN.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            pairs.setdefault(_normalise(key), value)
    return pairs


def _map_pairs_to_schema(pairs: Dict[str, str], schema: TableSchema | None) -> Dict[str, Any]:
    if not schema:
        return {}
    mapped: Dict[str, Any] = {}
    for column in schema.columns:
        candidates = {column, column.replace("_", " ")}
        candidates.update(schema.synonyms.get(column, []))
        best_score = 0.0
        best_value: Any = None
        for candidate in candidates:
            norm_candidate = _normalise(candidate)
            for key, value in pairs.items():
                score = _token_overlap(norm_candidate, key)
                if score > best_score:
                    best_score = score
                    best_value = value
        if best_value is not None and best_score >= 0.6:
            mapped[column] = best_value
    return mapped


def _token_overlap(a: str, b: str) -> float:
    set_a = set(re.findall(r"[a-z0-9]+", a))
    set_b = set(re.findall(r"[a-z0-9]+", b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extract_line_items(text: str, schema: TableSchema | None) -> List[Dict[str, Any]]:
    if not schema:
        return []
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if _TOTAL_STOP_WORDS.search(line):
            break
        tokens = [tok for tok in re.split(r"\s{2,}", line.strip()) if tok]
        if not tokens:
            continue
        lower_tokens = [tok.lower() for tok in tokens]
        if not headers and any("qty" in token for token in lower_tokens):
            headers = [re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_") for token in tokens]
            continue
        if not headers:
            continue
        if len(tokens) < len(headers):
            continue
        row = {headers[idx]: tokens[idx].strip() for idx in range(len(headers))}
        rows.append(row)
    return rows
