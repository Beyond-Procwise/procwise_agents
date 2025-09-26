from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


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
            "invoice_id": [
                "invoice number",
                "invoice no",
                "invoice #",
                "tax invoice",
                "bill number",
            ],
            "po_id": ["purchase order", "po number", "po no", "po #"],
            "supplier_name": ["vendor", "vendor name", "seller", "supplier"],
            "buyer_id": ["buyer", "bill to"],
            "requested_by": ["requested by", "requester"],
            "requested_date": ["requested date", "request date"],
            "invoice_date": ["invoice date", "date of invoice"],
            "due_date": ["due date", "payment due"],
            "invoice_paid_date": ["paid date", "payment date"],
            "payment_terms": ["payment terms", "terms"],
            "currency": ["currency", "curr"],
            "invoice_amount": ["subtotal", "invoice amount"],
            "tax_percent": ["tax %", "tax percent", "vat %", "gst %"],
            "tax_amount": ["vat", "gst", "tax", "tax amt"],
            "invoice_total_incl_tax": [
                "total",
                "total amount",
                "amount due",
                "grand total",
                "invoice total",
                "total incl tax",
            ],
            "country": ["country"],
            "region": ["region"],
            "invoice_status": ["status"],
            "vendor_name": ["vendor", "supplier name"],
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
            "invoice_line_id": ["line id"],
            "line_no": ["line", "line #", "item #", "row"],
            "item_id": ["item code", "sku", "product", "part number"],
            "item_description": ["description", "details", "item"],
            "quantity": ["qty", "quantity", "units"],
            "unit_of_measure": ["uom", "unit", "measure"],
            "unit_price": ["unit price", "price", "rate"],
            "line_amount": ["amount", "line total", "extended price"],
            "tax_percent": ["tax %", "vat %", "gst %"],
            "tax_amount": ["tax", "vat", "gst"],
            "total_amount_incl_tax": ["total", "total incl tax", "gross"],
            "po_id": ["po", "purchase order"],
            "delivery_date": ["delivery", "ship date"],
            "country": ["country"],
            "region": ["region"],
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
            "po_id": [
                "purchase order number",
                "po number",
                "po #",
                "po no",
                "order number",
            ],
            "supplier_name": ["vendor", "vendor name", "supplier"],
            "buyer_id": ["buyer", "purchaser"],
            "requested_by": ["requested by", "requester"],
            "requested_date": ["requested date", "request date"],
            "order_date": ["order date", "po date"],
            "expected_delivery_date": ["delivery date", "expected delivery", "ship date"],
            "ship_to_country": ["ship to country", "delivery country"],
            "delivery_region": ["region", "delivery region"],
            "incoterm": ["incoterms"],
            "incoterm_responsibility": ["incoterm responsibility", "inco responsibility"],
            "total_amount": ["order total", "grand total", "total", "total amount"],
            "delivery_address_line1": ["delivery address", "ship to"],
            "delivery_city": ["city"],
            "postal_code": ["postal code", "zip"],
            "default_currency": ["default currency"],
            "po_status": ["status"],
            "payment_terms": ["payment terms", "terms"],
            "exchange_rate_to_usd": ["exchange rate", "fx rate"],
            "converted_amount_usd": ["amount usd", "usd total"],
            "contract_id": ["contract", "contract number"],
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
            "po_line_id": ["line id"],
            "line_number": ["line", "line #", "item #", "row"],
            "item_id": ["item code", "sku", "product", "part number"],
            "item_description": ["description", "details", "item"],
            "quote_number": ["quote", "quote #"],
            "quantity": ["qty", "quantity", "units"],
            "unit_price": ["unit price", "price", "rate"],
            "unit_of_measue": ["uom", "unit", "measure"],
            "currency": ["currency"],
            "line_total": ["amount", "line total", "extended price"],
            "tax_percent": ["tax %", "vat %", "gst %"],
            "tax_amount": ["tax", "vat", "gst"],
            "total_amount": ["total", "total amount", "total incl tax"],
        },
    ),
    "proc.quote_agent": TableSchema(
        columns=[
            "quote_id",
            "supplier_id",
            "buyer_id",
            "quote_date",
            "validity_date",
            "currency",
            "total_amount",
            "tax_percent",
            "tax_amount",
            "total_amount_incl_tax",
            "po_id",
            "country",
            "region",
            "ai_flag_required",
            "trigger_type",
            "trigger_context_description",
            "created_date",
            "created_by",
            "last_modified_by",
            "last_modified_date",
        ],
        required=["quote_id", "supplier_id", "total_amount"],
        synonyms={
            "quote_id": ["quote number", "quote no", "quotation", "quote #"],
            "supplier_id": ["supplier", "vendor", "vendor id"],
            "buyer_id": ["buyer"],
            "quote_date": ["quote date", "date"],
            "validity_date": ["valid until", "validity", "expiry"],
            "currency": ["currency"],
            "total_amount": ["total", "amount", "quote total"],
            "tax_percent": ["tax %", "vat %", "gst %"],
            "tax_amount": ["tax", "vat", "gst"],
            "total_amount_incl_tax": ["total incl tax", "grand total", "total amount"],
            "po_id": ["po", "purchase order"],
            "country": ["country"],
            "region": ["region"],
        },
    ),
    "proc.quote_line_items_agent": TableSchema(
        columns=[
            "quote_line_id",
            "quote_id",
            "line_number",
            "item_id",
            "item_description",
            "quantity",
            "unit_of_measure",
            "unit_price",
            "line_total",
            "tax_percent",
            "tax_amount",
            "total_amount",
            "currency",
            "created_date",
            "created_by",
            "last_modified_by",
            "last_modified_date",
        ],
        required=["quote_id", "line_number", "item_description"],
        synonyms={
            "quote_line_id": ["line id"],
            "line_number": ["line", "line #", "item #", "row"],
            "item_id": ["item code", "sku", "product", "part number"],
            "item_description": ["description", "details", "item"],
            "quantity": ["qty", "quantity", "units"],
            "unit_of_measure": ["uom", "unit", "measure"],
            "unit_price": ["unit price", "price", "rate"],
            "line_total": ["amount", "line total", "extended price"],
            "tax_percent": ["tax %", "vat %", "gst %"],
            "tax_amount": ["tax", "vat", "gst"],
            "total_amount": ["total", "total amount", "total incl tax"],
            "currency": ["currency"],
        },
    ),
    "proc.contracts": TableSchema(
        columns=[
            "contract_id",
            "contract_title",
            "contract_type",
            "supplier_id",
            "buyer_org_id",
            "contract_start_date",
            "contract_end_date",
            "currency",
            "total_contract_value",
            "spend_category",
            "business_unit_id",
            "cost_centre_id",
            "is_amendment",
            "parent_contract_id",
            "auto_renew_flag",
            "renewal_term",
            "contract_lifecycle_status",
            "jurisdiction",
            "governing_law",
            "contract_signatory_name",
            "contract_signatory_role",
            "payment_terms",
            "risk_assessment_completed",
            "created_date",
            "created_by",
            "last_modified_by",
            "last_modified_date",
        ],
        required=["contract_id", "supplier_id", "contract_title"],
        synonyms={
            "contract_id": ["contract number", "contract no", "contract #"],
            "contract_title": ["contract title", "contract name", "agreement"],
            "contract_type": ["type"],
            "supplier_id": ["supplier", "vendor", "vendor id"],
            "buyer_org_id": ["buyer", "buyer organisation", "buyer org"],
            "contract_start_date": ["start date", "effective date", "commencement"],
            "contract_end_date": ["end date", "expiry", "expiration", "termination date"],
            "currency": ["currency"],
            "total_contract_value": ["total value", "contract value", "amount"],
            "spend_category": ["category", "spend category"],
            "business_unit_id": ["business unit", "business unit id"],
            "cost_centre_id": ["cost centre", "cost center"],
            "is_amendment": ["amendment"],
            "parent_contract_id": ["parent contract"],
            "auto_renew_flag": ["auto renew", "auto-renew"],
            "renewal_term": ["renewal term", "renewal"],
            "contract_lifecycle_status": ["status", "lifecycle status"],
            "jurisdiction": ["jurisdiction"],
            "governing_law": ["governing law", "law"],
            "contract_signatory_name": ["signatory", "signatory name", "signed by"],
            "contract_signatory_role": ["signatory role", "title"],
            "payment_terms": ["payment terms", "terms"],
            "risk_assessment_completed": ["risk assessment", "risk completed"],
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
    "Quote": (
        "proc.quote_agent",
        "proc.quote_line_items_agent",
    ),
    "Contract": (
        "proc.contracts",
        None,
    ),
}

_KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-/]{2,40})\s*[:\-]\s*(.+)$")
_TOTAL_STOP_WORDS = re.compile(r"\b(Subtotal|Tax|VAT|GST|Total|Amount Due)\b", re.IGNORECASE)
_SPLIT_PATTERN = re.compile(r"\s{2,}|\t+")


def _normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _schema_terms(schema: TableSchema) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for column in schema.columns:
        terms = {
            _normalise(column),
            _normalise(column.replace("_", " ")),
        }
        for synonym in schema.synonyms.get(column, []):
            terms.add(_normalise(synonym))
        lookup[column] = [term for term in terms if term]
    return lookup


def _match_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    overlap = _token_overlap(a, b)
    ratio = SequenceMatcher(None, a, b).ratio()
    return max(overlap, ratio)


def _best_schema_match(label: str, schema: TableSchema, used: set[str] | None = None) -> Tuple[str | None, float]:
    terms = _schema_terms(schema)
    best_col: str | None = None
    best_score = 0.0
    for column, candidates in terms.items():
        if used and column in used:
            continue
        for candidate in candidates:
            score = _match_score(label, candidate)
            if score > best_score:
                best_col = column
                best_score = score
    return best_col, best_score


def _looks_like_label(label: str) -> bool:
    label = label.strip()
    if len(label) < 3 or len(label) > 60:
        return False
    if not re.search(r"[A-Za-z]", label):
        return False
    if re.fullmatch(r"[A-Za-z]{1,3}", label):
        return False
    return True


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
        stripped = line.strip()
        if not stripped:
            continue
        match = _KEY_VALUE_PATTERN.match(stripped)
        if not match:
            parts = _SPLIT_PATTERN.split(stripped, maxsplit=1)
            if len(parts) == 2 and _looks_like_label(parts[0]) and parts[1].strip():
                key, value = parts[0].strip(), parts[1].strip()
            else:
                continue
        else:
            key = match.group(1).strip()
            value = match.group(2).strip()
        if key and value:
            pairs.setdefault(_normalise(key), value)
    return pairs


def _map_pairs_to_schema(pairs: Dict[str, str], schema: TableSchema | None) -> Dict[str, Any]:
    if not schema:
        return {}
    mapped: Dict[str, Any] = {}
    used: set[str] = set()
    for key, value in pairs.items():
        column, score = _best_schema_match(key, schema, used)
        if column and score >= 0.55:
            mapped[column] = value
            used.add(column)
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
    header_map: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    used_columns: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = [tok for tok in _SPLIT_PATTERN.split(stripped) if tok]
        if not tokens:
            continue
        if not headers:
            match_count = 0
            provisional_headers: List[str] = []
            provisional_map: Dict[int, str] = {}
            used_columns.clear()
            for idx, token in enumerate(tokens):
                column, score = _best_schema_match(_normalise(token), schema, used_columns)
                if column and score >= 0.5:
                    provisional_map[idx] = column
                    provisional_headers.append(token)
                    used_columns.add(column)
                    match_count += 1
            if match_count >= max(2, len(tokens) // 2):
                headers = tokens
                header_map = provisional_map
            continue
        if not headers:
            continue
        if _TOTAL_STOP_WORDS.search(stripped):
            break
        if len(tokens) < len(headers):
            tokens.extend([""] * (len(headers) - len(tokens)))
        row: Dict[str, Any] = {}
        for idx, token in enumerate(tokens[: len(headers)]):
            column = header_map.get(idx)
            if not column:
                column, score = _best_schema_match(_normalise(headers[idx]), schema, None)
                if not column or score < 0.5:
                    continue
                header_map[idx] = column
            row[column] = token.strip()
        if any(value for value in row.values()):
            rows.append(row)
    return rows
