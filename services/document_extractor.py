"""Lightweight document extraction pipeline for procurement artefacts.

This module focuses on deterministic parsing so that unit tests can
exercise the end-to-end flow without depending on the heavy
``DataExtractionAgent``.  The extractor is able to ingest plain text
documents (for the digital samples in the test-suite) as well as PDF
files when the optional ``pdfplumber`` dependency is available.  It
derives header fields directly from the detected content and persists
the results into the dedicated ``proc.raw_*`` staging tables.

The goal is to ensure that every ingestion produces a structured JSON
payload that captures headers, line items, and any detected tabular
content.  The JSON payload is stored in Postgres (or a compatible
DB-API connection used during tests) so downstream agents can join onto
the raw tables before promoting the data to curated schemas.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:  # Optional dependency – only required for PDF handling in tests.
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency is best-effort
    pdfplumber = None  # type: ignore

logger = logging.getLogger(__name__)

RAW_TABLE_MAPPING = {
    "Invoice": "proc.raw_invoice",
    "Purchase_Order": "proc.raw_purchase_order",
    "Contract": "proc.raw_contracts",
    "Quote": "proc.raw_quotes",
}

# Canonical procurement structures derived from ``docs/procurement_table_reference.md``.
# The keywords are used to align observed keys and column headers with the
# database schema so extracted payloads remain faithful to downstream tables.
PROCUREMENT_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "Invoice": {
        "header_fields": [
            "invoice_id",
            "po_id",
            "supplier_name",
            "invoice_date",
            "due_date",
            "invoice_paid_date",
            "payment_terms",
            "currency",
            "invoice_amount",
            "tax_percent",
            "tax_amount",
            "invoice_total_incl_tax",
        ],
        "line_item_fields": [
            "item_description",
            "item_id",
            "quantity",
            "unit_of_measure",
            "unit_price",
            "line_amount",
            "tax_percent",
            "tax_amount",
            "total_amount_incl_tax",
        ],
    },
    "Purchase_Order": {
        "header_fields": [
            "po_id",
            "supplier_name",
            "buyer_id",
            "requisition_id",
            "requested_by",
            "requested_date",
            "order_date",
            "expected_delivery_date",
            "currency",
            "total_amount",
            "payment_terms",
        ],
        "line_item_fields": [
            "item_description",
            "item_id",
            "line_number",
            "quantity",
            "unit_price",
            "unit_of_measure",
            "line_total",
            "tax_percent",
            "tax_amount",
        ],
    },
    "Contract": {
        "header_fields": [
            "contract_id",
            "contract_title",
            "contract_type",
            "supplier_name",
            "contract_start_date",
            "contract_end_date",
            "currency",
            "total_contract_value",
            "payment_terms",
        ],
        "line_item_fields": [
            "item_description",
            "quantity",
            "unit_price",
            "line_amount",
        ],
    },
    "Quote": {
        "header_fields": [
            "quote_id",
            "supplier_name",
            "buyer_id",
            "quote_date",
            "validity_date",
            "currency",
            "total_amount",
            "tax_percent",
            "tax_amount",
            "total_amount_incl_tax",
        ],
        "line_item_fields": [
            "item_description",
            "quantity",
            "unit_price",
            "line_amount",
            "tax_percent",
            "tax_amount",
        ],
    },
}

HEADER_KEYWORDS: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {
    "*": {
        "supplier_name": [("supplier",), ("vendor",), ("seller",), ("payee",)],
        "payment_terms": [("payment", "terms"), ("terms", "payment")],
        "currency": [("currency",), ("curr",), ("currency", "code"), ("amount", "usd"), ("amount", "eur")],
        "total_amount": [("total", "amount"), ("grand", "total"), ("amount", "due")],
        "tax_amount": [("tax", "amount")],
        "tax_percent": [("tax", "percent"), ("tax", "rate")],
    },
    "Invoice": {
        "invoice_id": [("invoice", "number"), ("invoice", "no"), ("invoice", "#")],
        "invoice_date": [("invoice", "date")],
        "due_date": [("due", "date")],
        "invoice_paid_date": [("paid", "date"), ("payment", "date")],
        "invoice_total_incl_tax": [
            ("total", "incl", "tax"),
            ("invoice", "total"),
            ("total", "due"),
            ("total", "invoice"),
            ("total", "amount"),
        ],
        "invoice_amount": [("invoice", "amount"), ("amount", "due"), ("amount", "payable")],
        "po_id": [
            ("purchase", "order", "number"),
            ("purchase", "order", "no"),
            ("po", "number"),
            ("po", "no"),
        ],
    },
    "Purchase_Order": {
        "po_id": [("purchase", "order", "number"), ("purchase", "order", "no"), ("po", "number"), ("po", "#")],
        "order_date": [("order", "date")],
        "requested_date": [("requested", "date")],
        "expected_delivery_date": [("delivery", "date"), ("expected", "delivery")],
        "total_amount": [("total", "amount"), ("order", "total")],
    },
    "Contract": {
        "contract_id": [("contract", "number"), ("contract", "no"), ("contract", "#")],
        "contract_title": [("contract", "title"), ("title", "contract")],
        "contract_start_date": [("start", "date"), ("commencement", "date")],
        "contract_end_date": [("end", "date"), ("expiry", "date"), ("expiration", "date")],
        "total_contract_value": [("total", "value"), ("contract", "value")],
    },
    "Quote": {
        "quote_id": [("quote", "number"), ("quotation", "number"), ("quote", "#"), ("quotation", "#")],
        "quote_date": [("quote", "date"), ("quotation", "date")],
        "validity_date": [("valid", "until"), ("valid", "date"), ("expiry", "date")],
        "total_amount": [("total", "amount"), ("quote", "total")],
    },
}

LINE_KEYWORDS: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {
    "*": {
        "item_description": [
            ("item",),
            ("description",),
            ("product",),
            ("service",),
        ],
        "quantity": [("qty",), ("quantity",), ("units",), ("unit", "qty")],
        "unit_price": [("unit", "price"), ("price", "unit"), ("unit", "cost")],
        "line_amount": [("line", "total"), ("line", "amount"), ("amount",), ("total",)],
        "unit_of_measure": [("uom",), ("unit", "measure"), ("measure",)],
    },
    "Purchase_Order": {
        "line_total": [("line", "total"), ("total",)],
    },
    "Quote": {
        "line_amount": [("amount",), ("line", "total")],
    },
    "Invoice": {
        "line_amount": [("line", "total"), ("line", "amount"), ("amount",)],
    },
}


class _LocalModelExtractor:
    """Thin wrapper around a local Ollama model for structured extraction."""

    def __init__(
        self,
        preferred_models: Optional[Iterable[str]] = None,
        *,
        chat_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._preferred_models = tuple(preferred_models or ("qwen3", "phi4"))
        self._chat_options = chat_options or {}
        self._ollama = None
        self._model_name: Optional[str] = None

        try:  # pragma: no cover - optional dependency
            import ollama  # type: ignore

            self._ollama = ollama
        except Exception:
            logger.debug("Ollama client not available for local extraction")
            return

        self._model_name = self._resolve_model_name()

    def _resolve_model_name(self) -> Optional[str]:  # pragma: no cover - optional dependency
        if not self._ollama:
            return None

        for model_name in self._preferred_models:
            if not model_name:
                continue
            try:
                self._ollama.show(model_name)
            except Exception:
                continue
            return model_name
        return None

    def available(self) -> bool:
        return bool(self._ollama and self._model_name)

    def extract(
        self,
        text: str,
        document_type: str,
        *,
        field_hints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.available():
            return None
        if not text.strip():
            return None

        header_columns: List[str] = []
        line_columns: List[str] = []
        table_headers: List[List[str]] = []
        if field_hints:
            header_columns = sorted({str(col) for col in field_hints.get("header_fields", [])})
            line_columns = sorted({str(col) for col in field_hints.get("line_item_fields", [])})
            table_headers = [
                [str(value) for value in headers]
                for headers in field_hints.get("table_headers", [])
            ]

        prompt = textwrap.dedent(
            f"""
            You are a specialised procurement data extractor.
            Determine the precise header fields and line items for a {document_type} document.
            Use the observed fields to remain consistent with the document formatting.
            Header columns already detected: {header_columns}.
            Line item columns already detected: {line_columns}.
            Table headers observed: {table_headers}.
            If a column is absent from the text, omit it.
            Represent amounts and dates exactly as they appear.

            Return a JSON object with keys:
              - document_type: canonical type name
              - header: object of header fields
              - line_items: list of objects
              - tables: list of tables with "headers" and "rows" keys

            Document text:
            ---
            {text}
            ---
            """
        ).strip()

        try:  # pragma: no cover - optional dependency
            response = self._ollama.chat(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You convert procurement documents into structured JSON. "
                            "Always respond with valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                options=self._chat_options,
                format="json",
            )
        except Exception:
            logger.warning("Local model extraction failed", exc_info=True)
            return None

        message = (response or {}).get("message", {})
        content = message.get("content", "").strip()
        if not content:
            return None

        payload = self._parse_json(content)
        return payload

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^```", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[: -3].strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            logger.warning("Unable to parse JSON returned by local model")
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed


@dataclass
class ExtractionResult:
    """Serialisable representation of an extracted document."""

    document_id: str
    document_type: str
    source_file: str
    header: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    raw_text: str
    metadata: Dict[str, Any]
    schema_reference: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "source_file": self.source_file,
            "header": self.header,
            "line_items": self.line_items,
            "tables": self.tables,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
            "schema_reference": self.schema_reference,
        }


class DocumentExtractor:
    """Parse procurement documents into structured JSON and persist them."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        reference_path: Optional[Path] = None,
        *,
        llm_client: Optional[_LocalModelExtractor] = None,
        preferred_models: Optional[Iterable[str]] = None,
        chat_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._ensured_tables: set[str] = set()
        self._llm = llm_client or _LocalModelExtractor(
            preferred_models=preferred_models,
            chat_options=chat_options,
        )
        self._db_dialect = "generic"
        try:
            with closing(self._connection_factory()) as conn:
                module_name = getattr(type(conn), "__module__", "").lower()
                if "sqlite" in module_name:
                    self._db_dialect = "sqlite"
                elif "psycopg" in module_name or "postgres" in module_name:
                    self._db_dialect = "postgres"
        except Exception:
            logger.debug("Unable to infer database dialect for identifier quoting", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        source: Path | str,
        *,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_label: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract a document from ``source`` and persist it.

        Parameters
        ----------
        source:
            File system path to the document.  Plain text files are
            supported out of the box while PDF parsing is enabled when
            :mod:`pdfplumber` is available.
        document_type:
            Optional explicit override for the detected document type.
        metadata:
            Additional contextual information (for example whether the
            document was scanned).  The metadata is stored in the raw
            table alongside the structured payload.
        """

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Document not found at {path}")

        source_name = source_label or path.name

        text, detected_tables = self._extract_text_and_tables(path)
        detected_type = document_type or self._detect_document_type(text)
        if detected_type not in RAW_TABLE_MAPPING:
            logger.warning(
                "Document type '%s' not recognised; defaulting to Contract",
                detected_type,
            )
            detected_type = "Contract"

        header = self._extract_header_fields(text, detected_type)
        line_items, derived_tables = self._extract_line_items_from_text(
            text, detected_type
        )
        field_hints = self._build_field_hints(
            detected_type,
            header,
            line_items,
            detected_tables,
            derived_tables,
        )

        llm_payload = self._invoke_local_model(text, detected_type, field_hints)
        if llm_payload:
            llm_type = self._normalise_document_type(
                llm_payload.get("document_type") if isinstance(llm_payload, dict) else None
            )
            if llm_type and llm_type in RAW_TABLE_MAPPING:
                detected_type = llm_type

            header = self._merge_header_fields(
                header, llm_payload.get("header"), detected_type
            )
            llm_line_items = self._normalise_llm_line_items(
                llm_payload.get("line_items"), detected_type
            )
            line_items = self._merge_line_items(line_items, llm_line_items)
            derived_tables.extend(
                self._normalise_llm_tables(llm_payload.get("tables"), detected_type)
            )

        tables = self._normalise_tables(
            detected_tables, line_items, derived_tables, detected_type
        )
        if not line_items and tables:
            line_items = [dict(row) for row in tables[0]["rows"]]

        schema_reference = self._build_structure_summary(
            detected_type,
            header,
            line_items,
            tables,
        )

        metadata_payload = dict(metadata or {})

        result = ExtractionResult(
            document_id=self._generate_document_id(path, header),
            document_type=detected_type,
            source_file=source_name,
            header=header,
            line_items=line_items,
            tables=tables,
            raw_text=text,
            metadata=metadata_payload,
            schema_reference=schema_reference,
        )

        self._persist_result(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_document_id(self, path: Path, header: Dict[str, Any]) -> str:
        for key in ("invoice_id", "po_id", "contract_id", "quote_id"):
            value = header.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        stem = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{stem or 'document'}-{timestamp}"

    def _detect_document_type(self, text: str) -> str:
        lowered = text.lower()
        keyword_map = {
            "Invoice": ["invoice", "tax invoice", "amount due"],
            "Purchase_Order": ["purchase order", "po number", "order date"],
            "Contract": ["contract", "agreement", "terms"],
            "Quote": ["quotation", "quote", "proposal"],
        }
        for doc_type, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords):
                return doc_type
        return "Contract"  # Fallback to the safest structure

    def _extract_text_and_tables(self, path: Path) -> Tuple[str, List[List[List[str]]]]:
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            return text, []

        if path.suffix.lower() == ".pdf" and pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    text_fragments: List[str] = []
                    tables: List[List[List[str]]] = []
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text_fragments.append(page_text)
                        page_tables = page.extract_tables() or []
                        for table in page_tables:
                            if table:
                                tables.append(table)
                    return "\n".join(text_fragments), tables
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Failed to parse PDF %s", path)
                return "", []

        # Fallback – treat as plain text for unknown extensions.
        try:
            return path.read_text(encoding="utf-8"), []
        except Exception:
            logger.exception("Unable to read document %s", path)
            return "", []

    def _invoke_local_model(
        self,
        text: str,
        document_type: str,
        field_hints: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_llm", None):
            return None
        extract_fn = getattr(self._llm, "extract", None)
        if not callable(extract_fn):
            return None
        try:
            return extract_fn(text, document_type, field_hints=field_hints)
        except Exception:
            logger.warning("Local model invocation encountered an error", exc_info=True)
            return None

    def _build_field_hints(
        self,
        document_type: str,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        raw_tables: List[List[List[str]]],
        derived_tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        canonical = PROCUREMENT_STRUCTURE.get(document_type, {})

        header_fields = [key for key, value in header.items() if self._clean_cell(value)]
        header_fields.extend(canonical.get("header_fields", []))
        line_item_fields: set[str] = set(canonical.get("line_item_fields", []))
        for item in line_items:
            line_item_fields.update(item.keys())

        table_headers: List[List[str]] = []
        for table in derived_tables:
            headers = list(table.get("headers", []))
            if headers:
                table_headers.append(headers)
                line_item_fields.update(headers)

        for table in raw_tables:
            if not table or not table[0]:
                continue
            headers = [
                self._normalise_line_header(self._clean_cell(cell), document_type)
                for cell in table[0]
            ]
            normalised_headers = [header for header in headers if header]
            if normalised_headers:
                table_headers.append(normalised_headers)
                line_item_fields.update(normalised_headers)

        return {
            "header_fields": sorted(dict.fromkeys(header_fields)),
            "line_item_fields": sorted(line_item_fields),
            "table_headers": table_headers,
        }

    def _build_structure_summary(
        self,
        document_type: str,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        canonical = PROCUREMENT_STRUCTURE.get(document_type, {})
        header_fields = sorted(header.keys())
        line_fields: set[str] = set()
        for item in line_items:
            line_fields.update(item.keys())
        table_summaries = [
            {
                "headers": list(table.get("headers", [])),
                "row_count": len(table.get("rows", [])),
            }
            for table in tables
        ]
        return {
            "document_type": document_type,
            "header_fields": header_fields,
            "line_item_fields": sorted(line_fields),
            "table_summaries": table_summaries,
            "expected_header_fields": canonical.get("header_fields", []),
            "expected_line_item_fields": canonical.get("line_item_fields", []),
        }

    def _normalise_tables(
        self,
        raw_tables: List[List[List[str]]],
        line_items: List[Dict[str, Any]],
        derived_tables: List[Dict[str, Any]],
        document_type: str,
    ) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = [
            {"headers": table["headers"], "rows": table["rows"]}
            for table in derived_tables
        ]

        for raw in raw_tables:
            if not raw:
                continue
            header_row = [
                self._normalise_line_header(self._clean_cell(cell), document_type)
                for cell in raw[0]
            ]
            rows = []
            for row in raw[1:]:
                row_dict: Dict[str, Any] = {}
                for idx, cell in enumerate(row):
                    key = (
                        header_row[idx]
                        if idx < len(header_row) and header_row[idx]
                        else f"column_{idx+1}"
                    )
                    row_dict[key] = self._clean_cell(cell)
                if any(str(value).strip() for value in row_dict.values()):
                    rows.append(row_dict)
            if rows:
                tables.append({"headers": header_row, "rows": rows})

        if not tables and line_items:
            # Derive a synthetic table from the line items to ensure table
            # consumers always have a rectangular structure to work with.
            headers = sorted({key for item in line_items for key in item.keys()})
            rows = []
            for item in line_items:
                rows.append({header: item.get(header) for header in headers})
            tables.append({"headers": headers, "rows": rows})

        return tables

    def _extract_header_fields(self, text: str, document_type: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header: Dict[str, Any] = {}

        for key, value in self._candidate_header_pairs(lines):
            canonical_key = self._normalise_header_key(key, document_type)
            if not canonical_key:
                continue
            header.setdefault(canonical_key, value.strip())

        if "supplier_name" not in header:
            supplier = self._infer_supplier(lines)
            if supplier:
                header["supplier_name"] = supplier

        self._enrich_currency(header, lines)
        self._standardise_header_values(header)
        return header

    def _candidate_header_pairs(self, lines: List[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip(" #\t-")
                value = value.strip()
                if key and value:
                    pairs.append((key, value))
                    continue

            dash_match = re.match(
                r"^(?P<key>[A-Za-z][A-Za-z0-9 .#/&-]{2,})\s+-\s+(?P<value>.+)$",
                line,
            )
            if dash_match:
                pairs.append((dash_match.group("key"), dash_match.group("value")))
                continue

            hash_match = re.match(
                r"^(?P<key>[A-Za-z][A-Za-z0-9 .&/-]{2,})\s*#\s*(?P<value>.+)$",
                line,
            )
            if hash_match:
                pairs.append((hash_match.group("key"), hash_match.group("value")))

        return pairs

    def _extract_line_items_from_text(
        self, text: str, document_type: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        lines = [line.rstrip() for line in text.splitlines()]
        collected_tables: List[Dict[str, Any]] = []

        for idx, line in enumerate(lines):
            if not self._looks_like_table_header(line):
                continue

            header_tokens = [
                self._normalise_line_header(token, document_type)
                for token in re.split(r"\s{2,}", line.strip())
                if token.strip()
            ]
            header_tokens = [token for token in header_tokens if token]
            if not header_tokens:
                continue

            rows: List[Dict[str, Any]] = []
            for row_line in lines[idx + 1 :]:
                if not row_line.strip():
                    if rows:
                        break
                    continue

                tokens = [
                    self._clean_cell(token)
                    for token in re.split(r"\s{2,}", row_line.strip())
                    if token.strip()
                ]
                if len(tokens) < 2:
                    if rows:
                        break
                    continue

                row: Dict[str, Any] = {}
                for column_index, header_token in enumerate(header_tokens):
                    value = tokens[column_index] if column_index < len(tokens) else ""
                    row[header_token] = value
                if any(str(value).strip() for value in row.values()):
                    rows.append(row)

            if rows:
                collected_tables.append({"headers": header_tokens, "rows": rows})

        if not collected_tables:
            return [], []

        primary_rows = [dict(row) for row in collected_tables[0]["rows"]]
        return primary_rows, collected_tables

    @staticmethod
    def _looks_like_table_header(line: str) -> bool:
        lowered = line.lower()
        return (
            "description" in lowered
            and (
                "qty" in lowered
                or "quantity" in lowered
                or "amount" in lowered
                or "total" in lowered
            )
            and re.search(r"\s{2,}", line) is not None
        )

    def _normalise_column_name(self, label: str) -> str:
        return self._normalise_line_header(label, "*")

    def _normalise_line_header(self, label: str, document_type: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", label).strip().lower()
        if not cleaned:
            return ""
        tokens = cleaned.split()

        def matches(pattern: Tuple[str, ...]) -> bool:
            for keyword in pattern:
                if not any(keyword in token for token in tokens):
                    return False
            return True

        for scope in (document_type, "*"):
            if not scope:
                continue
            keyword_map = LINE_KEYWORDS.get(scope, {})
            for canonical, patterns in keyword_map.items():
                for pattern in patterns:
                    if matches(pattern):
                        return canonical

        base_key = re.sub(r"[^a-z0-9]+", "_", cleaned)
        base_key = re.sub(r"_+", "_", base_key).strip("_")
        replacements = {
            "qty": "quantity",
            "amount": "line_amount",
            "total": "line_amount",
        }
        if base_key in replacements:
            return replacements[base_key]
        if base_key:
            return base_key
        return ""

    def _normalise_header_key(self, key: str, document_type: Optional[str] = None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", key).strip().lower()
        if not cleaned:
            return ""

        tokens = cleaned.split()
        token_set = set(tokens)
        document_type = document_type or ""

        def matches(pattern: Tuple[str, ...]) -> bool:
            for keyword in pattern:
                if not any(keyword in token for token in tokens):
                    return False
            return True

        for scope in (document_type, "*"):
            if not scope:
                continue
            keyword_map = HEADER_KEYWORDS.get(scope, {})
            for canonical, patterns in keyword_map.items():
                for pattern in patterns:
                    if matches(pattern):
                        return canonical

        if "invoice" in token_set and not ("date" in token_set or "total" in token_set or "amount" in token_set):
            return "invoice_id"
        if (
            document_type == "Quote"
            and ("quote" in token_set or "quotation" in token_set)
            and not ("date" in token_set or "total" in token_set or "amount" in token_set)
        ):
            return "quote_id"

        base_key = "_".join(token for token in tokens if token != "#")
        if base_key.endswith("_number"):
            if "invoice" in tokens:
                return "invoice_id"
            if "purchase" in tokens or "po" in tokens:
                return "po_id"
            if "contract" in tokens:
                return "contract_id"
            if "quote" in tokens or "quotation" in tokens:
                return "quote_id"
        if base_key and base_key not in {
            "item_description",
            "qty",
            "unit_price",
            "line_total",
        }:
            return base_key
        return ""

    def _infer_supplier(self, lines: List[str]) -> Optional[str]:
        supplier_pattern = re.compile(r"\b(?:supplier|vendor|seller)\b[:\s]+(.+)", re.IGNORECASE)
        for line in lines:
            match = supplier_pattern.search(line)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    def _enrich_currency(self, header: Dict[str, Any], lines: List[str]) -> None:
        if header.get("currency"):
            header["currency"] = header["currency"].strip().upper()
            return

        currency_codes = {"USD", "EUR", "GBP", "AUD", "CAD", "INR", "SGD"}
        joined = " ".join(lines)
        code_match = re.search(r"\b([A-Z]{3})\b", joined)
        if code_match and code_match.group(1) in currency_codes:
            header["currency"] = code_match.group(1)
            return

        symbol_map = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
        for symbol, code in symbol_map.items():
            if symbol in joined:
                header["currency"] = code
                return

    def _standardise_header_values(self, header: Dict[str, Any]) -> None:
        for key, value in list(header.items()):
            cleaned = self._clean_cell(value)
            if key.endswith("_id"):
                header[key] = cleaned.upper()
            elif key in {"currency"}:
                header[key] = cleaned.upper()
            else:
                header[key] = cleaned

    @staticmethod
    def _normalise_document_type(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalised = re.sub(r"[^a-z]", "", str(value).lower())
        mapping = {
            "invoice": "Invoice",
            "taxinvoice": "Invoice",
            "purchaseorder": "Purchase_Order",
            "purchaseord": "Purchase_Order",
            "po": "Purchase_Order",
            "contract": "Contract",
            "agreement": "Contract",
            "quote": "Quote",
            "quotation": "Quote",
        }
        return mapping.get(normalised)

    def _merge_header_fields(
        self,
        base_header: Dict[str, Any],
        llm_header: Any,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        merged = dict(base_header)
        if not isinstance(llm_header, dict):
            return merged

        for key, value in llm_header.items():
            normalized_key = self._normalise_header_key(str(key), document_type)
            if not normalized_key:
                continue
            cleaned_value = self._clean_cell(value)
            if not cleaned_value:
                continue
            if normalized_key not in merged or not self._clean_cell(merged[normalized_key]):
                merged[normalized_key] = cleaned_value
        return merged

    def _normalise_llm_line_items(
        self, items: Any, document_type: str
    ) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        normalised: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                row: Dict[str, Any] = {}
                for key, value in item.items():
                    normalized_key = self._normalise_line_header(str(key), document_type)
                    if not normalized_key:
                        continue
                    row[normalized_key] = self._clean_cell(value)
                if row:
                    normalised.append(row)
            elif isinstance(item, list):
                continue
        return normalised

    def _merge_line_items(
        self,
        base_items: List[Dict[str, Any]],
        llm_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not base_items:
            return [dict(item) for item in llm_items]
        if not llm_items:
            return base_items

        merged = [dict(item) for item in base_items]
        seen = {
            tuple(sorted((key, str(value)) for key, value in item.items()))
            for item in merged
        }
        for item in llm_items:
            signature = tuple(sorted((key, str(value)) for key, value in item.items()))
            if signature in seen:
                continue
            merged.append(dict(item))
            seen.add(signature)
        return merged

    def _normalise_llm_tables(
        self, tables: Any, document_type: str
    ) -> List[Dict[str, Any]]:
        if tables is None:
            return []
        if isinstance(tables, dict):
            tables = [tables]
        if not isinstance(tables, list):
            return []

        normalised: List[Dict[str, Any]] = []
        for table in tables:
            if not isinstance(table, dict):
                continue

            headers_raw = table.get("headers") or []
            headers: List[str] = []
            if isinstance(headers_raw, list):
                for header in headers_raw:
                    normalized_header = self._normalise_line_header(
                        str(header), document_type
                    )
                    if normalized_header and normalized_header not in headers:
                        headers.append(normalized_header)

            rows: List[Dict[str, Any]] = []
            raw_rows = table.get("rows") or []
            if isinstance(raw_rows, list) and raw_rows:
                if all(isinstance(row, dict) for row in raw_rows):
                    for row in raw_rows:
                        normalised_row: Dict[str, Any] = {}
                        for key, value in row.items():
                            normalized_key = self._normalise_line_header(
                                str(key), document_type
                            )
                            if not normalized_key:
                                continue
                            normalised_row[normalized_key] = self._clean_cell(value)
                        if normalised_row:
                            rows.append(normalised_row)
                elif headers and all(isinstance(row, (list, tuple)) for row in raw_rows):
                    for row in raw_rows:
                        normalised_row = {}
                        for idx, value in enumerate(row):
                            if idx >= len(headers):
                                break
                            normalised_row[headers[idx]] = self._clean_cell(value)
                        if normalised_row:
                            rows.append(normalised_row)

            if rows and not headers:
                headers = sorted({key for row in rows for key in row.keys()})

            if rows:
                normalised.append({"headers": headers, "rows": rows})

        return normalised

    @staticmethod
    def _clean_cell(cell: Any) -> str:
        if cell is None:
            return ""
        return str(cell).strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_result(self, result: ExtractionResult) -> None:
        table_name = RAW_TABLE_MAPPING[result.document_type]
        self._ensure_table_exists(table_name)

        payload = result.to_json()
        record = {
            "document_id": payload["document_id"],
            "document_type": payload["document_type"],
            "source_file": payload["source_file"],
            "extracted_at": datetime.utcnow().isoformat(timespec="seconds"),
            "header_json": json.dumps(payload["header"], ensure_ascii=False),
            "line_items_json": json.dumps(payload["line_items"], ensure_ascii=False),
            "tables_json": json.dumps(payload["tables"], ensure_ascii=False),
            "raw_text": payload["raw_text"],
            "metadata_json": json.dumps(payload["metadata"], ensure_ascii=False),
            "schema_reference_json": json.dumps(
                payload["schema_reference"], ensure_ascii=False
            ),
        }

        columns = list(record.keys())

        with closing(self._connection_factory()) as conn:
            cursor = conn.cursor()
            placeholder = self._placeholder(cursor)
            col_list = ", ".join(columns)
            placeholders = ", ".join([placeholder] * len(columns))
            quoted_table = self._quote_identifier(table_name)
            update_clause = ", ".join(
                f"{col}=excluded.{col}" for col in columns if col != "document_id"
            )
            sql = (
                f"INSERT INTO {quoted_table} ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id) DO UPDATE SET {update_clause}"
            )

            params: Iterable[Any]
            if placeholder == "%s":
                params = tuple(record[col] for col in columns)
            else:
                params = [record[col] for col in columns]

            cursor.execute(sql, params)
            conn.commit()

    def _ensure_table_exists(self, table_name: str) -> None:
        if table_name in self._ensured_tables:
            return

        create_sql = (
            f"CREATE TABLE IF NOT EXISTS {self._quote_identifier(table_name)} ("
            "document_id TEXT PRIMARY KEY,"
            "document_type TEXT NOT NULL,"
            "source_file TEXT,"
            "extracted_at TEXT,"
            "header_json TEXT,"
            "line_items_json TEXT,"
            "tables_json TEXT,"
            "raw_text TEXT,"
            "metadata_json TEXT,"
            "schema_reference_json TEXT"
            ")"
        )

        with closing(self._connection_factory()) as conn:
            cursor = conn.cursor()
            cursor.execute(create_sql)
            conn.commit()

        self._ensured_tables.add(table_name)

    @staticmethod
    def _placeholder(cursor: Any) -> str:
        module_name = getattr(type(cursor), "__module__", "")
        if module_name.startswith("psycopg"):
            return "%s"
        return "?"

    def _quote_identifier(self, identifier: str) -> str:
        if self._db_dialect == "sqlite":
            escaped = identifier.replace('"', '""')
            return f'"{escaped}"'

        parts = identifier.split(".")
        quoted_parts = []
        for part in parts:
            clean = part.strip('"').replace('"', '""')
            quoted_parts.append(f'"{clean}"')
        return ".".join(quoted_parts)


__all__ = ["DocumentExtractor", "ExtractionResult"]

