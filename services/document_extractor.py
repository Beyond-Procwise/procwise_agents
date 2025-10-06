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


class ProcurementSchemaReference:
    """Lightweight parser for ``docs/procurement_table_reference.md``."""

    _DOCUMENT_TABLE_MAP = {
        "Invoice": (
            "proc.invoice_agent",
            "proc.invoice_line_items_agent",
        ),
        "Purchase_Order": (
            "proc.purchase_order_agent",
            "proc.po_line_items_agent",
        ),
        "Contract": ("proc.contracts", None),
        "Quote": (
            "proc.quote_agent",
            "proc.quote_line_items_agent",
        ),
    }

    def __init__(self, reference_path: Optional[Path]) -> None:
        default_path = (
            reference_path
            if reference_path is not None
            else Path(__file__).resolve().parents[1]
            / "docs"
            / "procurement_table_reference.md"
        )
        self._path = default_path
        self._table_columns: Dict[str, List[str]] = {}
        if self._path.exists():
            try:
                content = self._path.read_text(encoding="utf-8")
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Unable to read procurement schema reference", exc_info=True)
            else:
                self._table_columns = self._parse_reference(content)

    @staticmethod
    def _parse_reference(content: str) -> Dict[str, List[str]]:
        table_columns: Dict[str, List[str]] = {}
        current_table: Optional[str] = None
        inside_code_block = False

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("### `") and line.endswith("`"):
                current_table = line.split("`")[1]
                table_columns.setdefault(current_table, [])
                continue

            if line.startswith("```"):
                inside_code_block = not inside_code_block
                continue

            if not inside_code_block or not current_table:
                continue

            if not line or line.startswith("--"):
                continue
            if line.upper().startswith("PRIMARY ") or line.upper().startswith("CONSTRAINT"):
                continue

            column_name = line.split()[0].strip("\",")
            if not column_name:
                continue

            if column_name not in table_columns[current_table]:
                table_columns[current_table].append(column_name)

        return table_columns

    def columns_for(self, table_name: Optional[str]) -> List[str]:
        if not table_name:
            return []
        return list(self._table_columns.get(table_name, []))

    def schema_for_document(self, document_type: str) -> Dict[str, Any]:
        header_table, line_table = self._DOCUMENT_TABLE_MAP.get(document_type, (None, None))
        schema: Dict[str, Any] = {}
        if header_table:
            schema["document_table"] = {
                "name": header_table,
                "columns": self.columns_for(header_table),
            }
        if line_table:
            schema["line_table"] = {
                "name": line_table,
                "columns": self.columns_for(line_table),
            }
        raw_table = RAW_TABLE_MAPPING.get(document_type)
        if raw_table:
            schema["raw_table"] = {"name": raw_table, "columns": []}
        return schema


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
        schema_hint: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.available():
            return None
        if not text.strip():
            return None

        header_columns = []
        line_columns = []
        if schema_hint:
            header_columns = schema_hint.get("document_table", {}).get("columns", [])
            line_columns = schema_hint.get("line_table", {}).get("columns", [])

        prompt = textwrap.dedent(
            f"""
            You are a specialised procurement data extractor.
            Determine the precise header fields and line items for a {document_type} document.
            Use the provided procurement schemas when naming fields. Header columns: {header_columns}.
            Line item columns: {line_columns}. If a column is absent from the text, omit it.
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
    schema_reference: Dict[str, str]

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
        self._schema_reference = ProcurementSchemaReference(reference_path)
        self._llm = llm_client or _LocalModelExtractor(
            preferred_models=preferred_models,
            chat_options=chat_options,
        )

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
            raise ValueError(
                f"Unsupported or undetected document type '{detected_type}'"
            )

        header = self._extract_header_fields(text, detected_type)
        line_items, derived_tables = self._extract_line_items_from_text(text)
        schema_hint = self._schema_reference.schema_for_document(detected_type)

        llm_payload = self._invoke_local_model(text, detected_type, schema_hint)
        if llm_payload:
            llm_type = self._normalise_document_type(
                llm_payload.get("document_type") if isinstance(llm_payload, dict) else None
            )
            if llm_type and llm_type in RAW_TABLE_MAPPING:
                detected_type = llm_type
                schema_hint = self._schema_reference.schema_for_document(detected_type)

            header = self._merge_header_fields(
                header, llm_payload.get("header"), detected_type
            )
            llm_line_items = self._normalise_llm_line_items(llm_payload.get("line_items"))
            line_items = self._merge_line_items(line_items, llm_line_items)
            derived_tables.extend(self._normalise_llm_tables(llm_payload.get("tables")))

        tables = self._normalise_tables(detected_tables, line_items, derived_tables)
        if not line_items and tables:
            line_items = [dict(row) for row in tables[0]["rows"]]

        schema_reference = self._schema_reference.schema_for_document(detected_type)

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
        schema_hint: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_llm", None):
            return None
        extract_fn = getattr(self._llm, "extract", None)
        if not callable(extract_fn):
            return None
        try:
            return extract_fn(text, document_type, schema_hint=schema_hint)
        except Exception:
            logger.warning("Local model invocation encountered an error", exc_info=True)
            return None

    def _normalise_tables(
        self,
        raw_tables: List[List[List[str]]],
        line_items: List[Dict[str, Any]],
        derived_tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = [
            {"headers": table["headers"], "rows": table["rows"]}
            for table in derived_tables
        ]

        for raw in raw_tables:
            if not raw:
                continue
            header_row = [self._normalise_column_name(self._clean_cell(cell)) for cell in raw[0]]
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

        return pairs

    def _extract_line_items_from_text(
        self, text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        lines = [line.rstrip() for line in text.splitlines()]
        collected_tables: List[Dict[str, Any]] = []

        for idx, line in enumerate(lines):
            if not self._looks_like_table_header(line):
                continue

            header_tokens = [
                self._normalise_column_name(token)
                for token in re.split(r"\s{2,}", line.strip())
                if token.strip()
            ]
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
            and ("qty" in lowered or "quantity" in lowered)
            and re.search(r"\s{2,}", line) is not None
        )

    @staticmethod
    def _normalise_column_name(label: str) -> str:
        label = label.strip().lower()
        label = label.replace("#", "number")
        label = re.sub(r"[^a-z0-9]+", "_", label)
        label = re.sub(r"_+", "_", label).strip("_")
        replacements = {
            "item": "item_description",
            "line_description": "item_description",
            "item_description": "item_description",
            "description": "item_description",
        }
        return replacements.get(label, label)

    def _normalise_header_key(self, key: str, document_type: Optional[str] = None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", key).strip().lower()
        if not cleaned:
            return ""

        tokens = cleaned.split()
        token_set = set(tokens)

        def has(*keywords: str) -> bool:
            return all(any(keyword == token or keyword in token for token in token_set) for keyword in keywords)

        document_type = document_type or ""

        if has("invoice") and (has("number") or has("no") or "#" in key):
            return "invoice_id"
        if has("invoice") and has("date"):
            return "invoice_date"
        if has("due") and has("date"):
            return "due_date"
        if (has("purchase") and has("order") and (has("number") or has("no"))) or (
            "po" in token_set and (has("number") or has("no"))
        ):
            return "po_id"
        if has("order") and has("date"):
            return "order_date"
        if has("contract") and (has("number") or has("no")):
            return "contract_id"
        if has("contract") and has("title"):
            return "contract_title"
        if document_type == "Contract" and has("start") and has("date"):
            return "contract_start_date"
        if document_type == "Contract" and has("end") and has("date"):
            return "contract_end_date"
        if has("payment") and has("terms"):
            return "payment_terms"
        if has("quotation") or has("quote"):
            if has("date"):
                return "quote_date"
            if (has("total") and has("amount")) or has("grand"):
                return "total_amount"
            if has("number") or has("no") or "#" in key or len(tokens) == 1:
                return "quote_id"
        if has("supplier") or has("vendor") or has("seller"):
            return "supplier_name"
        if has("total") and (has("amount") or has("invoice") or has("grand") or has("due")):
            return "total_amount"

        base_key = "_".join(token for token in tokens if token not in {"#"})
        if base_key.endswith("_number"):
            if "invoice" in token_set:
                return "invoice_id"
            if "purchase" in token_set or "po" in token_set:
                return "po_id"
            if "contract" in token_set:
                return "contract_id"
            if "quote" in token_set or "quotation" in token_set:
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

    def _normalise_llm_line_items(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        normalised: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                row: Dict[str, Any] = {}
                for key, value in item.items():
                    normalized_key = self._normalise_column_name(str(key))
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

    def _normalise_llm_tables(self, tables: Any) -> List[Dict[str, Any]]:
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
                    normalized_header = self._normalise_column_name(str(header))
                    if normalized_header and normalized_header not in headers:
                        headers.append(normalized_header)

            rows: List[Dict[str, Any]] = []
            raw_rows = table.get("rows") or []
            if isinstance(raw_rows, list) and raw_rows:
                if all(isinstance(row, dict) for row in raw_rows):
                    for row in raw_rows:
                        normalised_row: Dict[str, Any] = {}
                        for key, value in row.items():
                            normalized_key = self._normalise_column_name(str(key))
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

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return f'"{identifier}"'


__all__ = ["DocumentExtractor", "ExtractionResult"]

