"""Layout-aware document extraction pipeline for procurement artefacts."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:  # Optional dependency used for PDF parsing.
    import fitz  # type: ignore
except Exception:  # pragma: no cover - PyMuPDF may not be installed in some tests
    fitz = None  # type: ignore

try:  # Optional dependency that acts as a secondary PDF fallback.
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

from agents.document_jsonifier import (
    DocumentJsonifier,
    LayoutElement,
    LayoutTable,
    StructuredDocument,
)
from services.ocr_pipeline import OCRPreprocessor
from utils.procurement_schema import DOC_TYPE_TO_TABLE

logger = logging.getLogger(__name__)


RAW_TABLE_MAPPING = {
    "Invoice": "proc.raw_invoice",
    "Purchase_Order": "proc.raw_purchase_order",
    "Contract": "proc.raw_contracts",
    "Quote": "proc.raw_quotes",
}


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


class LayoutAwareParser:
    """Convert raw text into a layout-preserving structure."""

    def from_text(self, text: str, *, scanned: bool) -> StructuredDocument:
        lines = text.splitlines()
        elements: List[LayoutElement] = []
        tables: List[LayoutTable] = []
        buffer: List[str] = []

        for line in lines:
            if self._is_table_line(line):
                buffer.append(line)
                continue
            if buffer:
                tables.append(self._flush_table(buffer))
                buffer = []

            cleaned = line.strip()
            if not cleaned:
                continue
            if self._looks_like_key_value(cleaned):
                key, value = self._split_key_value(cleaned)
                elements.append(
                    LayoutElement(
                        type="key_value",
                        text=cleaned,
                        metadata={"key": key, "value": value},
                    )
                )
            elif self._looks_like_title(cleaned):
                elements.append(LayoutElement(type="title", text=cleaned))
            elif self._looks_like_heading(cleaned):
                elements.append(LayoutElement(type="heading", text=cleaned))
            else:
                elements.append(LayoutElement(type="paragraph", text=cleaned))

        if buffer:
            tables.append(self._flush_table(buffer))

        markdown = self._to_markdown(elements, tables)
        return StructuredDocument(
            elements=elements,
            tables=tables,
            markdown=markdown,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Heuristic detectors
    # ------------------------------------------------------------------
    def _is_table_line(self, line: str) -> bool:
        stripped = line.rstrip()
        if not stripped:
            return False
        if "\t" in stripped:
            return True
        return "  " in stripped

    def _looks_like_title(self, line: str) -> bool:
        return len(line.split()) <= 6 and line.isupper()

    def _looks_like_heading(self, line: str) -> bool:
        words = line.split()
        if len(words) > 1 and all(word[0].isupper() for word in words if word):
            return True
        return False

    def _looks_like_key_value(self, line: str) -> bool:
        return self._split_key_value(line) is not None

    def _split_key_value(self, line: str) -> Optional[tuple[str, str]]:
        separators = [":", "-", "–", "—"]
        for separator in separators:
            if separator in line:
                key, value = line.split(separator, 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    return key, value
        parts = [part for part in line.split() if part]
        if len(parts) >= 2:
            key = parts[0]
            value = " ".join(parts[1:])
            return key, value
        return None

    def _flush_table(self, buffer: List[str]) -> LayoutTable:
        rows = [self._split_table_columns(row) for row in buffer if row.strip()]
        if not rows:
            return LayoutTable(headers=[], rows=[])
        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        return LayoutTable(headers=headers, rows=data_rows)

    def _split_table_columns(self, line: str) -> List[str]:
        columns: List[str] = []
        current: List[str] = []
        space_run = 0
        for char in line.rstrip():
            if char == " ":
                space_run += 1
                if space_run >= 2:
                    if current:
                        columns.append("".join(current).strip())
                        current = []
                    continue
            else:
                if space_run >= 2:
                    space_run = 0
                elif space_run == 1:
                    current.append(" ")
                    space_run = 0
                current.append(char)
        if current:
            columns.append("".join(current).strip())
        return [col for col in columns if col]

    def _to_markdown(
        self,
        elements: List[LayoutElement],
        tables: List[LayoutTable],
    ) -> str:
        lines: List[str] = []
        for element in elements:
            if element.type == "title":
                lines.append(f"# {element.text}")
            elif element.type == "heading":
                lines.append(f"## {element.text}")
            elif element.type == "key_value":
                key = element.metadata.get("key") or element.text
                value = element.metadata.get("value") or ""
                lines.append(f"- **{key}**: {value}")
            else:
                lines.append(element.text)

        for index, table in enumerate(tables, start=1):
            lines.append("")
            lines.append(f"Table {index}")
            if table.headers:
                header = " | ".join(table.headers)
                separator = " | ".join(["---"] * len(table.headers))
                lines.append(f"| {header} |")
                lines.append(f"| {separator} |")
                for row in table.rows:
                    padded = list(row)
                    if len(padded) < len(table.headers):
                        padded.extend([""] * (len(table.headers) - len(padded)))
                    lines.append(f"| {' | '.join(padded)} |")
        return "\n".join(lines).strip()


class DocumentExtractor:
    """Parse procurement documents into structured JSON and persist them."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        jsonifier: Optional[DocumentJsonifier] = None,
        ocr_preprocessor: Optional[OCRPreprocessor] = None,
        layout_parser: Optional[LayoutAwareParser] = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._jsonifier = jsonifier or DocumentJsonifier()
        self._ocr = ocr_preprocessor or OCRPreprocessor()
        self._layout_parser = layout_parser or LayoutAwareParser()
        self._ensured_tables: set[str] = set()
        self._db_dialect = "generic"

        try:
            with closing(self._connection_factory()) as conn:
                module_name = getattr(type(conn), "__module__", "").lower()
                if "sqlite" in module_name:
                    self._db_dialect = "sqlite"
                elif "psycopg" in module_name or "postgres" in module_name:
                    self._db_dialect = "postgres"
        except Exception:  # pragma: no cover - inference is best effort
            logger.debug("Unable to infer database dialect", exc_info=True)

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
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Document not found at {path}")

        metadata_payload = dict(metadata or {})
        ingestion_mode = str(metadata_payload.get("ingestion_mode", "")).lower()
        if ingestion_mode:
            metadata_payload["ingestion_mode"] = ingestion_mode

        scanned_hint = ingestion_mode in {"scanned", "image"}
        scanned_pdf = self._detect_scanned_pdf(path) if path.suffix.lower() == ".pdf" else False
        scanned = scanned_hint or scanned_pdf

        structured = self._load_document(path, scanned=scanned)
        metadata_payload.setdefault(
            "layout_preview",
            [{"type": element.type, "text": element.text} for element in structured.elements[:20]],
        )

        jsonified = self._jsonifier.jsonify(structured, document_type_hint=document_type)

        detected_type = jsonified.document_type
        if detected_type not in RAW_TABLE_MAPPING:
            raise ValueError(f"Unsupported document type: {detected_type}")

        document_id = self._generate_document_id(path, jsonified.header)
        metadata_payload.setdefault("llm_prompt_tokens", len(jsonified.prompt.split()))

        result = ExtractionResult(
            document_id=document_id,
            document_type=detected_type,
            source_file=source_label or path.name,
            header=jsonified.header,
            line_items=jsonified.line_items,
            tables=jsonified.tables,
            raw_text=structured.raw_text,
            metadata=metadata_payload,
            schema_reference=jsonified.schema_reference,
        )

        self._persist_result(result)
        return result

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------
    def _load_document(self, path: Path, *, scanned: bool) -> StructuredDocument:
        if path.suffix.lower() == ".pdf":
            ocr_result = self._ocr.extract(path, scanned=scanned)
            if ocr_result:
                text = self._ocr.preprocess_text(ocr_result.text, scanned=True)
                return self._layout_parser.from_text(text, scanned=True)
            text = self._extract_pdf_text(path)
            text = self._ocr.preprocess_text(text, scanned=scanned)
            return self._layout_parser.from_text(text, scanned=scanned)

        text = path.read_text(encoding="utf-8", errors="ignore")
        text = self._ocr.preprocess_text(text, scanned=scanned)
        return self._layout_parser.from_text(text, scanned=scanned)

    def _extract_pdf_text(self, path: Path) -> str:
        if fitz is not None:
            try:
                with fitz.open(path.as_posix()) as pdf:
                    pages = [page.get_text("text") for page in pdf]
                return "\n".join(filter(None, pages))
            except Exception:
                logger.debug("PyMuPDF failed to extract text", exc_info=True)

        if pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(filter(None, pages))
            except Exception:
                logger.debug("pdfplumber failed to extract text", exc_info=True)

        raise RuntimeError("Unable to parse PDF: no extraction backend available")

    def _detect_scanned_pdf(self, path: Path) -> bool:
        if fitz is None:
            return False
        try:
            with fitz.open(path.as_posix()) as pdf:
                for page in pdf:
                    text = page.get_text("text").strip()
                    if text:
                        return False
        except Exception:
            logger.debug("Failed to inspect PDF for scan detection", exc_info=True)
            return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_document_id(self, path: Path, header: Dict[str, Any]) -> str:
        for key in ("invoice_id", "po_id", "contract_id", "quote_id"):
            value = header.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        stem = "".join(char if char.isalnum() else "-" for char in path.stem).strip("-")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{stem or 'document'}-{timestamp}"

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
            "schema_reference_json": json.dumps(payload["schema_reference"], ensure_ascii=False),
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

