from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Pydantic is required for the validation loop.
    from pydantic import BaseModel, ValidationError, create_model
except Exception as exc:  # pragma: no cover - import error is surfaced eagerly
    raise RuntimeError("Pydantic is required for the document jsonifier") from exc

from utils.procurement_schema import DOC_TYPE_TO_TABLE, PROCUREMENT_SCHEMAS, TableSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout primitives shared between the extractor and the jsonifier.
# ---------------------------------------------------------------------------


@dataclass
class LayoutElement:
    """Represents a semantic chunk of the source document."""

    type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutTable:
    """Tabular content preserved during layout-aware parsing."""

    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredDocument:
    """Layout-aware representation fed into the LLM prompting layer."""

    elements: List[LayoutElement]
    tables: List[LayoutTable]
    markdown: str
    raw_text: str


@dataclass
class JsonifiedDocument:
    """Normalised document output produced after validation."""

    document_type: str
    header: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    schema_reference: Dict[str, Any]
    prompt: str
    llm_response: str


@dataclass
class SchemaBundle:
    document_type: str
    header_table: str
    line_table: Optional[str]
    header_schema: TableSchema
    line_schema: Optional[TableSchema]


def _tokenise(value: str) -> List[str]:
    token: List[str] = []
    tokens: List[str] = []
    for char in value:
        if char.isalnum():
            token.append(char.lower())
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _normalise(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(_tokenise(a))
    tokens_b = set(_tokenise(b))
    overlap = 0.0
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    ratio = SequenceMatcher(None, a, b).ratio()
    return max(overlap, ratio)


def _schema_terms(schema: TableSchema) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for column in schema.columns:
        candidates = {_normalise(column)}
        spaced = column.replace("_", " ")
        candidates.add(_normalise(spaced))
        for synonym in schema.synonyms.get(column, []):
            candidates.add(_normalise(synonym))
        lookup[column] = [term for term in candidates if term]
    return lookup


class LLMClient:
    """Interface for large language model integrations."""

    def generate(
        self,
        prompt: str,
        *,
        structured_document: StructuredDocument,
        schema: SchemaBundle,
        previous: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:  # pragma: no cover - interface method
        raise NotImplementedError


class LocalExtractionLLM(LLMClient):
    """Deterministic LLM surrogate used in the test-suite."""

    def __init__(self, *, minimum_match: float = 0.55) -> None:
        self._minimum_match = minimum_match

    def generate(
        self,
        prompt: str,
        *,
        structured_document: StructuredDocument,
        schema: SchemaBundle,
        previous: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        header = self._extract_header(structured_document, schema.header_schema)
        line_items = self._extract_line_items(structured_document, schema.line_schema)
        payload = {
            "document_type": schema.document_type,
            "header_data": header,
            "line_items": line_items,
            "tables": self._format_tables(structured_document.tables),
        }
        if schema.line_table and not payload["line_items"]:
            payload["line_items"] = []
        for required in schema.header_schema.required:
            payload["header_data"].setdefault(required, None)
        return json.dumps(payload, ensure_ascii=False)

    def _extract_header(
        self,
        document: StructuredDocument,
        schema: TableSchema,
    ) -> Dict[str, Any]:
        lookup = _schema_terms(schema)
        used: set[str] = set()
        header: Dict[str, Any] = {}
        for element in document.elements:
            if element.type != "key_value":
                continue
            key = element.metadata.get("key") or element.text
            value = element.metadata.get("value") or ""
            column, score = self._best_match(key, lookup, used)
            if column and score >= self._minimum_match:
                header[column] = value.strip()
                used.add(column)
        return header

    def _extract_line_items(
        self,
        document: StructuredDocument,
        schema: Optional[TableSchema],
    ) -> List[Dict[str, Any]]:
        if not schema:
            return []
        lookup = _schema_terms(schema)
        results: List[Dict[str, Any]] = []
        for table in document.tables:
            header_map: Dict[int, str] = {}
            for idx, label in enumerate(table.headers):
                column, score = self._best_match(label, lookup)
                if column and score >= self._minimum_match:
                    header_map[idx] = column
            if len(header_map) < 2:
                continue
            for row in table.rows:
                row_payload: Dict[str, Any] = {}
                for idx, value in enumerate(row):
                    column = header_map.get(idx)
                    if column:
                        row_payload[column] = value.strip()
                if row_payload:
                    results.append(row_payload)
        return results

    def _format_tables(self, tables: Sequence[LayoutTable]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for table in tables:
            formatted.append(
                {
                    "headers": list(table.headers),
                    "rows": [list(row) for row in table.rows],
                }
            )
        return formatted

    def _best_match(
        self,
        label: str,
        lookup: Dict[str, List[str]],
        used: Optional[set[str]] = None,
    ) -> Tuple[Optional[str], float]:
        best_column: Optional[str] = None
        best_score = 0.0
        target = _normalise(label)
        if not target:
            return None, 0.0
        for column, terms in lookup.items():
            if used and column in used:
                continue
            for term in terms:
                score = _similarity(target, term)
                if score > best_score:
                    best_column = column
                    best_score = score
        return best_column, best_score


class DocumentJsonifier:
    """LLM-driven conversion of structured documents into schema-aligned JSON."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm_client or LocalExtractionLLM()
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def jsonify(
        self,
        document: StructuredDocument,
        *,
        document_type_hint: Optional[str] = None,
    ) -> JsonifiedDocument:
        schema = self._resolve_schema(document, document_type_hint)
        prompt = ""
        previous_json: Optional[str] = None
        error_message: Optional[str] = None

        for attempt in range(self._max_retries):
            prompt = self._build_prompt(document, schema, previous_json, error_message)
            raw_response = self._llm.generate(
                prompt,
                structured_document=document,
                schema=schema,
                previous=previous_json,
                error=error_message,
            )

            parsed = self._parse_response(raw_response)
            if parsed is None:
                error_message = "Response was not valid JSON"
                previous_json = raw_response
                continue

            try:
                validated = self._validate_payload(parsed, schema)
            except ValidationError as exc:
                logger.debug("Validation failed for attempt %s", attempt + 1, exc_info=True)
                error_message = exc.json()
                previous_json = json.dumps(parsed, ensure_ascii=False)
                continue

            tables = self._format_tables(document.tables)
            schema_reference = self._build_schema_reference(schema)
            return JsonifiedDocument(
                document_type=schema.document_type,
                header=validated["header_data"],
                line_items=validated["line_items"],
                tables=tables,
                schema_reference=schema_reference,
                prompt=prompt,
                llm_response=json.dumps(parsed, ensure_ascii=False),
            )

        raise ValueError("LLM failed to provide valid JSON after correction loop")

    # ------------------------------------------------------------------
    # Prompt construction & schema handling
    # ------------------------------------------------------------------
    def _resolve_schema(
        self,
        document: StructuredDocument,
        document_type_hint: Optional[str],
    ) -> SchemaBundle:
        doc_type = self._infer_document_type(document, document_type_hint)
        header_table, line_table = DOC_TYPE_TO_TABLE.get(doc_type, (None, None))
        if header_table is None:
            raise ValueError(f"Unsupported document type: {doc_type}")
        header_schema = PROCUREMENT_SCHEMAS[header_table]
        line_schema = PROCUREMENT_SCHEMAS.get(line_table) if line_table else None
        return SchemaBundle(
            document_type=doc_type,
            header_table=header_table,
            line_table=line_table,
            header_schema=header_schema,
            line_schema=line_schema,
        )

    def _infer_document_type(
        self,
        document: StructuredDocument,
        hint: Optional[str],
    ) -> str:
        if hint and hint in DOC_TYPE_TO_TABLE:
            return hint

        keyword_map: Dict[str, Sequence[Sequence[str]]] = {
            "Invoice": (
                ("invoice",),
                ("invoice", "number"),
                ("amount", "due"),
                ("total", "incl", "tax"),
            ),
            "Purchase_Order": (
                ("purchase", "order"),
                ("po", "number"),
                ("order", "date"),
            ),
            "Quote": (
                ("quote",),
                ("quotation",),
                ("valid", "until"),
            ),
            "Contract": (
                ("contract",),
                ("agreement",),
            ),
        }

        text = document.raw_text.lower()
        best_type = "Contract"
        best_score = 0
        for doc_type, keyword_groups in keyword_map.items():
            score = 0
            for group in keyword_groups:
                if all(keyword in text for keyword in group):
                    score += len(group)
            if score > best_score:
                best_type = doc_type
                best_score = score

        return best_type

    def _build_prompt(
        self,
        document: StructuredDocument,
        schema: SchemaBundle,
        previous: Optional[str],
        error: Optional[str],
    ) -> str:
        schema_payload = {
            "document_type": schema.document_type,
            "header_table": schema.header_table,
            "line_table": schema.line_table,
            "header_fields": schema.header_schema.columns,
            "line_item_fields": schema.line_schema.columns if schema.line_schema else [],
        }

        lines = [
            "You are an expert data extraction agent.",
            "Your task is to extract information from the document text and return a JSON object matching the schema.",
            "If a field is missing, emit a null value.",
            "",
            "[Document Text - Layout Preserved]",
            document.markdown,
            "[End of Document Text]",
            "",
            "[Target JSON Schema]",
            json.dumps(schema_payload, ensure_ascii=False, indent=2),
        ]

        if previous and error:
            lines.extend(
                [
                    "",
                    "The previous JSON failed validation with this error:",
                    error,
                    "",
                    "Previous JSON:",
                    previous,
                ]
            )

        lines.append("")
        lines.append("[Extracted JSON]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(response)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        schema: SchemaBundle,
    ) -> Dict[str, Any]:
        header_data = payload.get("header_data") if isinstance(payload.get("header_data"), dict) else {}
        line_items_payload = payload.get("line_items")
        if not isinstance(line_items_payload, list):
            line_items_payload = []

        header_model = self._build_header_model(schema.header_schema)
        header_instance = header_model(**self._normalise_dict(header_data))
        validated_header = self._model_dump(header_instance)

        validated_lines: List[Dict[str, Any]] = []
        if schema.line_schema and line_items_payload:
            line_model = self._build_line_model(schema.line_schema)
            for item in line_items_payload:
                if not isinstance(item, dict):
                    continue
                instance = line_model(**self._normalise_dict(item))
                validated_lines.append(self._model_dump(instance))

        validated_lines = self._harmonise_line_items(schema, validated_lines)

        return {
            "header_data": validated_header,
            "line_items": validated_lines,
        }

    def _normalise_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        normalised: Dict[str, Any] = {}
        for key, value in data.items():
            normalised[key] = self._normalise_value(value)
        return normalised

    def _normalise_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(value)
        return str(value).strip()

    def _build_header_model(self, schema: TableSchema) -> type[BaseModel]:
        fields: Dict[str, Tuple[type, Any]] = {}
        for column in schema.columns:
            annotation = Optional[str]
            default: Any = None
            if column in schema.required:
                default = ...
            fields[column] = (annotation, default)
        return create_model("HeaderModel", **fields)  # type: ignore[arg-type]

    def _build_line_model(self, schema: TableSchema) -> type[BaseModel]:
        fields: Dict[str, Tuple[type, Any]] = {}
        for column in schema.columns:
            fields[column] = (Optional[str], None)
        return create_model("LineItemModel", **fields)  # type: ignore[arg-type]

    def _model_dump(self, model: BaseModel) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _harmonise_line_items(
        self,
        schema: SchemaBundle,
        line_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not schema.line_schema or not line_items:
            return line_items

        columns = set(schema.line_schema.columns)
        harmonised: List[Dict[str, Any]] = []
        for item in line_items:
            normalised = dict(item)
            total_amount = normalised.get("total_amount")

            if schema.document_type == "Invoice" and "line_amount" in columns:
                if not normalised.get("line_amount") and total_amount:
                    normalised["line_amount"] = total_amount

            if schema.document_type == "Purchase_Order" and "line_total" in columns:
                if not normalised.get("line_total") and total_amount:
                    normalised["line_total"] = total_amount

            if schema.document_type == "Quote":
                if "line_total" in columns and not normalised.get("line_total") and total_amount:
                    normalised["line_total"] = total_amount
                if "line_amount" in columns and not normalised.get("line_amount"):
                    line_total = normalised.get("line_total") or total_amount
                    if line_total:
                        normalised["line_amount"] = line_total

            harmonised.append(normalised)

        return harmonised

    def _format_tables(self, tables: Sequence[LayoutTable]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for table in tables:
            formatted.append(
                {
                    "headers": list(table.headers),
                    "rows": [list(row) for row in table.rows],
                }
            )
        return formatted

    def _build_schema_reference(self, schema: SchemaBundle) -> Dict[str, Any]:
        reference = {
            "document_type": schema.document_type,
            "header_table": schema.header_table,
            "header_fields": list(schema.header_schema.columns),
        }
        if schema.line_schema:
            reference["line_table"] = schema.line_table
            reference["line_item_fields"] = list(schema.line_schema.columns)
        else:
            reference["line_table"] = None
            reference["line_item_fields"] = []
        return reference


__all__ = [
    "DocumentJsonifier",
    "JsonifiedDocument",
    "LayoutElement",
    "LayoutTable",
    "StructuredDocument",
    "LLMClient",
]

