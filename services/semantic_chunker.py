"""Structure-aware semantic chunker for document ingestion."""
from __future__ import annotations

import csv
import hashlib
import logging
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    tiktoken = None  # type: ignore

from services.document_extractor import LayoutTable, StructuredDocument


logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """Normalized chunk of text paired with structured metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Segment:
    """Intermediate representation of document segments prior to packing."""

    text: str
    section_path: Tuple[str, ...]
    type: str
    metadata: Dict[str, Any]


class SemanticChunker:
    """Convert structured document output into overlapping semantic chunks."""

    def __init__(
        self,
        *,
        settings: Optional[Any] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.settings = settings
        self._tokenizer_name = tokenizer_name
        self._encoding = self._initialise_tokenizer(tokenizer_name)
        defaults = self._derive_defaults(min_tokens, max_tokens, overlap_ratio)
        self._min_tokens = defaults["min_tokens"]
        self._max_tokens = defaults["max_tokens"]
        self._overlap_tokens = max(0, int(math.ceil(self._max_tokens * defaults["overlap_ratio"])) )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_from_structured(
        self,
        structured: StructuredDocument,
        *,
        document_type: str,
        base_metadata: Optional[Dict[str, Any]] = None,
        title_hint: Optional[str] = None,
        default_section: str = "document_overview",
    ) -> List[SemanticChunk]:
        """Generate semantic chunks from a structured document."""

        if not structured.raw_text.strip():
            return []

        title = title_hint or self._detect_title(structured)
        boilerplate = self._detect_boilerplate_lines(structured.raw_text)
        base_metadata = dict(base_metadata or {})
        base_metadata.setdefault("document_type", document_type)
        base_metadata.setdefault("title", title or base_metadata.get("doc_name") or "")
        base_metadata.setdefault("source_type", self._derive_source_type(document_type))
        base_metadata.setdefault("section_path", default_section)

        segments = self._build_segments(structured, boilerplate, default_section)
        if not segments:
            return []

        return self._pack_segments(segments, base_metadata)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _derive_defaults(
        self,
        min_tokens: Optional[int],
        max_tokens: Optional[int],
        overlap_ratio: Optional[float],
    ) -> Dict[str, float]:
        settings = self.settings
        if settings is not None:
            if min_tokens is None:
                min_tokens = getattr(settings, "rag_chunk_min_tokens", None)
            if max_tokens is None:
                max_tokens = getattr(settings, "rag_chunk_max_tokens", None)
            if overlap_ratio is None:
                overlap_ratio = getattr(settings, "rag_chunk_overlap_ratio", None)
            if min_tokens is None or max_tokens is None:
                approx_chars = getattr(settings, "rag_chunk_chars", None)
                approx_overlap = getattr(settings, "rag_chunk_overlap", None)
                if min_tokens is None and isinstance(approx_chars, (int, float)):
                    min_tokens = max(250, int(float(approx_chars) / 4.5))
                if max_tokens is None and isinstance(approx_chars, (int, float)):
                    max_tokens = max(450, int(float(approx_chars) / 3.5))
                if overlap_ratio is None and isinstance(approx_overlap, (int, float)) and isinstance(approx_chars, (int, float)) and approx_chars:
                    overlap_ratio = max(0.05, min(0.25, float(approx_overlap) / float(approx_chars)))

        min_tokens = int(min_tokens or 360)
        min_tokens = max(300, min_tokens)

        max_tokens = int(max_tokens or 760)
        if max_tokens <= min_tokens:
            max_tokens = min_tokens + 120
        max_tokens = min(max_tokens, 800)
        if max_tokens - min_tokens < 60:
            max_tokens = min(800, max(min_tokens + 60, max_tokens))

        overlap_ratio = float(overlap_ratio or 0.1)
        overlap_ratio = max(0.05, min(overlap_ratio, 0.18))

        return {
            "min_tokens": int(min_tokens),
            "max_tokens": int(max_tokens),
            "overlap_ratio": overlap_ratio,
        }

    def _initialise_tokenizer(self, tokenizer_name: str):
        if tiktoken is None:
            return None
        try:
            return tiktoken.get_encoding(tokenizer_name)
        except Exception:
            try:
                return tiktoken.encoding_for_model(tokenizer_name)
            except Exception:
                logger.warning("Falling back to whitespace tokenisation for chunking", exc_info=True)
                return None

    # ------------------------------------------------------------------
    # Segment construction
    # ------------------------------------------------------------------
    def _build_segments(
        self,
        structured: StructuredDocument,
        boilerplate: set[str],
        default_section: str,
    ) -> List[_Segment]:
        segments: List[_Segment] = []
        section_path: List[str] = [default_section]

        def current_path() -> Tuple[str, ...]:
            return tuple(section_path)

        for element in structured.elements:
            raw_text = self._canonicalise(element.text)
            if not raw_text or raw_text in boilerplate:
                continue

            if element.type in {"title", "heading"}:
                normalised = self._normalise_section_name(raw_text)
                section_path = [default_section, normalised]
                meta = {"content_type": element.type, "section": normalised}
                segments.append(
                    _Segment(
                        text=raw_text,
                        section_path=current_path(),
                        type="heading",
                        metadata=meta,
                    )
                )
                continue

            if element.type == "key_value":
                key = element.metadata.get("key") or raw_text
                value = element.metadata.get("value", "")
                formatted = f"{key}: {value}".strip()
                if not formatted or formatted in boilerplate:
                    continue
                meta = {
                    "content_type": "key_value",
                    "field_key": key,
                    "field_value": value,
                    "section": section_path[-1] if section_path else default_section,
                }
                segments.append(
                    _Segment(
                        text=formatted,
                        section_path=current_path(),
                        type="key_value",
                        metadata=meta,
                    )
                )
                continue

            meta = {
                "content_type": element.type or "paragraph",
                "section": section_path[-1] if section_path else default_section,
            }
            segments.append(
                _Segment(
                    text=raw_text,
                    section_path=current_path(),
                    type="paragraph",
                    metadata=meta,
                )
            )

        for table in structured.tables:
            rendered, csv_payload = self._render_table(table)
            if not rendered.strip() or rendered in boilerplate:
                continue
            meta = {
                "content_type": "table",
                "section": section_path[-1] if section_path else default_section,
                "table_headers": table.headers,
                "table_row_count": len(table.rows),
                "table_csv": csv_payload,
            }
            meta.update(table.metadata or {})
            segments.append(
                _Segment(
                    text=rendered,
                    section_path=current_path(),
                    type="table",
                    metadata=meta,
                )
            )

        return segments

    def _render_table(self, table: LayoutTable) -> Tuple[str, str]:
        headers = [self._canonicalise(value) for value in table.headers]
        rows = [[self._canonicalise(cell) for cell in row] for row in table.rows]
        lines: List[str] = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append(" | ".join(["---"] * len(headers)))
        for row in rows:
            lines.append(" | ".join(row))
        rendered = "\n".join(line for line in lines if line.strip())

        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        if headers:
            writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
        csv_payload = csv_buffer.getvalue()
        return rendered, csv_payload

    # ------------------------------------------------------------------
    # Segment packing
    # ------------------------------------------------------------------
    def _pack_segments(
        self,
        segments: Sequence[_Segment],
        base_metadata: Dict[str, Any],
    ) -> List[SemanticChunk]:
        packed: List[SemanticChunk] = []
        token_buffer: List[int] = []
        segment_buffer: List[_Segment] = []
        seen_hashes: set[str] = set()

        queue: Deque[Tuple[List[int], _Segment]] = deque()
        for segment in segments:
            tokenised_parts = self._split_segment_tokens(segment)
            for tokens, seg_part in tokenised_parts:
                queue.append((tokens, seg_part))

        while queue:
            tokens, segment = queue.popleft()
            if not tokens:
                continue
            if not token_buffer and segment_buffer:
                segment_buffer.clear()

            token_buffer.extend(tokens)
            segment_buffer.append(segment)

            should_flush = False
            buffer_len = len(token_buffer)

            if buffer_len >= self._max_tokens:
                should_flush = True
            elif buffer_len >= self._min_tokens:
                next_tokens = len(queue[0][0]) if queue else 0
                if next_tokens and buffer_len + next_tokens > self._max_tokens:
                    should_flush = True
            if not queue and buffer_len > 0:
                should_flush = True

            if should_flush:
                chunk_text = self._normalise_chunk_text(
                    [seg.text for seg in segment_buffer if seg.text]
                )
                if not chunk_text:
                    token_buffer = []
                    segment_buffer = []
                    continue
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    if self._overlap_tokens > 0:
                        token_buffer = list(token_buffer[-self._overlap_tokens :])
                    else:
                        token_buffer = []
                    if token_buffer:
                        overlap_segment = self._build_overlap_segment(segment_buffer, token_buffer)
                        segment_buffer = [overlap_segment]
                    else:
                        segment_buffer = []
                    continue
                seen_hashes.add(chunk_hash)

                chunk_metadata = self._compose_metadata(segment_buffer, base_metadata)
                chunk_metadata.update(
                    {
                        "chunk_hash": chunk_hash,
                        "chunk_token_count": len(token_buffer),
                    }
                )
                packed.append(SemanticChunk(content=chunk_text, metadata=chunk_metadata))

                if self._overlap_tokens > 0:
                    token_buffer = list(token_buffer[-self._overlap_tokens :])
                else:
                    token_buffer = []
                if token_buffer:
                    overlap_segment = self._build_overlap_segment(segment_buffer, token_buffer)
                    segment_buffer = [overlap_segment]
                else:
                    segment_buffer = []

        return packed

    def _split_segment_tokens(self, segment: _Segment) -> List[Tuple[List[int], _Segment]]:
        tokens = self._encode(segment.text)
        if not tokens:
            return []
        if len(tokens) <= self._max_tokens:
            clone = _Segment(
                text=self._normalise_chunk_text([segment.text]),
                section_path=segment.section_path,
                type=segment.type,
                metadata=dict(segment.metadata),
            )
            return [(tokens, clone)]

        parts: List[Tuple[List[int], _Segment]] = []
        step = max(1, self._max_tokens - self._overlap_tokens)
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self._max_tokens)
            part_tokens = tokens[start:end]
            part_text = self._normalise_chunk_text([self._decode(part_tokens)])
            clone = _Segment(
                text=part_text,
                section_path=segment.section_path,
                type=segment.type,
                metadata=dict(segment.metadata),
            )
            parts.append((part_tokens, clone))
            if end == len(tokens):
                break
            start += step
        return parts

    def _build_overlap_segment(
        self, segments: Sequence[_Segment], tokens: Sequence[int]
    ) -> _Segment:
        reference = segments[-1] if segments else _Segment(
            text="", section_path=("document_overview",), type="overlap", metadata={}
        )
        text = self._normalise_chunk_text([self._decode(tokens)])
        return _Segment(
            text=text,
            section_path=reference.section_path,
            type="overlap",
            metadata=dict(reference.metadata),
        )

    def _compose_metadata(
        self,
        segments: Sequence[_Segment],
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = dict(base_metadata)
        if segments:
            section_paths = [segment.section_path for segment in segments if segment.section_path]
            if section_paths:
                section_path = section_paths[-1]
                metadata["section_path"] = " > ".join(filter(None, section_path))
                metadata["section"] = section_path[-1]
            content_types = {segment.metadata.get("content_type") for segment in segments}
            if "table" in content_types:
                metadata["content_type"] = "table"
            elif "key_value" in content_types and len(content_types) == 1:
                metadata["content_type"] = "key_value"
            else:
                metadata.setdefault("content_type", "composite")
            for segment in segments:
                if segment.metadata:
                    for key, value in segment.metadata.items():
                        metadata.setdefault(key, value)
        metadata.setdefault("section", metadata.get("section", "document_overview"))
        return metadata

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._encoding is None:
            return re.findall(r"\S+", text)
        try:
            return self._encoding.encode(text)
        except Exception:
            logger.debug("Tokenisation failed; falling back to whitespace split", exc_info=True)
            return re.findall(r"\S+", text)

    def _decode(self, tokens: Sequence[int]) -> str:
        if self._encoding is None:
            return " ".join(str(token) for token in tokens)
        try:
            return self._encoding.decode(list(tokens))
        except Exception:
            logger.debug("Token decode failed; returning placeholder join", exc_info=True)
            return " ".join(str(token) for token in tokens)

    def _canonicalise(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value or "").strip()
        return cleaned

    def _normalise_chunk_text(self, lines: Sequence[str]) -> str:
        parts: List[str] = []
        for line in lines:
            cleaned = re.sub(r"\s+", " ", (line or "")).strip()
            if cleaned:
                parts.append(cleaned)
        text = "\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_boilerplate_lines(self, text: str) -> set[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return set()
        frequency = Counter(lines)
        threshold = max(3, int(len(lines) * 0.2))
        boilerplate: set[str] = set()
        for line, count in frequency.items():
            if len(line) <= 80 and count >= threshold:
                boilerplate.add(line)
        return boilerplate

    def _detect_title(self, structured: StructuredDocument) -> Optional[str]:
        for element in structured.elements:
            if element.type == "title" and element.text.strip():
                return self._canonicalise(element.text)
        for element in structured.elements:
            if element.type == "heading" and element.text.strip():
                return self._canonicalise(element.text)
        return None

    def _normalise_section_name(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "section"

    def _derive_source_type(self, document_type: str) -> str:
        mapping = {
            "invoice": "Invoice",
            "purchase_order": "PO",
            "po": "PO",
            "contract": "Contract",
            "quote": "Quote",
            "policy": "Policy",
        }
        key = (document_type or "").replace(" ", "_").lower()
        return mapping.get(key, "Upload")


__all__ = ["SemanticChunk", "SemanticChunker"]

