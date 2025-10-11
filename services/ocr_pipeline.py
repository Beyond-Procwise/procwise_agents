"""Utilities for OCR-aware text extraction and preprocessing.

The goal of this module is not to deliver state-of-the-art OCR inside the
unit tests – that would introduce heavyweight dependencies and network
requirements.  Instead we provide a light abstraction that mirrors the
production pipeline:

* When high quality OCR engines such as Mindee's DocTR are available we
  prefer them, falling back to Tesseract if required.
* Scanned documents frequently contain fragmented text (broken words,
  newline separated key/value pairs, noisy whitespace).  The preprocessor
  applies deterministic clean-up passes so downstream parsers receive
  coherent sentences and table rows.

The :class:`OCRPreprocessor` intentionally degrades gracefully when optional
packages are missing.  In the test-suite the extractor operates on plain text
files, so the class mainly provides the improved text normalisation behaviour
required by the new acceptance criteria without introducing hard dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Optional runtime integrations -------------------------------------------------
try:  # pragma: no cover - optional dependency
    from doctr.io import DocumentFile  # type: ignore
    from doctr.models import ocr_predictor  # type: ignore
except Exception:  # pragma: no cover - docTR is optional
    DocumentFile = None  # type: ignore
    ocr_predictor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - pytesseract is optional
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


@dataclass
class OCRResult:
    text: str
    tables: List[List[List[str]]]


class OCRPreprocessor:
    """Perform OCR extraction and clean-up for scanned documents."""

    def __init__(self) -> None:
        self._doctr = None
        self._tesseract = None
        self._doctr_enabled = os.getenv("PROCWISE_ENABLE_DOCTR", "").lower() in {
            "1",
            "true",
            "yes",
        }

    # ------------------------------------------------------------------
    # OCR drivers
    # ------------------------------------------------------------------
    def extract(self, path: Path, *, scanned: bool) -> Optional[OCRResult]:
        """Return OCR text when ``scanned`` metadata indicates it is required."""

        if not scanned:
            return None

        if DocumentFile is not None and ocr_predictor is not None and self._doctr_enabled:
            text = self._extract_with_doctr(path)
            if text:
                return OCRResult(text=text, tables=[])

        text = self._extract_with_tesseract(path)
        if text:
            return OCRResult(text=text, tables=[])

        return None

    def _extract_with_doctr(self, path: Path) -> Optional[str]:  # pragma: no cover - optional
        try:
            if self._doctr is None:
                self._doctr = ocr_predictor(pretrained=True)  # type: ignore[call-arg]
            if self._doctr is None:
                return None
            document = self._load_document(path)
            if document is None:
                return None
            result = self._doctr(document)
            pages = []
            for page in result.export().get("pages", []):
                content = [word.get("value") for block in page.get("blocks", []) for line in block.get("lines", []) for word in line.get("words", [])]
                pages.append(" ".join(content))
            return "\n".join(filter(None, pages))
        except Exception:
            logger.debug("DocTR OCR failed", exc_info=True)
            return None

    def _load_document(self, path: Path):  # pragma: no cover - optional helper
        try:
            if path.suffix.lower() == ".pdf":
                return DocumentFile.from_pdf(path.as_posix())  # type: ignore[attr-defined]
            return DocumentFile.from_images([path.as_posix()])  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Unable to load document for DocTR OCR", exc_info=True)
            return None

    def _extract_with_tesseract(self, path: Path) -> Optional[str]:  # pragma: no cover - optional
        if pytesseract is None or Image is None:
            return None
        try:
            image = Image.open(path.as_posix())
        except Exception:
            logger.debug("PIL could not open %s for OCR", path, exc_info=True)
            return None
        try:
            text = pytesseract.image_to_string(image)
        except Exception:
            logger.debug("Tesseract OCR failed", exc_info=True)
            return None
        return text

    # ------------------------------------------------------------------
    # Text pre-processing helpers
    # ------------------------------------------------------------------
    def preprocess_text(self, text: str, *, scanned: bool) -> str:
        """Apply deterministic clean-up to OCR/digital text."""

        if not text:
            return text

        lines = [line.rstrip("\n") for line in text.splitlines()]
        lines = self._normalise_whitespace(lines)
        if scanned:
            lines = self._merge_vertical_pairs(lines)
            lines = self._repair_split_tokens(lines)
        else:
            lines = self._trim_spurious_spaces(lines)

        cleaned = [line for line in lines if line.strip()]
        return "\n".join(cleaned)

    def _normalise_whitespace(self, lines: Sequence[str]) -> List[str]:
        normalised: List[str] = []
        for line in lines:
            collapsed = line.replace("\u00a0", " ")  # non-breaking space
            collapsed = re.sub(r"\t+", "    ", collapsed)
            # Preserve multiple spaces that signal table separators.
            collapsed = re.sub(r"\s+$", "", collapsed)
            normalised.append(collapsed)
        return normalised

    def _trim_spurious_spaces(self, lines: Sequence[str]) -> List[str]:
        trimmed: List[str] = []
        for line in lines:
            if self._looks_like_table(line):
                trimmed.append(line)
            else:
                trimmed.append(re.sub(r"\s{2,}", " ", line).strip())
        return trimmed

    def _merge_vertical_pairs(self, lines: Sequence[str]) -> List[str]:
        merged: List[str] = []
        idx = 0
        while idx < len(lines):
            current = lines[idx].strip()
            if not current:
                idx += 1
                continue

            lowered_current = current.lower()
            if lowered_current in {
                "invoice",
                "purchase order",
                "contract",
                "quotation",
                "quote",
            }:
                merged.append(current)
                idx += 1
                continue

            if self._looks_like_table(current):
                merged.append(lines[idx])
                idx += 1
                continue

            if self._contains_value_delimiter(current):
                merged.append(current)
                idx += 1
                continue

            next_index = idx + 1
            if next_index < len(lines):
                candidate = lines[next_index].strip()
                if (
                    candidate
                    and not self._contains_value_delimiter(candidate)
                    and self._is_probable_value(candidate)
                ):
                    merged.append(f"{current}: {candidate}")
                    idx += 2
                    continue

            merged.append(current)
            idx += 1
        return merged

    def _repair_split_tokens(self, lines: Sequence[str]) -> List[str]:
        repaired: List[str] = []
        for line in lines:
            if self._looks_like_table(line):
                repaired.append(self._repair_numeric_blocks(line))
                continue
            fixed = self._repair_numeric_blocks(line)
            fixed = re.sub(r"(?i)contract\s+number", "Contract Number", fixed)
            fixed = re.sub(r"(?i)invoice\s+number", "Invoice Number", fixed)
            repaired.append(fixed)
        return repaired

    def _repair_numeric_blocks(self, line: str) -> str:
        return re.sub(r"(?<=\d)[\s](?=\d{2,})", "", line)

    def _contains_value_delimiter(self, line: str) -> bool:
        return ":" in line or re.search(r"\s+-\s+", line) is not None or "#" in line

    def _looks_like_table(self, line: str) -> bool:
        if "|" in line:
            return True
        return bool(re.search(r"\s{2,}[A-Za-z0-9]", line)) and bool(re.search(r"\s{2,}\S", line))

    def _is_probable_value(self, text: str) -> bool:
        if not text:
            return False
        if self._looks_like_table(text):
            return False
        if len(text) > 80:
            return False
        if re.match(r"^[A-Za-z0-9][A-Za-z0-9 .,#/&-]*$", text):
            lowered_tokens = re.findall(r"[a-z]+", text.lower())
            header_tokens = {
                "invoice",
                "contract",
                "agreement",
                "purchase",
                "order",
                "quote",
                "quotation",
                "number",
                "reference",
                "supplier",
                "vendor",
                "payment",
                "terms",
                "currency",
                "total",
                "amount",
                "date",
                "po",
            }
            if lowered_tokens and set(lowered_tokens).issubset(header_tokens):
                return False
            return True
        if re.search(r"\d", text):
            return True
        return False

    # ------------------------------------------------------------------
    # Convenience helpers for tests
    # ------------------------------------------------------------------
    def enhance(self, text: str, *, scanned: bool) -> str:
        """Compatibility wrapper for older code paths."""

        return self.preprocess_text(text, scanned=scanned)


__all__ = ["OCRPreprocessor", "OCRResult"]
