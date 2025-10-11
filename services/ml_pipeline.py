"""Machine-learning inspired helpers for procurement document extraction.

This module implements lightweight, dependency-free facsimiles of the
approaches recommended in the business requirements.  The intent is to
mirror a fine-tuned LayoutLM classifier for header detection and a Donut
style sequence-to-sequence generator for table understanding without
requiring heavy GPU-bound libraries inside the unit test environment.

The models below learn tiny logistic decision surfaces so that they can be
trained inside the repository.  They expose deterministic APIs that make the
behaviour testable while still representing the algorithmic steps required to
achieve high accuracy on heterogeneous procurement documents.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class MLExtractionResult:
    """Container for the hybrid document understanding output."""

    header: Dict[str, str]
    line_items: List[Dict[str, str]]
    tables: List[Dict[str, Any]]


@dataclass
class _TrainingSample:
    text: str
    position: float
    label: int


class _LogisticScorer:
    """Tiny logistic regression implementation used by the LayoutLM emulator."""

    def __init__(self, feature_count: int) -> None:
        self._weights = [0.0] * (feature_count + 1)  # bias term included

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def predict(self, features: Sequence[float]) -> float:
        value = self._weights[0]
        for weight, feature in zip(self._weights[1:], features):
            value += weight * feature
        return self._sigmoid(value)

    def fit(self, samples: Iterable[Tuple[Sequence[float], int]] , *, epochs: int = 400, lr: float = 0.35) -> None:
        for _ in range(epochs):
            for features, label in samples:
                prediction = self.predict(features)
                error = label - prediction
                self._weights[0] += lr * error
                for idx, feature in enumerate(features, start=1):
                    self._weights[idx] += lr * error * feature


class LayoutLMFineTuner:
    """Emulates fine-tuned LayoutLM style classification for header fields."""

    def __init__(
        self,
        header_lookup: Dict[str, Dict[str, str]],
        header_keywords: Dict[str, Dict[str, List[Tuple[str, ...]]]],
    ) -> None:
        self._header_lookup = header_lookup
        self._header_keywords = header_keywords
        self._scorer = _LogisticScorer(feature_count=9)
        self._train()

    def _train(self) -> None:
        corpus = [
            _TrainingSample("Invoice Number: INV-1001", 0.05, 1),
            _TrainingSample("Vendor: ACME Components", 0.08, 1),
            _TrainingSample("Invoice Date: 2024-03-02", 0.12, 1),
            _TrainingSample("Due Date: 2024-03-16", 0.16, 1),
            _TrainingSample("Invoice Total: 1500.00", 0.2, 1),
            _TrainingSample("Currency", 0.22, 1),
            _TrainingSample("USD", 0.23, 0),
            _TrainingSample("Item Description    Qty    Unit Price    Line Total", 0.35, 0),
            _TrainingSample("Laptop Pro 15       2      700           1400", 0.4, 0),
            _TrainingSample("Purchase Order Number", 0.05, 1),
            _TrainingSample("PO-9001", 0.055, 0),
            _TrainingSample("Total Amount: 2500.00", 0.18, 1),
            _TrainingSample("Service Agreement", 0.07, 1),
            _TrainingSample("Contract Number", 0.08, 1),
            _TrainingSample("Contract Value", 0.2, 1),
            _TrainingSample("Line Description   Qty   Unit Price   Total", 0.4, 0),
            _TrainingSample("Subtotal", 0.6, 0),
            _TrainingSample("Grand Total", 0.62, 0),
            _TrainingSample("Ship To", 0.09, 1),
            _TrainingSample("Billing Address", 0.1, 1),
        ]

        training_data = [
            (self._featurise(sample.text, sample.position, 1.0), sample.label)
            for sample in corpus
        ]
        self._scorer.fit(training_data, epochs=450, lr=0.32)

    def extract(self, lines: Sequence[str], document_type: str) -> Dict[str, str]:
        header: Dict[str, str] = {}
        probabilities = [
            self._scorer.predict(self._featurise(line, idx / max(1, len(lines) - 1 or 1), len(lines)))
            for idx, line in enumerate(lines)
        ]

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            score = probabilities[idx] if idx < len(probabilities) else 0.0
            if not line:
                idx += 1
                continue
            if score < 0.45:
                idx += 1
                continue

            key_text, value_text, consumed = self._derive_pair(lines, idx)
            if not key_text or not value_text:
                idx += max(consumed, 1)
                continue

            canonical = self._resolve_header_key(key_text, document_type)
            if canonical:
                header.setdefault(canonical, value_text.strip())
            idx += max(consumed, 1)
        return header

    def _derive_pair(self, lines: Sequence[str], index: int) -> Tuple[str, Optional[str], int]:
        raw = lines[index].strip()
        if not raw:
            return "", None, 1

        colon_split = raw.split(":", 1)
        if len(colon_split) == 2 and colon_split[0].strip():
            key, value = colon_split[0].strip(" #\t-"), colon_split[1].strip()
            if value:
                return key, value, 1

        dash_match = re.match(
            r"^(?P<key>[A-Za-z][A-Za-z0-9 .#/&-]{2,})\s+-\s+(?P<value>.+)$", raw
        )
        if dash_match:
            return dash_match.group("key"), dash_match.group("value").strip(), 1

        next_index = index + 1
        if next_index < len(lines):
            candidate_value = lines[next_index].strip()
            if candidate_value and self._value_confidence(candidate_value) > 0.55:
                return raw, candidate_value, 2

        return raw, None, 1

    def _value_confidence(self, value: str) -> float:
        if not value:
            return 0.0
        if len(value) > 120:
            return 0.0
        if re.match(r"^[0-9]{2,4}[-/][0-9]{1,2}[-/][0-9]{1,2}$", value):
            return 0.95
        if re.search(r"\d", value) and not re.search(r"[A-Za-z]{5,}\s{2,}", value):
            return 0.8
        if value.isupper() and len(value.split()) <= 4:
            return 0.7
        if len(value.split()) <= 5:
            return 0.65
        return 0.4

    def _resolve_header_key(self, key: str, document_type: str) -> str:
        normalised = _normalise_label(key)
        for scope in filter(None, (document_type, "*")):
            lookup = self._header_lookup.get(scope, {})
            mapped = lookup.get(normalised)
            if mapped:
                return mapped

        tokens = re.sub(r"[^A-Za-z0-9]+", " ", key).lower().split()
        token_set = set(tokens)
        if not tokens:
            return ""

        patterns = self._header_keywords.get(document_type, {})
        combined = {}
        combined.update(self._header_keywords.get("*", {}))
        combined.update(patterns)

        for canonical, sequences in combined.items():
            for sequence in sequences:
                if all(any(keyword in token for token in tokens) for keyword in sequence):
                    return canonical

        heuristic_map = {
            "invoice_id": [{"invoice", "number"}],
            "invoice_date": [{"invoice", "date"}],
            "due_date": [{"due", "date"}],
            "po_id": [{"po", "number"}, {"purchase", "order", "number"}],
            "supplier_name": [{"supplier"}, {"vendor"}, {"seller"}],
            "currency": [{"currency"}],
            "total_amount": [{"total"}, {"amount"}],
            "invoice_total_incl_tax": [{"total"}, {"invoice"}],
            "contract_id": [{"contract", "number"}],
            "contract_title": [{"contract", "title"}, {"agreement", "title"}],
            "contract_start_date": [{"start", "date"}, {"effective", "date"}],
            "contract_end_date": [{"end", "date"}, {"expiry", "date"}],
        }
        for canonical, patterns in heuristic_map.items():
            for pattern in patterns:
                if pattern <= token_set:
                    return canonical

        if "date" in token_set:
            return "invoice_date" if document_type == "Invoice" else "order_date"
        if "total" in token_set:
            return "invoice_total_incl_tax" if document_type == "Invoice" else "total_amount"

        return ""

    def _featurise(self, text: str, position: float, total_lines: int) -> List[float]:
        cleaned = text.strip()
        tokens = cleaned.split()
        uppercase_chars = sum(1 for char in cleaned if char.isupper())
        alpha_chars = sum(1 for char in cleaned if char.isalpha()) or 1
        digit_chars = sum(1 for char in cleaned if char.isdigit())
        multi_space = 1.0 if re.search(r"\s{2,}\S", text) else 0.0
        has_keyword = 1.0 if re.search(r"(?i)(invoice|order|quote|contract|supplier|vendor|due|total|currency)", text) else 0.0
        trailing_colon = 1.0 if cleaned.endswith(":") else 0.0
        colon_present = 1.0 if ":" in cleaned else 0.0
        short_line = 1.0 if len(tokens) <= 6 else 0.0
        uppercase_ratio = uppercase_chars / alpha_chars
        digit_ratio = digit_chars / max(len(cleaned), 1)
        hyphen_present = 1.0 if "-" in cleaned else 0.0
        position_feature = position if total_lines > 1 else 0.0
        return [
            colon_present,
            has_keyword,
            short_line,
            multi_space,
            uppercase_ratio,
            digit_ratio,
            position_feature,
            hyphen_present,
            trailing_colon,
        ]


class DonutGenerator:
    """OCR-free seq2seq style table generator for procurement documents."""

    def __init__(
        self,
        line_lookup: Dict[str, Dict[str, str]],
        line_keywords: Dict[str, Dict[str, List[Tuple[str, ...]]]],
    ) -> None:
        self._line_lookup = line_lookup
        self._line_keywords = line_keywords

    def extract(self, lines: Sequence[str], document_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        tables: List[Dict[str, Any]] = []
        structured_items: List[Dict[str, str]] = []

        idx = 0
        while idx < len(lines):
            header_line = lines[idx].strip()
            if not header_line:
                idx += 1
                continue
            if not self._looks_like_table_header(header_line):
                idx += 1
                continue

            headers = self._split_row(header_line)
            if len(headers) < 2:
                idx += 1
                continue

            rows: List[List[str]] = []
            cursor = idx + 1
            while cursor < len(lines):
                candidate = lines[cursor].strip()
                if not candidate:
                    break
                split = self._split_row(candidate)
                if len(split) < 2:
                    break
                if self._looks_like_table_header(candidate):
                    break
                rows.append(split)
                cursor += 1

            if rows:
                table_rows = [dict(zip(headers, row)) for row in rows]
                tables.append({"headers": headers, "rows": table_rows})
                structured_items.extend(self._project_rows(headers, rows, document_type))
                idx = cursor
            else:
                idx += 1

        return tables, structured_items

    def _project_rows(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]], document_type: str
    ) -> List[Dict[str, str]]:
        projected: List[Dict[str, str]] = []
        canonical_headers = [self._resolve_line_key(header, document_type) for header in headers]
        for row in rows:
            row_dict: Dict[str, str] = {}
            for idx, value in enumerate(row):
                if idx >= len(canonical_headers):
                    break
                key = canonical_headers[idx]
                if not key:
                    continue
                cleaned = value.strip()
                if not cleaned:
                    continue
                row_dict[key] = cleaned
            description = row_dict.get("item_description", "").lower()
            if description and "total" in description:
                continue
            if row_dict:
                projected.append(row_dict)
        return projected

    def _resolve_line_key(self, label: str, document_type: str) -> str:
        normalised = _normalise_label(label)
        for scope in filter(None, (document_type, "*")):
            lookup = self._line_lookup.get(scope, {})
            mapped = lookup.get(normalised)
            if mapped:
                return mapped

        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", label).lower()
        tokens = cleaned.split()
        keyword_sets = self._line_keywords.get(document_type, {})
        combined = {}
        combined.update(self._line_keywords.get("*", {}))
        combined.update(keyword_sets)
        for canonical, patterns in combined.items():
            for pattern in patterns:
                if all(any(keyword in token for token in tokens) for keyword in pattern):
                    return canonical

        if "qty" in tokens or "quantity" in tokens:
            return "quantity"
        if "price" in tokens and "unit" in tokens:
            return "unit_price"
        if "description" in tokens or "item" in tokens:
            return "item_description"
        if "total" in tokens or "amount" in tokens:
            return "line_amount"
        return ""

    @staticmethod
    def _looks_like_table_header(line: str) -> bool:
        lowered = line.lower()
        has_descriptor = any(keyword in lowered for keyword in ("description", "item", "product", "service", "line"))
        has_quantity = any(keyword in lowered for keyword in ("qty", "quantity", "units"))
        has_amount = any(keyword in lowered for keyword in ("amount", "total", "price"))
        return bool(re.search(r"\s{2,}", line)) and ((has_descriptor and has_quantity) or (has_descriptor and has_amount))

    @staticmethod
    def _split_row(text: str) -> List[str]:
        if "\t" in text:
            parts = [part.strip() for part in text.split("\t") if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"\s{2,}", text) if part.strip()]
        return parts


class DocumentUnderstandingModel:
    """High level orchestrator combining LayoutLM and Donut emulators."""

    def __init__(
        self,
        header_lookup: Dict[str, Dict[str, str]],
        line_lookup: Dict[str, Dict[str, str]],
        header_keywords: Dict[str, Dict[str, List[Tuple[str, ...]]]],
        line_keywords: Dict[str, Dict[str, List[Tuple[str, ...]]]],
    ) -> None:
        self._layout_model = LayoutLMFineTuner(header_lookup, header_keywords)
        self._donut_model = DonutGenerator(line_lookup, line_keywords)

    def infer(
        self, text: str, document_type: str, *, scanned: bool = False
    ) -> Optional[MLExtractionResult]:
        if not text.strip():
            return None
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        header = self._layout_model.extract(lines, document_type)
        tables, items = self._donut_model.extract(lines, document_type)

        if not header and not items and not tables:
            return None

        return MLExtractionResult(header=header, line_items=items, tables=tables)


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


__all__ = ["DocumentUnderstandingModel", "MLExtractionResult"]
