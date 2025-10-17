import json
import os
import re
import sqlite3
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.document_extractor import DocumentExtractor
from services.ocr_pipeline import OCRPreprocessor, OCRResult


def _connection_factory(db_path: Path):
    def factory():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    return factory


class _FixtureOCR(OCRPreprocessor):
    """Lightweight OCR stub that reuses deterministic preprocessing."""

    def extract(self, path: Path, *, scanned: bool) -> OCRResult | None:  # type: ignore[override]
        if not scanned:
            return None
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return OCRResult(text=text, tables=[])


@pytest.fixture
def extractor(tmp_path):
    db_path = tmp_path / "extraction.sqlite"
    factory = _connection_factory(db_path)
    service = DocumentExtractor(factory, ocr_preprocessor=_FixtureOCR())
    return service, db_path


def _read_table(db_path: Path, table: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(f'SELECT * FROM "{table}"').fetchone()


def _header_f1(header: dict, expected_keys: Sequence[str]) -> float:
    predicted_keys = {
        key for key, value in header.items() if value is not None and str(value).strip()
    }
    expected = set(expected_keys)
    if not predicted_keys or not expected:
        return 0.0

    intersection = predicted_keys & expected
    precision = len(intersection) / len(predicted_keys)
    recall = len(intersection) / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalise_decimal(value: object) -> Decimal:
    cleaned = re.sub(r"[^0-9.+-]", "", str(value))
    if not cleaned:
        return Decimal("0")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _line_mismatch_rate(
    extracted: Sequence[dict],
    expected: Sequence[dict],
    *,
    total_key: str,
) -> float:
    if not expected:
        return 0.0

    mismatches = 0
    for golden in expected:
        target = golden["item_description"].lower()
        match = next(
            (item for item in extracted if item.get("item_description", "").lower() == target),
            None,
        )
        if match is None:
            mismatches += 1
            continue

        for field in ("quantity", "unit_price", total_key):
            if field not in golden:
                continue
            predicted_value = match.get(field)
            if predicted_value is None:
                mismatches += 1
                break
            if _normalise_decimal(predicted_value) != _normalise_decimal(golden[field]):
                mismatches += 1
                break

    return mismatches / len(expected)


def _assert_extraction_metrics(
    result,
    expected_header_keys: Sequence[str],
    expected_line_items: Sequence[dict],
    *,
    total_key: str,
) -> None:
    header_f1 = _header_f1(result.header, expected_header_keys)
    assert header_f1 >= 0.95, f"Header F1 too low: {header_f1:.2f}"

    mismatch_rate = _line_mismatch_rate(
        result.line_items,
        expected_line_items,
        total_key=total_key,
    )
    assert (
        mismatch_rate < 0.02
    ), f"Line reconciliation mismatch rate {mismatch_rate:.2%} exceeds threshold"


def test_digital_invoice_layout_pipeline(tmp_path, extractor):
    doc = tmp_path / "digital_invoice.txt"
    doc.write_text(
        "\n".join(
            [
                "Invoice Number: INV-5010",
                "Vendor: Oriole Components",
                "Invoice Date: 2024-03-05",
                "Due Date: 2024-03-19",
                "Currency: USD",
                "Invoice Total: 1820.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Laptop Pro 15       2      700           1400",
                "Freight             1      120           120",
                "Setup Fee           1      300           300",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Invoice"
    assert result.header["invoice_id"] == "INV-5010"
    assert result.header["supplier_name"].lower() == "oriole components"
    assert len(result.line_items) == 3
    assert result.tables
    assert result.schema_reference["document_type"] == "Invoice"

    _assert_extraction_metrics(
        result,
        [
            "invoice_id",
            "supplier_name",
            "invoice_date",
            "due_date",
            "currency",
            "invoice_total_incl_tax",
        ],
        [
            {
                "item_description": "Laptop Pro 15",
                "quantity": "2",
                "unit_price": "700",
                "line_amount": "1400",
            },
            {
                "item_description": "Freight",
                "quantity": "1",
                "unit_price": "120",
                "line_amount": "120",
            },
            {
                "item_description": "Setup Fee",
                "quantity": "1",
                "unit_price": "300",
                "line_amount": "300",
            },
        ],
        total_key="line_amount",
    )

    row = _read_table(db_path, "proc.raw_invoice")
    header = json.loads(row["header_json"])
    assert header["invoice_id"] == "INV-5010"
    line_items = json.loads(row["line_items_json"])
    assert line_items[0]["item_description"].lower() == "laptop pro 15"


def test_digital_purchase_order_layout_pipeline(tmp_path, extractor):
    doc = tmp_path / "digital_po.txt"
    doc.write_text(
        "\n".join(
            [
                "Purchase Order Number: PO-7201",
                "Supplier: Northwind Traders",
                "Order Date: 2024-02-10",
                "Total Amount: 5120.00",
                "Currency: USD",
                "Line Description    Qty    Unit Price    Total",
                "Workstations        8      450           3600",
                "Docking Stations    8      80            640",
                "24in Monitors       8      110           880",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Purchase_Order"
    assert result.header["po_id"] == "PO-7201"
    assert result.header["total_amount"] == "5120.00"
    assert result.schema_reference["document_type"] == "Purchase_Order"

    _assert_extraction_metrics(
        result,
        ["po_id", "supplier_name", "order_date", "currency", "total_amount"],
        [
            {
                "item_description": "Workstations",
                "quantity": "8",
                "unit_price": "450",
                "line_total": "3600",
            },
            {
                "item_description": "Docking Stations",
                "quantity": "8",
                "unit_price": "80",
                "line_total": "640",
            },
            {
                "item_description": "24in Monitors",
                "quantity": "8",
                "unit_price": "110",
                "line_total": "880",
            },
        ],
        total_key="line_total",
    )

    row = _read_table(db_path, "proc.raw_purchase_order")
    payload = json.loads(row["header_json"])
    assert payload["po_id"] == "PO-7201"


def test_scanned_invoice_ocr_fallback(tmp_path, extractor):
    doc = tmp_path / "scanned_invoice.txt"
    doc.write_text(
        "\n".join(
            [
                "INVOICE",
                "Invoice Number",
                "INV-3003",
                "Vendor",
                "Helios Manufacturing",
                "Invoice Date",
                "2024-04-02",
                "Due Date",
                "2024-04-16",
                "Currency",
                "EUR",
                "Invoice Total",
                "2150.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Solar Panel Kit     5      350.00        1750.00",
                "Freight             1      400.00        400.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned", "ocr": True})

    assert result.document_type == "Invoice"
    assert result.header["invoice_id"] == "INV-3003"
    assert result.header["supplier_name"].lower() == "helios manufacturing"
    assert result.header["invoice_total_incl_tax"] == "2150.00"

    _assert_extraction_metrics(
        result,
        [
            "invoice_id",
            "supplier_name",
            "invoice_date",
            "due_date",
            "currency",
            "invoice_total_incl_tax",
        ],
        [
            {
                "item_description": "Solar Panel Kit",
                "quantity": "5",
                "unit_price": "350.00",
                "line_amount": "1750.00",
            },
            {
                "item_description": "Freight",
                "quantity": "1",
                "unit_price": "400.00",
                "line_amount": "400.00",
            },
        ],
        total_key="line_amount",
    )


def test_scanned_quote_layout_alignment(tmp_path, extractor):
    doc = tmp_path / "scanned_quote.txt"
    doc.write_text(
        "\n".join(
            [
                "QUOTE",
                "Quote Number",
                "Q-8820",
                "Supplier",
                "Aurora Design Studio",
                "Quote Date",
                "2024-05-12",
                "Validity Date",
                "2024-06-12",
                "Currency",
                "USD",
                "Quote Total",
                "2840.00",
                "Item Description    Qty    Unit Price    Amount",
                "Brand Workshop      2      600.00        1200.00",
                "Design Sprint       1      850.00        850.00",
                "Prototype Mockups   3      263.33        790.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    assert result.document_type == "Quote"
    assert result.header["quote_id"] == "Q-8820"
    assert result.header["total_amount"] == "2840.00"

    _assert_extraction_metrics(
        result,
        [
            "quote_id",
            "supplier_id",
            "quote_date",
            "validity_date",
            "currency",
            "total_amount",
        ],
        [
            {
                "item_description": "Brand Workshop",
                "quantity": "2",
                "unit_price": "600.00",
                "line_total": "1200.00",
            },
            {
                "item_description": "Design Sprint",
                "quantity": "1",
                "unit_price": "850.00",
                "line_total": "850.00",
            },
            {
                "item_description": "Prototype Mockups",
                "quantity": "3",
                "unit_price": "263.33",
                "line_total": "790.00",
            },
        ],
        total_key="line_total",
    )
