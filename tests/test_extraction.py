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


def _connection_factory(db_path: Path):
    def factory():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    return factory


@pytest.fixture
def extractor(tmp_path):
    db_path = tmp_path / "extraction.sqlite"
    factory = _connection_factory(db_path)
    return DocumentExtractor(factory), db_path


def _read_table(db_path: Path, table: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f'SELECT * FROM "{table}"').fetchone()
        return row


def _header_f1(header: dict, expected_keys: Sequence[str]) -> float:
    predicted_keys = {key for key, value in header.items() if str(value).strip()}
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


def test_digital_invoice_extraction(tmp_path, extractor):
    doc = tmp_path / "invoice_digital.txt"
    doc.write_text(
        "\n".join(
            [
                "Invoice Number: INV-1001",
                "Vendor: ACME Components",
                "Invoice Date: 2024-03-02",
                "Due Date: 2024-03-16",
                "Invoice Total: 1500.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Laptop Pro 15       2      700           1400",
                "Freight             1      100           100",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Invoice"
    assert result.header["invoice_id"] == "INV-1001"
    assert result.header["supplier_name"].lower() == "acme components"
    assert len(result.line_items) == 2
    assert result.tables  # ensure at least one table captured
    assert result.line_items[0]["quantity"] == "2"
    assert result.line_items[0]["line_amount"] == "1400"
    assert result.schema_reference["document_type"] == "Invoice"
    assert "invoice_id" in result.schema_reference["header_fields"]
    assert {"item_description", "quantity", "unit_price", "line_amount"}.issubset(
        set(result.schema_reference["line_item_fields"])
    )

    _assert_extraction_metrics(
        result,
        [
            "invoice_id",
            "supplier_name",
            "invoice_date",
            "due_date",
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
                "unit_price": "100",
                "line_amount": "100",
            },
        ],
        total_key="line_amount",
    )

    row = _read_table(db_path, "proc.raw_invoice")
    assert row["document_type"] == "Invoice"
    header = json.loads(row["header_json"])
    assert header["invoice_id"] == "INV-1001"
    assert header["supplier_name"].lower() == "acme components"
    line_items = json.loads(row["line_items_json"])
    assert line_items[0]["item_description"].lower() == "laptop pro 15"
    assert line_items[0]["quantity"] == "2"
    assert line_items[0]["line_amount"] == "1400"


def test_digital_invoice_with_multiline_headers(tmp_path, extractor):
    doc = tmp_path / "invoice_multiline.txt"
    doc.write_text(
        "\n".join(
            [
                "Invoice Number",
                "INV-7000",
                "Vendor",
                "Newlight Systems",
                "Invoice Date",
                "2024-04-01",
                "Currency",
                "USD",
                "Invoice Total",
                "500.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Support Subscription    1      500.00    500.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Invoice"
    assert result.header["invoice_id"] == "INV-7000"
    assert result.header["supplier_name"].lower() == "newlight systems"
    assert result.header["invoice_total_incl_tax"] == "500.00"
    assert result.header["currency"] == "USD"
    assert result.line_items[0]["item_description"].lower() == "support subscription"

    _assert_extraction_metrics(
        result,
        [
            "invoice_id",
            "supplier_name",
            "invoice_date",
            "currency",
            "invoice_total_incl_tax",
        ],
        [
            {
                "item_description": "Support Subscription",
                "quantity": "1",
                "unit_price": "500.00",
                "line_amount": "500.00",
            }
        ],
        total_key="line_amount",
    )

    row = _read_table(db_path, "proc.raw_invoice")
    header = json.loads(row["header_json"])
    assert header["invoice_id"] == "INV-7000"
    assert header["currency"] == "USD"


def test_digital_purchase_order_extraction(tmp_path, extractor):
    doc = tmp_path / "purchase_order.txt"
    doc.write_text(
        "\n".join(
            [
                "Purchase Order Number: PO-9001",
                "Supplier: Northwind Traders",
                "Order Date: 2024-01-10",
                "Total Amount: 2500.00",
                "Line Description    Qty    Unit Price    Total",
                "Workstations        5      400           2000",
                "Docking Stations    5      100           500",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Purchase_Order"
    assert result.header["po_id"] == "PO-9001"
    assert result.header["supplier_name"].lower() == "northwind traders"
    assert len(result.line_items) == 2
    assert result.line_items[0]["quantity"] == "5"
    assert result.line_items[0]["line_total"] == "2000"
    assert result.schema_reference["document_type"] == "Purchase_Order"
    assert "po_id" in result.schema_reference["header_fields"]
    assert {"item_description", "quantity", "unit_price", "line_total"}.issubset(
        set(result.schema_reference["line_item_fields"])
    )

    _assert_extraction_metrics(
        result,
        ["po_id", "supplier_name", "order_date", "total_amount"],
        [
            {
                "item_description": "Workstations",
                "quantity": "5",
                "unit_price": "400",
                "line_total": "2000",
            },
            {
                "item_description": "Docking Stations",
                "quantity": "5",
                "unit_price": "100",
                "line_total": "500",
            },
        ],
        total_key="line_total",
    )

    row = _read_table(db_path, "proc.raw_purchase_order")
    payload = json.loads(row["header_json"])
    assert payload["po_id"] == "PO-9001"


def test_document_type_detection_prefers_invoice(tmp_path, extractor):
    doc = tmp_path / "ambiguous_invoice.txt"
    doc.write_text(
        "\n".join(
            [
                "Commercial Document",
                "Invoice Number: INV-8800",
                "Purchase Order Reference: PO-7711",
                "Vendor: Apex Integration",
                "Invoice Date: 2024-05-11",
                "Due Date: 2024-05-25",
                "Total Due: 980.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Enterprise Support   1      980.00        980.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Invoice"
    expected_keys = [
        "invoice_id",
        "supplier_name",
        "invoice_date",
        "due_date",
        "invoice_total_incl_tax",
        "po_id",
    ]
    assert _header_f1(result.header, expected_keys) >= 0.95
    assert result.header["invoice_id"] == "INV-8800"
    assert result.line_items and result.line_items[0]["line_amount"].startswith("980")


def test_document_type_detection_for_agreement(tmp_path, extractor):
    doc = tmp_path / "service_agreement.txt"
    doc.write_text(
        "\n".join(
            [
                "Master Service Agreement",
                "Agreement Number: C-5501",
                "Agreement Title: Managed Infrastructure Support",
                "Supplier: Horizon Digital",
                "Effective Date: 2024-01-15",
                "Expiry Date: 2025-01-14",
                "Payment Terms: Net 45",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Contract"
    assert result.header["contract_id"] == "C-5501"
    assert result.header["contract_title"].startswith("Managed Infrastructure")
    assert result.header["payment_terms"] == "Net 45"


def test_scanned_contract_with_multiline_headers(tmp_path, extractor):
    doc = tmp_path / "contract_multiline.txt"
    doc.write_text(
        "\n".join(
            [
                "Master Service Contract",
                "Contract Number",
                "C-7788",
                "Contract Title",
                "Managed Support Services",
                "Supplier",
                "Alpha Works Ltd",
                "Contract Start Date",
                "2024-02-01",
                "Contract End Date",
                "2025-02-01",
                "Payment Terms",
                "Net 30",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(
        doc, metadata={"ingestion_mode": "scanned", "ocr": True}
    )

    assert result.document_type == "Contract"
    assert result.header["contract_id"] == "C-7788"
    assert result.header["contract_title"].startswith("Managed Support")
    assert result.header["payment_terms"] == "Net 30"
    assert result.header["supplier_name"] == "Alpha Works Ltd"

    row = _read_table(db_path, "proc.raw_contracts")
    header = json.loads(row["header_json"])
    assert header["contract_id"] == "C-7788"
    assert header["payment_terms"] == "Net 30"


def test_scanned_contract_extraction(tmp_path, extractor):
    doc = tmp_path / "contract_scanned.txt"
    doc.write_text(
        "\n".join(
            [
                "Master Service Contract",
                "Contract Number: C-2024-001",
                "Contract Title: Managed Services Agreement",
                "Supplier: Stellar Tech Ltd",
                "Contract Start Date: 2024-01-01",
                "Contract End Date: 2025-01-01",
                "Payment Terms: Net 45",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(
        doc,
        metadata={"ingestion_mode": "scanned", "ocr": True},
    )

    assert result.document_type == "Contract"
    assert result.header["contract_id"] == "C-2024-001"
    assert result.header["contract_title"].startswith("Managed Services")
    assert result.metadata["ingestion_mode"] == "scanned"
    assert result.schema_reference["document_type"] == "Contract"
    assert "contract_id" in result.schema_reference["header_fields"]

    row = _read_table(db_path, "proc.raw_contracts")
    header = json.loads(row["header_json"])
    assert header["contract_id"] == "C-2024-001"


def test_scanned_quote_with_table(tmp_path, extractor):
    doc = tmp_path / "quote_scanned.txt"
    doc.write_text(
        "\n".join(
            [
                "Quotation #: Q-7788",
                "Supplier: Bright Office Co",
                "Quote Date: 2024-02-12",
                "Total Amount: 980.00",
                "Item Description    Qty    Unit Price    Amount",
                "Ergonomic Chair     4      120           480",
                "Standing Desk       2      250           500",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(
        doc,
        metadata={"ingestion_mode": "scanned"},
    )

    assert result.document_type == "Quote"
    assert result.tables and len(result.tables[0]["rows"]) == 2
    assert result.line_items[0]["item_description"].lower() == "ergonomic chair"
    assert result.schema_reference["document_type"] == "Quote"
    assert "quote_id" in result.schema_reference["header_fields"]

    row = _read_table(db_path, "proc.raw_quotes")
    tables = json.loads(row["tables_json"])
    assert tables[0]["rows"][1]["item_description"].lower() == "standing desk"


def test_document_type_detection_scanned_purchase_order(tmp_path, extractor):
    doc = tmp_path / "scanned_po_detection.txt"
    doc.write_text(
        "\n".join(
            [
                "Order Summary",
                "Order Reference #: PO-3300",
                "Vendor: Summit Gear",
                "Order Date: 2024-04-15",
                "Expected Delivery Date: 2024-04-30",
                "Line Description    Qty    Unit Price    Total",
                "Rugged Tablet       10     650.00        6500.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    assert result.document_type == "Purchase_Order"
    assert result.header["po_id"] == "PO-3300"
    assert result.header["supplier_name"].lower() == "summit gear"
    assert result.line_items[0]["line_total"].startswith("6500")


def test_document_type_detection_scanned_quote(tmp_path, extractor):
    doc = tmp_path / "scanned_quote_detection.txt"
    doc.write_text(
        "\n".join(
            [
                "Commercial Proposal",
                "Proposal Number: Q-9090",
                "Supplier: Skyline Interiors",
                "Valid Until: 2024-05-30",
                "Total Amount: 1450.00",
                "Item Description    Qty    Unit Price    Amount",
                "Boardroom Table     1      850.00        850.00",
                "Leather Chairs      4      150.00        600.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    assert result.document_type == "Quote"
    assert result.header["quote_id"] == "Q-9090"
    assert result.header["supplier_name"].lower() == "skyline interiors"
    assert len(result.line_items) == 2


def test_accuracy_metrics_digital_invoice(tmp_path, extractor):
    doc = tmp_path / "invoice_accuracy.txt"
    doc.write_text(
        "\n".join(
            [
                "Invoice Number: INV-5010",
                "Supplier: Starlight Manufacturing",
                "Invoice Date: 2024-06-15",
                "Due Date: 2024-07-15",
                "Payment Terms: Net 30",
                "Currency: USD",
                "Invoice Total (incl tax): 2135.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Industrial Sensors    5    350.00    1750.00",
                "Calibration Service   1    200.00    200.00",
                "Sales Tax             1    185.00    185.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    expected_keys = [
        "invoice_id",
        "supplier_name",
        "invoice_date",
        "due_date",
        "payment_terms",
        "currency",
        "invoice_total_incl_tax",
    ]
    f1 = _header_f1(result.header, expected_keys)
    assert f1 >= 0.95

    expected_lines = [
        {
            "item_description": "Industrial Sensors",
            "quantity": "5",
            "unit_price": "350.00",
            "line_amount": "1750.00",
        },
        {
            "item_description": "Calibration Service",
            "quantity": "1",
            "unit_price": "200.00",
            "line_amount": "200.00",
        },
        {
            "item_description": "Sales Tax",
            "quantity": "1",
            "unit_price": "185.00",
            "line_amount": "185.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_amount"
    )
    assert mismatch_rate < 0.02
    assert result.header["currency"] == "USD"


def test_accuracy_metrics_digital_purchase_order(tmp_path, extractor):
    doc = tmp_path / "po_accuracy.txt"
    doc.write_text(
        "\n".join(
            [
                "Purchase Order Number: PO-4521",
                "Supplier: Orion Industrial Ltd",
                "Order Date: 2024-03-12",
                "Expected Delivery Date: 2024-03-26",
                "Currency: EUR",
                "Total Amount: 4820.00",
                "Item Description    Qty    Unit Price    Line Total",
                "High Torque Motors   4    950.00    3800.00",
                "Freight              1    220.00    220.00",
                "Install Services     1    800.00    800.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    expected_keys = [
        "po_id",
        "supplier_name",
        "order_date",
        "expected_delivery_date",
        "currency",
        "total_amount",
    ]
    f1 = _header_f1(result.header, expected_keys)
    assert f1 >= 0.95

    expected_lines = [
        {
            "item_description": "High Torque Motors",
            "quantity": "4",
            "unit_price": "950.00",
            "line_total": "3800.00",
        },
        {
            "item_description": "Freight",
            "quantity": "1",
            "unit_price": "220.00",
            "line_total": "220.00",
        },
        {
            "item_description": "Install Services",
            "quantity": "1",
            "unit_price": "800.00",
            "line_total": "800.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_total"
    )
    assert mismatch_rate < 0.02
    assert result.header["currency"] == "EUR"


def test_accuracy_metrics_scanned_invoice(tmp_path, extractor):
    doc = tmp_path / "invoice_scanned_accuracy.txt"
    doc.write_text(
        "\n".join(
            [
                "TAX INVOICE",
                "Invoice # INV-8844",
                "Supplier - Horizon Labs",
                "Invoice Date - 2024-05-10",
                "Due Date - 2024-05-24",
                "Payment Terms - Net 14",
                "Currency - USD",
                "Total Amount Due - 1480.00",
                "Item Description    Quantity    Unit Price    Line Total",
                "Lab Analysis Fee    2           600.00        1200.00",
                "Expedited Shipping  1           120.00        120.00",
                "Environmental Levy  1           160.00        160.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    expected_keys = [
        "invoice_id",
        "supplier_name",
        "invoice_date",
        "due_date",
        "payment_terms",
        "currency",
        "invoice_total_incl_tax",
    ]
    f1 = _header_f1(result.header, expected_keys)
    assert f1 >= 0.95

    expected_lines = [
        {
            "item_description": "Lab Analysis Fee",
            "quantity": "2",
            "unit_price": "600.00",
            "line_amount": "1200.00",
        },
        {
            "item_description": "Expedited Shipping",
            "quantity": "1",
            "unit_price": "120.00",
            "line_amount": "120.00",
        },
        {
            "item_description": "Environmental Levy",
            "quantity": "1",
            "unit_price": "160.00",
            "line_amount": "160.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_amount"
    )
    assert mismatch_rate < 0.02


def test_accuracy_metrics_scanned_quote(tmp_path, extractor):
    doc = tmp_path / "quote_scanned_accuracy.txt"
    doc.write_text(
        "\n".join(
            [
                "Quotation # Q-9905",
                "Supplier - Brightworks Studio",
                "Quote Date - 2024-04-18",
                "Validity Date - 2024-05-18",
                "Currency - USD",
                "Total Amount - 2250.00",
                "Item Description    Qty    Unit Price    Amount",
                "Creative Workshop   3    450.00    1350.00",
                "Prototype Mockups   3    300.00    900.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    expected_keys = [
        "quote_id",
        "supplier_name",
        "quote_date",
        "validity_date",
        "currency",
        "total_amount",
    ]
    f1 = _header_f1(result.header, expected_keys)
    assert f1 >= 0.95

    expected_lines = [
        {
            "item_description": "Creative Workshop",
            "quantity": "3",
            "unit_price": "450.00",
            "line_amount": "1350.00",
        },
        {
            "item_description": "Prototype Mockups",
            "quantity": "3",
            "unit_price": "300.00",
            "line_amount": "900.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_amount"
    )
    assert mismatch_rate < 0.02


def test_local_model_payload_enrichment(tmp_path):
    db_path = tmp_path / "llm.sqlite"
    factory = _connection_factory(db_path)

    class DummyLLM:
        def __init__(self) -> None:
            self.calls = []

        def extract(self, text, document_type, *, field_hints=None):
            self.calls.append(
                {"text": text, "document_type": document_type, "field_hints": field_hints}
            )
            return {
                "document_type": "Invoice",
                "header": {"Invoice Number": "INV-2000", "Currency": "USD"},
                "line_items": [
                    {
                        "Item Description": "Support Plan",
                        "Qty": "1",
                        "Unit Price": "1000",
                        "Line Total": "1000",
                    }
                ],
                "tables": [
                    {
                        "headers": [
                            "Item Description",
                            "Qty",
                            "Unit Price",
                            "Line Total",
                        ],
                        "rows": [
                            {
                                "Item Description": "Support Plan",
                                "Qty": "1",
                                "Unit Price": "1000",
                                "Line Total": "1000",
                            }
                        ],
                    }
                ],
            }

    llm = DummyLLM()
    extractor_service = DocumentExtractor(factory, llm_client=llm)

    doc = tmp_path / "invoice_llm.txt"
    doc.write_text(
        "\n".join(
            [
                "Invoice Number: INV-2000",
                "Vendor: Example Co",
                "Invoice Date: 2024-01-01",
                "Invoice Total: 1000",
            ]
        ),
        encoding="utf-8",
    )

    result = extractor_service.extract(doc)

    assert result.header["invoice_id"] == "INV-2000"
    assert result.header["currency"] == "USD"
    assert result.line_items and result.line_items[0]["line_amount"] == "1000"
    assert result.tables and result.tables[0]["rows"][0]["item_description"] == "Support Plan"
    assert llm.calls
    assert "invoice_id" in llm.calls[0]["field_hints"]["header_fields"]


def test_unclassified_document_defaults_to_contract(tmp_path, extractor):
    doc = tmp_path / "unclassified_note.txt"
    doc.write_text(
        "\n".join(
            [
                "Summary Record",
                "Reference Id: REF-3301",
                "Supplier: Evergreen Holdings",
                "Total Amount: 8750.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, db_path = extractor
    result = extractor_service.extract(doc)

    assert result.document_type == "Contract"
    assert result.header["reference_id"] == "REF-3301"

    row = _read_table(db_path, "proc.raw_contracts")
    payload = json.loads(row["header_json"])
    assert payload["reference_id"] == "REF-3301"


def test_schema_guided_digital_invoice_synonyms(tmp_path, extractor):
    doc = tmp_path / "invoice_schema_synonyms.txt"
    doc.write_text(
        "\n".join(
            [
                "Enterprise Billing Statement",
                "Bill Number - INV-8833",
                "Supplier - Aurora Analytics Ltd",
                "Invoice Date - 2024-07-01",
                "Due Date - 2024-07-31",
                "Currency - eur",
                "Grand Total - 2400.00",
                "Product / Service    Units    Rate    Amount",
                "AI Advisory Hours     10       200.00   2000.00",
                "Implementation Fee    1        400.00   400.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    expected_keys = [
        "invoice_id",
        "supplier_name",
        "invoice_date",
        "due_date",
        "currency",
        "invoice_total_incl_tax",
    ]
    assert result.header["invoice_id"] == "INV-8833"
    assert result.header["currency"] == "EUR"
    assert result.header["invoice_total_incl_tax"] == "2400.00"
    assert _header_f1(result.header, expected_keys) >= 0.95

    expected_lines = [
        {
            "item_description": "AI Advisory Hours",
            "quantity": "10",
            "unit_price": "200.00",
            "line_amount": "2000.00",
        },
        {
            "item_description": "Implementation Fee",
            "quantity": "1",
            "unit_price": "400.00",
            "line_amount": "400.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_amount"
    )
    assert mismatch_rate < 0.02


def test_schema_guided_digital_quote_synonyms(tmp_path, extractor):
    doc = tmp_path / "quote_schema_synonyms.txt"
    doc.write_text(
        "\n".join(
            [
                "Creative Proposal",
                "Quotation Reference: QT-2040",
                "Vendor: Lumen Studio",
                "Quote Date: 2024-08-05",
                "Valid Until: 2024-09-05",
                "Currency: gbp",
                "Quote Total: 4200.00",
                "Service Category    Qty    Unit Rate    Total Amount",
                "Brand Sprint         2     1200.00      2400.00",
                "UX Prototyping       2     900.00       1800.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    expected_keys = [
        "quote_id",
        "supplier_name",
        "quote_date",
        "validity_date",
        "currency",
        "total_amount",
    ]
    assert result.document_type == "Quote"
    assert result.header["quote_id"] == "QT-2040"
    assert result.header["currency"] == "GBP"
    assert _header_f1(result.header, expected_keys) >= 0.95

    expected_lines = [
        {
            "item_description": "Brand Sprint",
            "quantity": "2",
            "unit_price": "1200.00",
            "line_amount": "2400.00",
        },
        {
            "item_description": "UX Prototyping",
            "quantity": "2",
            "unit_price": "900.00",
            "line_amount": "1800.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_amount"
    )
    assert mismatch_rate < 0.02


def test_schema_guided_scanned_purchase_order_synonyms(tmp_path, extractor):
    doc = tmp_path / "po_schema_synonyms.txt"
    doc.write_text(
        "\n".join(
            [
                "PURCHASE ORDER",
                "PO Reference # PO-7785",
                "Vendor - Redwood Logistics",
                "Order Date - 2024-07-12",
                "Expected Delivery - 2024-07-22",
                "Payment Terms - Net 15",
                "Total Order Amount - 5600.00",
                "Item Description    Units    Unit Cost    Line Total",
                "Forklift Rental     1        4500.00      4500.00",
                "Maintenance Package 1        1100.00      1100.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(
        doc, metadata={"ingestion_mode": "scanned", "ocr": True}
    )

    expected_keys = [
        "po_id",
        "supplier_name",
        "order_date",
        "expected_delivery_date",
        "payment_terms",
        "total_amount",
    ]
    assert result.document_type == "Purchase_Order"
    assert result.header["po_id"] == "PO-7785"
    assert result.header["supplier_name"].lower() == "redwood logistics"
    assert _header_f1(result.header, expected_keys) >= 0.95

    expected_lines = [
        {
            "item_description": "Forklift Rental",
            "quantity": "1",
            "unit_price": "4500.00",
            "line_total": "4500.00",
        },
        {
            "item_description": "Maintenance Package",
            "quantity": "1",
            "unit_price": "1100.00",
            "line_total": "1100.00",
        },
    ]
    mismatch_rate = _line_mismatch_rate(
        result.line_items, expected_lines, total_key="line_total"
    )
    assert mismatch_rate < 0.02


def test_schema_guided_scanned_contract_synonyms(tmp_path, extractor):
    doc = tmp_path / "contract_schema_synonyms.txt"
    doc.write_text(
        "\n".join(
            [
                "Master Services Agreement",
                "Contract Ref - C-9901",
                "Agreement Title - Strategic Services Package",
                "Supplier - Northern Apex GmbH",
                "Effective Date - 2024-09-01",
                "Expiration Date - 2025-08-31",
                "Total Contract Value - 125000.00",
                "Payment Terms - Net 45",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(
        doc, metadata={"ingestion_mode": "scanned", "ocr": True}
    )

    expected_keys = [
        "contract_id",
        "contract_title",
        "supplier_name",
        "contract_start_date",
        "contract_end_date",
        "total_contract_value",
        "payment_terms",
    ]
    assert result.document_type == "Contract"
    assert result.header["contract_id"] == "C-9901"
    assert result.header["contract_title"].startswith("Strategic Services")
    assert result.header["contract_start_date"] == "2024-09-01"
    assert result.header["contract_end_date"] == "2025-08-31"
    assert result.header["total_contract_value"] == "125000.00"
    assert result.header["payment_terms"] == "Net 45"
    assert _header_f1(result.header, expected_keys) >= 0.95


def test_digital_contract_ml_metrics(tmp_path, extractor):
    doc = tmp_path / "contract_digital_ml.txt"
    doc.write_text(
        "\n".join(
            [
                "Contract Number: C-9901",
                "Contract Title: Data Center Maintenance",
                "Supplier: Vertex Infrastructure",
                "Contract Start Date: 2024-02-01",
                "Contract End Date: 2025-01-31",
                "Currency: USD",
                "Total Contract Value: 240000.00",
                "Service Description    Quantity    Unit Price    Line Amount",
                "Preventative Maintenance  12   15000.00   180000.00",
                "Emergency Callouts        6    10000.00   60000.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Contract"
    _assert_extraction_metrics(
        result,
        [
            "contract_id",
            "contract_title",
            "supplier_name",
            "contract_start_date",
            "contract_end_date",
            "currency",
            "total_contract_value",
        ],
        [
            {
                "item_description": "Preventative Maintenance",
                "quantity": "12",
                "unit_price": "15000.00",
                "line_amount": "180000.00",
            },
            {
                "item_description": "Emergency Callouts",
                "quantity": "6",
                "unit_price": "10000.00",
                "line_amount": "60000.00",
            },
        ],
        total_key="line_amount",
    )


def test_digital_quote_ml_metrics(tmp_path, extractor):
    doc = tmp_path / "quote_digital_ml.txt"
    doc.write_text(
        "\n".join(
            [
                "Quote Number: Q-5521",
                "Supplier: Aurora Systems",
                "Quote Date: 2024-03-15",
                "Validity Date: 2024-04-15",
                "Currency: EUR",
                "Total Amount: 18250.00",
                "Item Description    Quantity    Unit Price    Amount",
                "Cloud Migration Assessment    1    7500.00    7500.00",
                "Implementation Services      2    5000.00    10000.00",
                "Hypercare Support            1    750.00     750.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "digital"})

    assert result.document_type == "Quote"
    _assert_extraction_metrics(
        result,
        [
            "quote_id",
            "supplier_name",
            "quote_date",
            "validity_date",
            "currency",
            "total_amount",
        ],
        [
            {
                "item_description": "Cloud Migration Assessment",
                "quantity": "1",
                "unit_price": "7500.00",
                "line_amount": "7500.00",
            },
            {
                "item_description": "Implementation Services",
                "quantity": "2",
                "unit_price": "5000.00",
                "line_amount": "10000.00",
            },
            {
                "item_description": "Hypercare Support",
                "quantity": "1",
                "unit_price": "750.00",
                "line_amount": "750.00",
            },
        ],
        total_key="line_amount",
    )


def test_scanned_invoice_ml_metrics(tmp_path, extractor):
    doc = tmp_path / "invoice_scanned_ml.txt"
    doc.write_text(
        "\n".join(
            [
                "INVOICE",
                "Supplier Name",
                "Zenith Lighting Ltd",
                "Invoice Number",
                "INV-8842",
                "Invoice Date",
                "2024-02-12",
                "Due Date",
                "2024-03-13",
                "Currency",
                "EUR",
                "Total Amount Due",
                "3180.00",
                "Item Description    Qty    Unit Price    Line Total",
                "Lighting Panel Set    10    250.00    2500.00",
                "Installation Service  1     680.00    680.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    assert result.document_type == "Invoice"
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
                "item_description": "Lighting Panel Set",
                "quantity": "10",
                "unit_price": "250.00",
                "line_amount": "2500.00",
            },
            {
                "item_description": "Installation Service",
                "quantity": "1",
                "unit_price": "680.00",
                "line_amount": "680.00",
            },
        ],
        total_key="line_amount",
    )


def test_scanned_purchase_order_ml_metrics(tmp_path, extractor):
    doc = tmp_path / "po_scanned_ml.txt"
    doc.write_text(
        "\n".join(
            [
                "PURCHASE ORDER",
                "Supplier",
                "City Works Ltd",
                "PO Number",
                "PO-7789",
                "Order Date",
                "2024-05-01",
                "Expected Delivery Date",
                "2024-05-20",
                "Currency",
                "USD",
                "Total Amount",
                "4575.00",
                "Line Description    Qty    Unit Price    Total",
                "Concrete Mix         15    180.00    2700.00",
                "Steel Rods           25    75.00     1875.00",
            ]
        ),
        encoding="utf-8",
    )

    extractor_service, _ = extractor
    result = extractor_service.extract(doc, metadata={"ingestion_mode": "scanned"})

    assert result.document_type == "Purchase_Order"
    _assert_extraction_metrics(
        result,
        [
            "po_id",
            "supplier_name",
            "order_date",
            "expected_delivery_date",
            "currency",
            "total_amount",
        ],
        [
            {
                "item_description": "Concrete Mix",
                "quantity": "15",
                "unit_price": "180.00",
                "line_total": "2700.00",
            },
            {
                "item_description": "Steel Rods",
                "quantity": "25",
                "unit_price": "75.00",
                "line_total": "1875.00",
            },
        ],
        total_key="line_total",
    )
