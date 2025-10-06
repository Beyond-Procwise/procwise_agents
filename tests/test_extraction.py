import json
import os
import sqlite3
import sys
from pathlib import Path

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
    assert result.schema_reference["document_type"] == "Invoice"
    assert "invoice_id" in result.schema_reference["header_fields"]
    assert {"item_description", "qty", "unit_price", "line_total"}.issubset(
        set(result.schema_reference["line_item_fields"])
    )

    row = _read_table(db_path, "proc.raw_invoice")
    assert row["document_type"] == "Invoice"
    header = json.loads(row["header_json"])
    assert header["invoice_id"] == "INV-1001"
    assert header["supplier_name"].lower() == "acme components"
    line_items = json.loads(row["line_items_json"])
    assert line_items[0]["item_description"].lower() == "laptop pro 15"


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
    assert result.schema_reference["document_type"] == "Purchase_Order"
    assert "po_id" in result.schema_reference["header_fields"]

    row = _read_table(db_path, "proc.raw_purchase_order")
    payload = json.loads(row["header_json"])
    assert payload["po_id"] == "PO-9001"


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
    assert result.line_items and result.line_items[0]["line_total"] == "1000"
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

