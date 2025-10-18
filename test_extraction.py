#!/usr/bin/env python3
"""
Test script for data extraction improvements.
Run: python test_extraction.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

from agents.data_extraction_agent import DataExtractionAgent
from agents.base_agent import AgentNick


SAMPLE_DOCS_DIR = Path("tests/sample_docs")


def load_expected_results() -> Dict[str, Dict[str, Any]]:
    """Load expected extraction results for test documents."""
    return {
        "INV600264": {
            "invoice_id": "INV600264",
            "invoice_date": "2019-06-22",
            "po_id": "PO502004",
            "vendor_name": "City of Newport",
            "buyer_id": "Assurity Ltd",
            "currency": "GBP",
            "invoice_amount": 8335.0,
            "tax_percent": 20.0,
            "tax_amount": 1667.0,
            "invoice_total_incl_tax": 10002.0,
            "line_items_count": 1,
        },
        "PO502004": {
            "po_id": "PO502004",
            "order_date": "2019-07-08",
            "quote_id": "QUT599390",
            "vendor_name": "Assurity Ltd",
            "buyer_id": "City of Newport",
            "currency": "GBP",
            "total_amount": 100000.0,
            "line_items_count": 2,
        },
        "QUT599390": {
            "quote_id": "QUT599390",
            "quote_date": "2019-07-01",
            "vendor_name": "City of Newport",
            "buyer_id": "Assurity Ltd",
            "currency": "GBP",
            "total_amount": 100000.0,
            "line_items_count": 2,
        },
    }


def compare_values(expected: Any, actual: Any, field_name: str) -> tuple[bool, str]:
    """Compare expected vs actual value with tolerance for floats."""
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, f"Expected {expected}, got {actual}"
    if isinstance(expected, (int, float)):
        try:
            actual_num = float(actual)
        except (ValueError, TypeError):
            return False, f"Expected number {expected}, got non-numeric {actual}"
        diff = abs(float(expected) - actual_num)
        if diff > 0.02:
            return False, f"Expected {expected}, got {actual_num} (diff: {diff:.2f})"
        return True, ""
    if isinstance(expected, str):
        expected_norm = expected.strip().upper()
        actual_norm = str(actual).strip().upper()
        if expected_norm == actual_norm:
            return True, ""
        return False, f"Expected '{expected}', got '{actual}'"
    if expected == actual:
        return True, ""
    return False, f"Expected {expected}, got {actual}"


def test_document(agent: DataExtractionAgent, file_path: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Test extraction on a single document."""
    print("\n" + "=" * 60)
    print(f"Testing: {file_path.name}")
    print("=" * 60)

    result = {
        "file": str(file_path),
        "passed": True,
        "errors": [],
        "warnings": [],
        "confidence": 0.0,
        "fields_correct": 0,
        "fields_total": 0,
    }

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_bytes = file_path.read_bytes()
        text_bundle = agent._extract_text(file_bytes, str(file_path))
        text = text_bundle.full_text
        doc_type = agent._classify_doc_type(text)
        print(f"Document type: {doc_type}")

        extraction = agent._extract_structured_data(text, file_bytes, doc_type)
        header = extraction.header
        line_items = extraction.line_items
        validation = header.get("_validation", {})
        result["confidence"] = validation.get("confidence_score", 0.0)

        for field, expected_value in expected.items():
            actual_value = len(line_items) if field == "line_items_count" else header.get(field)
            result["fields_total"] += 1
            matches, error = compare_values(expected_value, actual_value, field)
            if matches:
                result["fields_correct"] += 1
                print(f"  ✅ {field}: {actual_value}")
            else:
                result["passed"] = False
                result["errors"].append(f"{field}: {error}")
                print(f"  ❌ {field}: {error}")

        for error in validation.get("errors", []):
            result["warnings"].append(f"Validation: {error}")
            print(f"  ⚠️  Validation error: {error}")

        print(f"\nLine items extracted: {len(line_items)}")
        for idx, item in enumerate(line_items, 1):
            desc = str(item.get("item_description", "N/A"))[:80]
            amount = item.get("line_amount") or item.get("line_total") or 0
            try:
                amount_float = float(amount)
            except (ValueError, TypeError):
                amount_float = 0.0
            print(f"  {idx}. {desc}... (£{amount_float:,.2f})")

    except Exception as exc:  # pragma: no cover - diagnostic output
        result["passed"] = False
        result["errors"].append(f"Exception: {exc}")
        print(f"  ❌ Exception: {exc}")

    return result


def main() -> int:
    print("\n" + "=" * 60)
    print("DATA EXTRACTION AGENT - TEST SUITE")
    print("=" * 60)

    agent_nick = AgentNick()
    agent = DataExtractionAgent(agent_nick)
    expected_results = load_expected_results()

    if not SAMPLE_DOCS_DIR.exists():
        print(f"Sample documents directory not found: {SAMPLE_DOCS_DIR}")
        return 1

    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    for doc_id, expectations in expected_results.items():
        pdf_path = SAMPLE_DOCS_DIR / f"{doc_id}.pdf"
        result = test_document(agent, pdf_path, expectations)
        summary["total"] += 1
        summary["details"].append(result)
        if result["passed"]:
            summary["passed"] += 1
        else:
            summary["failed"] += 1

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(json.dumps(summary, indent=2))

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
