"""Helpers for persisting supplier responses consistently."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from repositories import supplier_response_repo
from repositories.supplier_response_repo import SupplierResponseRow


def store_supplier_response(
    *,
    workflow_id: str,
    unique_id: str,
    supplier_id: Optional[str],
    supplier_email: Optional[str],
    body_text: Optional[str],
    body_html: Optional[str],
    received_at: datetime,
    response_time: Optional[Decimal],
    response_message_id: Optional[str],
    response_subject: Optional[str],
    response_from: Optional[str],
    original_message_id: Optional[str],
    original_subject: Optional[str],
    match_confidence: Optional[Decimal],
    price: Optional[Decimal] = None,
    lead_time: Optional[int] = None,
    processed: bool = False,
) -> SupplierResponseRow:
    """Persist and return a :class:`SupplierResponseRow` instance.

    The helper normalises missing text/html payloads so callers can supply
    whichever representation is available while guaranteeing both columns are
    populated for downstream processing.
    """

    text_value = (body_text or "").strip()
    html_value = body_html.strip() if isinstance(body_html, str) else body_html

    if not text_value and html_value:
        text_value = html_value

    row = SupplierResponseRow(
        workflow_id=workflow_id,
        unique_id=unique_id,
        supplier_id=supplier_id,
        supplier_email=supplier_email,
        response_text=text_value,
        response_body=html_value or text_value,
        received_time=received_at,
        response_time=response_time,
        response_message_id=response_message_id,
        response_subject=response_subject,
        response_from=response_from,
        original_message_id=original_message_id,
        original_subject=original_subject,
        match_confidence=match_confidence,
        price=price,
        lead_time=lead_time,
        processed=processed,
    )

    supplier_response_repo.insert_response(row)
    return row
