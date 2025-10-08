"""Utilities for embedding invisible procurement identifiers in email bodies."""

from __future__ import annotations

import re
import uuid
from typing import Optional, Tuple

_MARKER_DETECTION = re.compile(
    r"(?:PROCWISE:RFQ_ID|RFQ-ID)\s*[:=]\s*([A-Za-z0-9_-]+)", re.IGNORECASE
)
_TOKEN_DETECTION = re.compile(r"TOKEN\s*[:=]\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_RUN_ID_DETECTION = re.compile(r"RUN_ID\s*[:=]\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
def split_hidden_marker(body: str) -> Tuple[Optional[str], str]:
    """Split ``body`` into an optional hidden marker comment and the remainder."""

    if not isinstance(body, str):
        return None, ""

    match = re.match(r"\s*(<!--.*?-->)", body, flags=re.DOTALL)
    if not match:
        return None, body

    comment = match.group(1).strip()
    if not _MARKER_DETECTION.search(comment):
        return None, body

    remainder = body[match.end() :].lstrip("\n")
    return comment, remainder


def extract_rfq_id(comment: Optional[str]) -> Optional[str]:
    """Return the RFQ identifier embedded in ``comment`` if present."""

    if not comment:
        return None
    match = _MARKER_DETECTION.search(comment)
    if match:
        return match.group(1).strip()
    return None


def _normalise_supplier(supplier_id: Optional[str]) -> str:
    if not supplier_id:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(supplier_id))
    return cleaned


def ensure_hidden_marker(
    *,
    rfq_id: Optional[str],
    supplier_id: Optional[str] = None,
    comment: Optional[str] = None,
    token: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return a hidden marker comment for ``rfq_id`` and its tracking token."""

    if not rfq_id:
        return comment if comment else None, token

    existing_token = None
    existing_run_id = None
    if comment:
        token_match = _TOKEN_DETECTION.search(comment)
        if token_match:
            existing_token = token_match.group(1).strip()
        run_match = _RUN_ID_DETECTION.search(comment)
        if run_match:
            existing_run_id = run_match.group(1).strip()

    marker_token = token or existing_token or uuid.uuid4().hex
    run_identifier = run_id or existing_run_id or marker_token
    supplier_segment = _normalise_supplier(supplier_id)

    segments = [f"PROCWISE:RFQ_ID={rfq_id}"]
    if supplier_segment:
        segments.append(f"SUPPLIER={supplier_segment}")
    segments.append(f"TOKEN={marker_token}")
    if run_identifier:
        segments.append(f"RUN_ID={run_identifier}")

    return f"<!-- {';'.join(segments)} -->", marker_token


def attach_hidden_marker(
    body: str,
    *,
    rfq_id: Optional[str],
    supplier_id: Optional[str] = None,
    token: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Ensure ``body`` includes an invisible comment identifying the RFQ."""

    comment, remainder = split_hidden_marker(body)
    updated_comment, marker_token = ensure_hidden_marker(
        rfq_id=rfq_id,
        supplier_id=supplier_id,
        comment=comment,
        token=token,
        run_id=run_id,
    )

    if updated_comment:
        if remainder:
            combined = f"{updated_comment}\n{remainder}".strip()
        else:
            combined = updated_comment
        return combined, marker_token

    return body, marker_token


def extract_marker_token(comment: Optional[str]) -> Optional[str]:
    """Return the dispatch tracking token stored in ``comment`` if present."""

    if not comment:
        return None
    match = _TOKEN_DETECTION.search(comment)
    if match:
        return match.group(1).strip()
    return None


def extract_run_id(comment: Optional[str]) -> Optional[str]:
    """Return the dispatch run identifier embedded in ``comment`` if present."""

    if not comment:
        return None
    match = _RUN_ID_DETECTION.search(comment)
    if match:
        return match.group(1).strip()
    return None

