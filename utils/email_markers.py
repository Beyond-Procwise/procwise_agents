"""Utilities for embedding invisible procurement identifiers in email bodies."""

from __future__ import annotations

import re
import uuid
from typing import Dict, Optional, Tuple

_MODERN_MARKER_PATTERN = re.compile(
    r"<!--\s*PROCWISE_MARKER:(.*?)-->", re.IGNORECASE | re.DOTALL
)
_LEGACY_MARKER_PATTERN = re.compile(
    r"<!--\s*(?:PROCWISE:(?:UID|RFQ_ID)[^>]*|RFQ-ID[^>]*)-->",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_KEY_VALUE = re.compile(
    r"(PROCWISE:UID|RFQ_ID|RFQ-ID|UID|TOKEN|RUN_ID|RUN|SUPPLIER)\s*[:=]\s*([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _locate_marker_comment(text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not isinstance(text, str):
        return None, None, None
    modern = _MODERN_MARKER_PATTERN.search(text)
    if modern:
        return modern.group(0).strip(), modern.start(), modern.end()
    legacy = _LEGACY_MARKER_PATTERN.search(text)
    if legacy:
        return legacy.group(0).strip(), legacy.start(), legacy.end()
    return None, None, None


def _parse_marker_metadata(comment: Optional[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not comment:
        return metadata

    modern = _MODERN_MARKER_PATTERN.search(comment)
    if modern:
        payload = modern.group(1) or ""
        for part in payload.split("|"):
            if not part or ":" not in part:
                continue
            key, value = part.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
        return metadata

    for key, value in _LEGACY_KEY_VALUE.findall(comment):
        key_lower = key.lower()
        cleaned_value = value.strip()
        if key_lower in {"procwise:uid", "rfq_id", "rfq-id", "uid"}:
            metadata["tracking"] = cleaned_value
        elif key_lower == "token":
            metadata["token"] = cleaned_value
        elif key_lower in {"run_id", "run"}:
            metadata["run"] = cleaned_value
        elif key_lower in {"workflow", "workflow_id"}:
            metadata["workflow"] = cleaned_value
        elif key_lower == "supplier":
            metadata["supplier"] = cleaned_value
    return metadata


def _normalise_supplier(supplier_id: Optional[str]) -> str:
    if not supplier_id:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(supplier_id))
    return cleaned


def _strip_visible_identifiers(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"(?i)\b(RFQ|rfq)[\s:-]*\d{8}[\s:-]*[A-Za-z0-9]{8}\b",
        "",
        text,
    )
    cleaned = re.sub(
        r"(?i)\b(UID|uid)[\s:-]*[A-Za-z0-9-]{12,}\b",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\bPROC-WF-[A-F0-9]{6,}\b",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(workflow|WF)[\s:-]*[A-Za-z0-9-]{8,}\b",
        "",
        cleaned,
    )
    return cleaned


def split_hidden_marker(body: str) -> Tuple[Optional[str], str]:
    """Split ``body`` into an optional hidden marker comment and the remainder."""

    if not isinstance(body, str):
        return None, ""

    comment, start, end = _locate_marker_comment(body)
    if comment is None or start is None or end is None:
        return None, body

    remainder = f"{body[:start]}{body[end:]}"
    remainder = remainder.rstrip()
    remainder = re.sub(r"\n{3,}", "\n\n", remainder)
    return comment, remainder


def extract_rfq_id(comment: Optional[str]) -> Optional[str]:
    """Return the unique identifier embedded in ``comment`` if present."""

    metadata = _parse_marker_metadata(comment)
    tracking = metadata.get("tracking") or metadata.get("procwise:uid")
    return tracking


def ensure_hidden_marker(
    *,
    rfq_id: Optional[str],  # Deprecated
    supplier_id: Optional[str] = None,
    comment: Optional[str] = None,
    token: Optional[str] = None,
    run_id: Optional[str] = None,
    unique_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return a hidden marker comment anchored on ``unique_id`` and its tracking token."""

    base_text = comment or ""
    marked_body, marker_token = attach_hidden_marker(
        base_text,
        supplier_id=supplier_id,
        unique_id=unique_id,
        token=token,
        run_id=run_id,
    )
    new_comment, _ = split_hidden_marker(marked_body)
    return new_comment, marker_token


def attach_hidden_marker(
    body: str,
    *,
    rfq_id: Optional[str] = None,  # Deprecated but kept for compatibility
    supplier_id: Optional[str] = None,
    unique_id: Optional[str] = None,
    token: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Attach hidden tracking marker ensuring it never appears to recipients."""

    base_text = body or ""
    existing_comment, _, _ = _locate_marker_comment(base_text)
    existing_metadata = _parse_marker_metadata(existing_comment)

    base_text = re.sub(r"<!--\s*PROCWISE_MARKER:.*?-->", "", base_text, flags=re.DOTALL | re.IGNORECASE)

    tracking_id = unique_id or existing_metadata.get("tracking")
    if not tracking_id:
        tracking_id = f"PROC-WF-{uuid.uuid4().hex[:12].upper()}"

    token_value = token or existing_metadata.get("token")
    if not token_value:
        token_value = uuid.uuid4().hex[:12].upper()

    run_value = run_id or existing_metadata.get("run")

    supplier_segment = _normalise_supplier(supplier_id) or existing_metadata.get("supplier", "")

    metadata_parts = [f"TRACKING:{tracking_id}"]

    if supplier_segment:
        metadata_parts.append(f"SUPPLIER:{supplier_segment}")

    if token_value:
        metadata_parts.append(f"TOKEN:{token_value}")

    if run_value:
        metadata_parts.append(f"RUN:{run_value}")

    metadata_string = "|".join(metadata_parts)
    hidden_marker = f"<!-- PROCWISE_MARKER:{metadata_string} -->"

    clean_body = _strip_visible_identifiers(base_text)
    clean_body = clean_body.strip()

    if clean_body:
        marked_body = f"{clean_body}\n\n{hidden_marker}"
    else:
        marked_body = hidden_marker

    return marked_body, token_value


def extract_marker_token(comment: Optional[str]) -> Optional[str]:
    """Return the dispatch tracking token stored in ``comment`` if present."""

    metadata = _parse_marker_metadata(comment)
    return metadata.get("token")


def extract_run_id(comment: Optional[str]) -> Optional[str]:
    """Return the dispatch run identifier embedded in ``comment`` if present."""

    metadata = _parse_marker_metadata(comment)
    return metadata.get("run")


def extract_supplier_id(comment: Optional[str]) -> Optional[str]:
    """Return the supplier identifier embedded in ``comment`` if present."""

    metadata = _parse_marker_metadata(comment)
    return metadata.get("supplier")
