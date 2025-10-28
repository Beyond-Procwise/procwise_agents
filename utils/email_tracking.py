"""Utilities for embedding and extracting ProcWise email tracking markers."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import string
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple


_TRACKING_PATTERN = re.compile(r"<!--\s*PROCWISE:(\{.*?\})\s*-->", re.IGNORECASE | re.DOTALL)
_MARKER_COMMENT_PATTERN = re.compile(
    r"<!--\s*PROCWISE_MARKER:(.*?)-->", re.IGNORECASE | re.DOTALL
)
_SIMPLE_UID_PATTERN = re.compile(r"<!--\s*PROCWISE:UID=([A-Za-z0-9-]+)\s*-->", re.IGNORECASE)
_HIDDEN_SPAN_PATTERN = re.compile(
    r"<span\s+style=\"display:?none;?\">\s*PROCWISE:UID=([A-Za-z0-9-]+)\s*</span>",
    re.IGNORECASE,
)

_UID_PREFIX = "PROC-WF-"
_UID_ALPHABET = "0123456789ABCDEF"
_DEFAULT_UID_LENGTH = 12


@dataclass(frozen=True)
class EmailTrackingMetadata:
    """Structured metadata embedded within outbound supplier emails."""

    workflow_id: str
    unique_id: str
    supplier_id: Optional[str] = None
    token: Optional[str] = None
    run_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "unique_id": self.unique_id,
        }
        if self.supplier_id:
            data["supplier_id"] = self.supplier_id
        if self.token:
            data["token"] = self.token
        if self.run_id:
            data["run_id"] = self.run_id
        return data


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_marker_payload(payload: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for part in payload.split("|"):
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def build_tracking_comment(
    *,
    workflow_id: str,
    unique_id: Optional[str] = None,
    supplier_id: Optional[str] = None,
    token: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[str, EmailTrackingMetadata]:
    """Create a HTML comment that stores the tracking metadata."""

    if not workflow_id:
        raise ValueError("workflow_id is required for tracking markers")

    workflow = _normalise_identifier(workflow_id)
    uid = _normalise_identifier(unique_id) or uuid.uuid4().hex
    supplier = _normalise_identifier(supplier_id)
    marker_token = _normalise_identifier(token) or uuid.uuid4().hex
    run_identifier = _normalise_identifier(run_id) or marker_token

    metadata = EmailTrackingMetadata(
        workflow_id=workflow,
        unique_id=uid,
        supplier_id=supplier,
        token=marker_token,
        run_id=run_identifier,
    )

    comment = f"<!-- PROCWISE:{json.dumps(metadata.as_dict(), separators=(',', ':'))} -->"
    return comment, metadata


def ensure_tracking_prefix(body: str, comment: str) -> str:
    """Prefix ``body`` with the tracking ``comment`` if not already present."""

    if not body:
        return comment

    if body.lstrip().startswith("<!--") and _TRACKING_PATTERN.search(body):
        return body

    return f"{comment}\n{body}" if body else comment


def extract_tracking_metadata(text: Optional[str]) -> Optional[EmailTrackingMetadata]:
    """Parse tracking metadata from a comment embedded within ``text``."""

    if not text:
        return None

    marker_match = _MARKER_COMMENT_PATTERN.search(text)
    if marker_match:
        payload = marker_match.group(1) or ""
        fields = _parse_marker_payload(payload)
        workflow = _normalise_identifier(fields.get("workflow"))
        unique = _normalise_identifier(fields.get("tracking") or fields.get("procwise:uid"))
        if not unique:
            return None
        supplier = _normalise_identifier(fields.get("supplier"))
        token = _normalise_identifier(fields.get("token"))
        run_id = _normalise_identifier(fields.get("run"))
        workflow_value = workflow or ""
        return EmailTrackingMetadata(
            workflow_id=workflow_value,
            unique_id=unique,
            supplier_id=supplier,
            token=token,
            run_id=run_id,
        )

    match = _TRACKING_PATTERN.search(text)
    if not match:
        return None

    try:
        payload = json.loads(match.group(1))
    except Exception:
        return None

    workflow = _normalise_identifier(payload.get("workflow_id"))
    unique = _normalise_identifier(payload.get("unique_id"))
    if not workflow or not unique:
        return None

    return EmailTrackingMetadata(
        workflow_id=workflow,
        unique_id=unique,
        supplier_id=_normalise_identifier(payload.get("supplier_id")),
        token=_normalise_identifier(payload.get("token")),
        run_id=_normalise_identifier(payload.get("run_id")),
    )


def strip_tracking_comment(body: str) -> Tuple[Optional[EmailTrackingMetadata], str]:
    """Return the tracking metadata (if any) and the body without the comment."""

    if not body:
        return None, ""

    match = _TRACKING_PATTERN.match(body.strip())
    if not match:
        return extract_tracking_metadata(body), body

    metadata = extract_tracking_metadata(match.group(0))
    remainder = body[match.end() :].lstrip("\n")
    return metadata, remainder


def _canonicalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def generate_unique_email_id(workflow_id: str, supplier_id: Optional[str] = None) -> str:
    """Create a deterministic-looking unique identifier for outbound emails.

    The identifier is always prefixed with ``PROC-WF-`` followed by a
    12-character uppercase hexadecimal token.  Workflow and supplier entropy
    are folded into the hash input to avoid sequential identifiers while
    keeping the value opaque to external recipients.
    """

    workflow = _canonicalise_identifier(workflow_id)
    if not workflow:
        raise ValueError("workflow_id is required to generate a unique email id")

    supplier = _canonicalise_identifier(supplier_id) or "anon"
    random_bits = secrets.token_hex(12)
    digest_source = f"{workflow}|{supplier}|{random_bits}|{datetime.utcnow().isoformat()}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest().upper()

    token = "".join(ch for ch in digest if ch in _UID_ALPHABET)
    if len(token) < _DEFAULT_UID_LENGTH:
        token += "".join(secrets.choice(_UID_ALPHABET) for _ in range(_DEFAULT_UID_LENGTH - len(token)))
    return f"{_UID_PREFIX}{token[:_DEFAULT_UID_LENGTH]}"


def embed_unique_id_in_email_body(body: Optional[str], unique_id: str) -> str:
    """Embed ``unique_id`` in ``body`` using a hidden HTML comment.

    Existing ProcWise tracking comments are preserved.  When a JSON tracking
    comment is already present we simply prepend the new UID marker to ensure
    EmailWatcher V2 can recover the identifier without breaking compatibility
    with earlier tooling.
    """

    identifier = _canonicalise_identifier(unique_id)
    if not identifier:
        raise ValueError("unique_id is required to embed in the email body")

    comment = f"<!-- PROCWISE:UID={identifier} -->"
    text = body or ""

    if _SIMPLE_UID_PATTERN.search(text):
        # Already embedded with the new style marker.
        return text

    cleaned = text.strip()
    if cleaned.startswith("<!--") and _TRACKING_PATTERN.search(cleaned):
        # Legacy JSON marker present – prefix the new comment to maintain
        # backwards compatibility.
        return f"{comment}\n{text}" if text else comment

    return f"{comment}\n{text}" if text else comment


def extract_unique_id_from_body(body: Optional[str]) -> Optional[str]:
    """Extract a previously embedded unique identifier from ``body``."""

    if not body:
        return None

    match = _SIMPLE_UID_PATTERN.search(body)
    if match:
        return match.group(1).strip()

    span_match = _HIDDEN_SPAN_PATTERN.search(body)
    if span_match:
        return span_match.group(1).strip()

    metadata = extract_tracking_metadata(body)
    return metadata.unique_id if metadata else None


def extract_unique_id_from_headers(headers: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the canonical unique identifier from email ``headers`` if present."""

    if not headers:
        return None

    candidate_keys = (
        "X-ProcWise-Unique-ID",
        "X-Procwise-Unique-Id",
        "X-Procwise-Unique-ID",
        "X-Procwise-Uid",
    )

    for key in candidate_keys:
        value = headers.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
        else:
            text = str(value).strip()
            if text:
                return text

    return None

