"""Utilities for embedding and extracting ProcWise email tracking markers."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


_TRACKING_PATTERN = re.compile(r"<!--\s*PROCWISE:(\{.*?\})\s*-->", re.IGNORECASE | re.DOTALL)


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

