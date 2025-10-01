from __future__ import annotations

import secrets
from datetime import datetime, timezone


def generate_rfq_id() -> str:
    """Return a globally unique RFQ identifier for outbound correspondence."""

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"RFQ-{today}-{secrets.token_hex(4)}".upper()
