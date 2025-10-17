"""Supplier interaction collector that reads from the authoritative response store."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Set

from repositories import supplier_responses_repo


class SupplierInteractionAgent:
    """Read-only collector that inspects supplier responses stored in the database."""

    def __init__(self, response_repo=supplier_responses_repo):
        self._repo = response_repo
        self._cached_responses: Dict[str, Dict[str, Any]] = {}
        self._observed_hidden_ids: Set[str] = set()

    async def collect_response(
        self, supplier_id: str, hidden_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch the latest response for a supplier without blocking workflow timing."""

        record = await self._fetch_latest_response(hidden_identifier)
        if record is None:
            return None

        payload = {
            "supplier_id": supplier_id,
            "hidden_identifier": hidden_identifier,
            "content": record.get("body_text") or record.get("response_text"),
            "received_at": self._coerce_timestamp(record.get("received_at")),
            "raw": record,
        }
        self._cached_responses[hidden_identifier] = payload
        self._observed_hidden_ids.add(hidden_identifier)
        return payload

    async def refresh_received_from_db(
        self, expected_hidden_ids: Iterable[str]
    ) -> Dict[str, Set[str]]:
        """Refresh the cache of observed hidden identifiers from the database."""

        expected = {hid for hid in expected_hidden_ids if hid}
        received = await self._list_hidden_identifiers(
            expected if expected else None
        )
        missing = expected - received if expected else set()
        self._observed_hidden_ids |= received
        return {"expected": expected, "received": received, "missing": missing}

    def all_responses_collected(self, expected_hidden_ids: Iterable[str]) -> bool:
        """Check if every expected hidden identifier has been observed."""

        expected = {hid for hid in expected_hidden_ids if hid}
        return expected.issubset(self._observed_hidden_ids)

    async def _list_hidden_identifiers(
        self, expected: Optional[Set[str]]
    ) -> Set[str]:
        loop = asyncio.get_running_loop()
        if expected is None:
            return await loop.run_in_executor(
                None, self._repo.distinct_hidden_identifiers
            )
        return await loop.run_in_executor(
            None, self._repo.distinct_hidden_identifiers, expected
        )

    async def _fetch_latest_response(
        self, hidden_identifier: str
    ) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._repo.fetch_latest_by_hidden_identifier, hidden_identifier
        )

    @staticmethod
    def _coerce_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now(timezone.utc).isoformat()
