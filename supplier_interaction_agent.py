from datetime import datetime
from typing import Dict, Optional, List


class SupplierInteractionAgent:
    """Collects responses ONLY - NO workflow control"""

    def __init__(self):
        self.responses_collected = {}

    async def collect_response(self, supplier_id: str, hidden_identifier: str) -> Optional[Dict]:
        """Collect response - does NOT control workflow progression"""
        response = await self._monitor_email_response(hidden_identifier)

        if response:
            self.responses_collected[supplier_id] = {
                "content": response,
                "received_at": datetime.now().isoformat(),
                "hidden_identifier": hidden_identifier
            }
            return self.responses_collected[supplier_id]

        return None

    async def _monitor_email_response(self, hidden_identifier: str) -> Optional[str]:
        # TODO: Integrate with email API
        return None

    def all_responses_collected(self, expected_supplier_ids: List[str]) -> bool:
        return all(sid in self.responses_collected for sid in expected_supplier_ids)
