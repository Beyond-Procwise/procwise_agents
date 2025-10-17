from datetime import datetime
from typing import Dict
import uuid

from email_thread import make_action_id
from workflow_context_manager import WorkflowContextManager


class EmailDraftingAgent:
    """Generates emails with hidden identifiers embedded"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "qwen3:30b"

    async def generate_initial_email(
        self, supplier: Dict, hidden_identifier: str, **kwargs
    ) -> Dict:
        supplier_unique_id = (
            kwargs.get("supplier_unique_id")
            or supplier.get("supplier_unique_id")
            or f"SUP-{uuid.uuid4().hex[:10].upper()}"
        )
        action_id = make_action_id(0, supplier_unique_id)

        prompt = f"""
Generate professional RFQ email for:
Supplier: {supplier['name']}
Requirements: {supplier.get('requirements', 'Standard procurement')}

CRITICAL: Embed this tracking identifier in email footer:
Reference: {hidden_identifier}

Ensure the identifier appears verbatim in the email body so it survives forwarding chains.

Make it professional, clear, and engaging.
"""

        email_content = await self._call_ollama(prompt)

        return {
            "action_id": action_id,
            "supplier_id": supplier['id'],
            "hidden_identifier": hidden_identifier,
            "content": email_content,
            "type": "initial_email",
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": True
        }

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate with actual OLLAMA API
        return f"[Generated email content]"
