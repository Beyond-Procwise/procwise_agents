from datetime import datetime
from typing import Dict
import uuid

from workflow_context_manager import WorkflowContextManager


class EmailDraftingAgent:
    """Generates emails with hidden identifiers embedded"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "qwen3:30b"

    async def generate_initial_email(self, supplier: Dict, hidden_identifier: str) -> Dict:
        action_id = f"EMAIL-{int(datetime.now().timestamp())}-{uuid.uuid4().hex[:6].upper()}"

        prompt = f"""
Generate professional RFQ email for:
Supplier: {supplier['name']}
Requirements: {supplier.get('requirements', 'Standard procurement')}

CRITICAL: Embed this tracking identifier in email footer:
Reference: {hidden_identifier}

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
