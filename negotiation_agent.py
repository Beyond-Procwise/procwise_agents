from typing import Any, Dict, Optional

from email_thread import EmailThread, make_action_id
from workflow_context_manager import WorkflowContextManager


class NegotiationAgent:
    """Context-driven intelligent negotiation"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "mixtral:8x7b"

    async def generate_negotiation_response(
        self, supplier: Dict, thread: EmailThread,
        round_num: int, supplier_latest_response: Optional[Dict]
    ) -> Dict:
        """CRITICAL: Analyzes actual supplier response before generating"""

        # Get strategy for this round
        playbook = self.context.embeddings.get('negotiation_playbook', {})
        strategy = playbook.get(f'round_{round_num}', {})

        # Get full thread context
        thread_context = thread.get_full_thread()

        # Analyze supplier's response for signal extraction
        response_analysis = await self._analyze_supplier_response(
            supplier, supplier_latest_response
        )

        # Generate unique action ID using shared helper
        action_id = make_action_id(round_num, thread.supplier_unique_id)

        prompt = f"""
You are expert procurement negotiator in Round {round_num} of 3.

STRATEGY: {strategy.get('strategy', '')}
TONE: {strategy.get('tone', '')}

COMPLETE EMAIL THREAD:
{thread_context}

SUPPLIER'S LATEST RESPONSE:
{(supplier_latest_response or {}).get('content', '')}

RESPONSE ANALYSIS:
{response_analysis}

Generate contextually intelligent negotiation email that:
1. Directly addresses supplier's specific points (pricing, concerns, questions)
2. Applies Round {round_num} strategy
3. Shows deep understanding of their position
4. Maintains thread continuity
5. Provides clear next steps
6. Is human-like and creative

This will be reviewed by human before sending.
"""

        content = await self._call_ollama(prompt)

        entry = thread.add_message(
            f"negotiation_round_{round_num}",
            content,
            action_id,
            round_num=round_num,
        )

        return {
            "action_id": action_id,
            "supplier_id": supplier['id'],
            "round": round_num,
            "content": content,
            "thread_id": thread.thread_id,
            "type": f"negotiation_round_{round_num}",
            "timestamp": entry["timestamp"],
            "requires_human_review": True
        }

    async def _analyze_supplier_response(
        self,
        supplier: Dict[str, Any],
        supplier_latest_response: Optional[Dict[str, Any]],
    ) -> str:
        """Analyze supplier response to extract structured negotiation signals."""

        response_text = (supplier_latest_response or {}).get("content", "")
        response_metadata = {
            "supplier_id": supplier.get("id"),
            "supplier_name": supplier.get("name"),
            "received_at": (supplier_latest_response or {}).get("received_at"),
        }
        analysis_prompt = f"""
Analyze the supplier's latest response for negotiation planning.

SUPPLIER CONTEXT:
{response_metadata}

RESPONSE TEXT:
{response_text}

Extract and summarise:
1. Pricing or commercial figures mentioned.
2. Any delivery, service level, or lead time commitments.
3. Stated constraints, risks, or objections.
4. Questions directed at us that require an answer.
5. Evidence of flexibility or concessions.
6. Tone and intent (collaborative, hesitant, defensive, etc.).
7. Suggested next steps or deadlines from the supplier.

Return a concise analysis that can be fed into the negotiation LLM.
"""
        return await self._call_ollama(analysis_prompt)

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate with OLLAMA API
        return "[Generated content]"
