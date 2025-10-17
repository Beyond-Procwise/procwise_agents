from datetime import datetime
from typing import Dict, Optional
import uuid

from email_thread import EmailThread
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

        # CRITICAL: Analyze supplier's response
        supplier_response_text = supplier_latest_response.get('content', '') if supplier_latest_response else ''
        response_analysis = await self._analyze_supplier_response(supplier_response_text, round_num)

        # Generate unique action ID
        action_id = f"NEG-R{round_num}-{thread.supplier_unique_id}-{uuid.uuid4().hex[:6].upper()}"

        prompt = f"""
You are expert procurement negotiator in Round {round_num} of 3.

STRATEGY: {strategy.get('strategy', '')}
TONE: {strategy.get('tone', '')}

COMPLETE EMAIL THREAD:
{thread_context}

SUPPLIER'S LATEST RESPONSE:
{supplier_response_text}

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

        return {
            "action_id": action_id,
            "supplier_id": supplier['id'],
            "round": round_num,
            "content": content,
            "thread_id": thread.thread_id,
            "type": f"negotiation_round_{round_num}",
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": True
        }

    async def _analyze_supplier_response(self, response_text: str, round_num: int) -> str:
        """CRITICAL: Analyze supplier response to extract context"""
        analysis_prompt = f"""
Analyze supplier response for Round {round_num} negotiation:

RESPONSE:
{response_text}

Extract:
1. Pricing: Any numbers, costs mentioned
2. Payment Terms: Any terms mentioned
3. Concerns/Objections: Issues they raised
4. Questions: What they're asking
5. Concessions: Flexibility shown
6. Tone: Receptive/defensive/eager
7. Key Dates: Any timelines mentioned

Provide structured analysis for response strategy.
"""
        return await self._call_ollama(analysis_prompt)

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate with OLLAMA API
        return "[Generated content]"
