from typing import Dict, List

from email_thread import EmailThread
from workflow_context_manager import WorkflowContextManager


class QuoteEvaluationAgent:
    """Evaluate final quotes after 3 negotiation rounds"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "qwen3:30b"

    async def evaluate_quotes(self, threads: Dict[str, EmailThread]) -> Dict:
        evaluations = {}

        for supplier_id, thread in threads.items():
            eval = await self._evaluate_single_quote(thread)
            evaluations[supplier_id] = eval

        ranked = self._rank_suppliers(evaluations)

        return {
            "evaluations": evaluations,
            "ranking": ranked,
            "recommendation": ranked[0] if ranked else None,
            "requires_human_review": True
        }

    async def _evaluate_single_quote(self, thread: EmailThread) -> Dict:
        prompt = f"""
Evaluate procurement negotiation:
{thread.get_full_thread()}

Analyze:
1. Final price competitiveness
2. Terms and conditions
3. Supplier responsiveness
4. Value for money
5. Risk factors

Provide structured evaluation.
"""
        evaluation = await self._call_ollama(prompt)

        return {
            "supplier_id": thread.supplier_id,
            "evaluation": evaluation,
            "total_rounds": thread.current_round
        }

    def _rank_suppliers(self, evaluations: Dict) -> List[str]:
        return list(evaluations.keys())

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate
        return "[Evaluation]"
