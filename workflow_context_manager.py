import json
from datetime import datetime


class WorkflowContextManager:
    """CRITICAL: Initialize BEFORE any workflow execution"""

    def __init__(self, ollama_model: str = "qwen3:30b"):
        self.model = ollama_model
        self.workflow_context = {
            "phases": ["ranking", "initial_email", "negotiation_rounds", "evaluation"],
            "max_negotiation_rounds": 3,
            "human_checkpoints": ["email_review", "negotiation_review", "final_review"]
        }
        self.embeddings = {}

    async def initialize_workflow_understanding(self):
        """Call this FIRST before workflow execution"""
        workflow_prompt = f"""
You are managing a procurement workflow:
{json.dumps(self.workflow_context, indent=2)}

Key principles:
- Email threads maintain continuity via hidden identifiers
- Negotiation is 3 rounds maximum
- Human reviews all supplier communications
- Each supplier has unique thread tracked throughout
- All suppliers processed in parallel per round
- Context from previous rounds informs next rounds

Create mental model for optimal execution.
"""
        self.embeddings['workflow'] = await self._create_embedding(workflow_prompt)
        self.embeddings['negotiation_playbook'] = await self._load_negotiation_playbook()
        print("✅ Context initialized")
        return self.embeddings

    async def _create_embedding(self, text: str):
        return {"text": text, "timestamp": datetime.now().isoformat()}

    async def _load_negotiation_playbook(self):
        return {
            "round_1": {
                "strategy": "Express interest, ask clarifications, gather information",
                "tone": "Inquisitive, professional"
            },
            "round_2": {
                "strategy": "Present counter-offer, negotiate terms, provide justification",
                "tone": "Confident, data-driven"
            },
            "round_3": {
                "strategy": "Best and final offer, set deadline, firm stance",
                "tone": "Decisive, respectful but firm"
            }
        }
