"""Simple context holder used by the illustrative workflow orchestrator."""

from __future__ import annotations

import json  # Required for serialising the workflow context via json.dumps
from datetime import datetime, timezone
from typing import Any, Dict


class WorkflowContextManager:
    """Maintain high level procurement workflow context for prompt construction."""

    def __init__(self, ollama_model: str = "qwen3:30b") -> None:
        self.model = ollama_model
        self.workflow_context = {
            "phases": [
                "ranking",
                "initial_email",
                "negotiation_rounds",
                "evaluation",
            ],
            "max_negotiation_rounds": 3,
            "human_checkpoints": [
                "email_review",
                "negotiation_review",
                "final_review",
            ],
        }
        self.embeddings: Dict[str, Any] = {}

    async def initialize_workflow_understanding(self) -> Dict[str, Any]:
        """Seed the minimal context artefacts used by the lightweight agents."""

        workflow_prompt = json.dumps(self.workflow_context, indent=2)
        self.embeddings["workflow"] = await self._create_embedding(workflow_prompt)
        self.embeddings["negotiation_playbook"] = await self._load_negotiation_playbook()
        return self.embeddings

    async def _create_embedding(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _load_negotiation_playbook(self) -> Dict[str, Dict[str, str]]:
        return {
            "round_1": {
                "strategy": "Express interest, ask clarifications, gather information",
                "tone": "Inquisitive, professional",
            },
            "round_2": {
                "strategy": "Present counter-offer, negotiate terms, provide justification",
                "tone": "Confident, data-driven",
            },
            "round_3": {
                "strategy": "Best and final offer, set deadline, firm stance",
                "tone": "Decisive, respectful but firm",
            },
        }
