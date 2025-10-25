# ProcWise/orchestration/langgraph_state.py

"""LangGraph state helpers for bridging existing agent context data.

The Beyond ProcWise orchestration layer is gradually migrating from a purely
imperative workflow controller to LangGraph-based state machines.  This module
provides a lightweight state container that mirrors the information carried in
``AgentContext`` so it can be threaded through LangGraph nodes without losing
critical procurement metadata such as workflow identifiers, supplier details,
or policy manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import guard for static analysis only
    from agents.base_agent import AgentContext


@dataclass
class LangGraphAgentState:
    """Structured state shared across LangGraph nodes.

    Attributes capture the procurement workflow context (``workflow_id``,
    ``supplier_id``, negotiation ``round``), conversational history, and any
    intermediate artefacts produced by agents.  The ``manifest`` field embeds
    the policy/task context emitted by :class:`AgentContext.manifest` so that
    downstream nodes can respect sourcing guardrails without additional lookups.
    """

    workflow_id: str
    agent_id: str
    supplier_id: Optional[str] = None
    round: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent_context(
        cls,
        agent_context: "AgentContext",
        **overrides: Any,
    ) -> "LangGraphAgentState":
        """Create state seeded from an :class:`AgentContext` instance."""

        base_context = dict(getattr(agent_context, "input_data", {}) or {})
        manifest = agent_context.manifest() if hasattr(agent_context, "manifest") else {}
        supplier_id = base_context.get("supplier_id") or base_context.get("supplier")
        negotiation_round = (
            base_context.get("round")
            or base_context.get("negotiation_round")
            or overrides.get("round")
        )

        state = cls(
            workflow_id=agent_context.workflow_id,
            agent_id=agent_context.agent_id,
            supplier_id=supplier_id,
            round=negotiation_round,
            context=base_context,
            manifest=manifest,
        )
        state.metadata.update(
            {
                "parent_agent": getattr(agent_context, "parent_agent", None),
                "routing_history": list(getattr(agent_context, "routing_history", [])),
                "process_id": getattr(agent_context, "process_id", None),
            }
        )
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def push_message(self, role: str, content: str, **extra: Any) -> Dict[str, Any]:
        """Append a chat-style message to the state history."""

        message = {"role": role, "content": content}
        message.update({k: v for k, v in extra.items() if v is not None})
        self.messages.append(message)
        return message

    def update_context(self, **fields: Any) -> None:
        """Merge new context fields into the state payload."""

        self.context.update({k: v for k, v in fields.items() if v is not None})

    def record_result(self, key: str, value: Any) -> None:
        """Store intermediate artefacts produced by LangGraph nodes."""

        self.results[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the state."""

        return {
            "workflow_id": self.workflow_id,
            "agent_id": self.agent_id,
            "supplier_id": self.supplier_id,
            "round": self.round,
            "context": dict(self.context),
            "manifest": dict(self.manifest),
            "messages": [dict(message) for message in self.messages],
            "results": dict(self.results),
            "metadata": dict(self.metadata),
        }
