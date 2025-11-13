"""Email thread utilities ensuring identifier continuity across negotiation rounds."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _generate_thread_id() -> str:
    return str(uuid.uuid4())


def _generate_supplier_unique_id() -> str:
    return f"SUP-{uuid.uuid4().hex[:10].upper()}"


def _generate_unique_identifier(workflow_id: str, supplier_id: str) -> str:
    seed = f"{workflow_id}-{supplier_id}-{uuid.uuid4().hex[:8]}"
    token = seed.replace("_", "-").upper()
    return f"PROC-WF-{token[:16]}"


def make_action_id(round_num: int, supplier_unique_id: str) -> str:
    """Create deterministic action identifiers for outbound negotiation drafts."""

    prefix = supplier_unique_id.upper()
    if not prefix.startswith("SUP-"):
        prefix = f"SUP-{prefix}"
    suffix = uuid.uuid4().hex[:6].upper()
    return f"NEG-R{round_num}-{prefix}-{suffix}"


@dataclass
class EmailThread:
    """Represents a single supplier communication thread across the workflow."""

    workflow_id: str
    supplier_id: str
    supplier_unique_id: str = field(default_factory=_generate_supplier_unique_id)
    unique_id: str = ""
    thread_id: str = field(default_factory=_generate_thread_id)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0

    def __post_init__(self) -> None:
        self.supplier_unique_id = self._normalise_supplier_unique_id(self.supplier_unique_id)
        if not self.unique_id:
            self.unique_id = _generate_unique_identifier(self.workflow_id, self.supplier_id)

    @staticmethod
    def _normalise_supplier_unique_id(value: str) -> str:
        token = (value or "").upper()
        if not token.startswith("SUP-"):
            token = f"SUP-{token}" if token else _generate_supplier_unique_id()
        if len(token.split("-")) == 1:
            token = f"SUP-{token}"  # ensure prefix
        return token

    def add_message(
        self,
        message_type: str,
        content: str,
        action_id: str,
        *,
        round_num: Optional[int] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach a message to the thread without mutating identifiers."""

        effective_round = self.current_round if round_num is None else max(round_num, 0)
        self.current_round = max(self.current_round, effective_round)

        entry: Dict[str, Any] = {
            "thread_id": self.thread_id,
            "workflow_id": self.workflow_id,
            "supplier_id": self.supplier_id,
            "supplier_unique_id": self.supplier_unique_id,
            "unique_id": self.unique_id,
            "message_type": message_type,
            "content": content,
            "action_id": action_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": effective_round,
        }
        if headers:
            entry["headers"] = dict(headers)

        self.messages.append(entry)
        return entry

    def get_full_thread(self) -> str:
        """Format the entire thread for downstream LLM context windows."""

        header = (
            f"Thread ID: {self.thread_id}\n"
            f"Workflow ID: {self.workflow_id}\n"
            f"Supplier ID: {self.supplier_id}\n"
            f"Supplier Unique ID: {self.supplier_unique_id}\n"
            f"Unique Identifier: {self.unique_id}\n"
        )
        body_segments = []
        for message in self.messages:
            body_segments.append(
                f"[Round {message['round']}] {message['message_type']}\n{message['content']}"
            )
        return header + "\n".join(body_segments)


@dataclass
class NegotiationSession:
    """Metadata for a workflow execution session."""

    session_id: str
    workflow_id: str
    start_time: str
    current_round: int = 0
    supplier_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
