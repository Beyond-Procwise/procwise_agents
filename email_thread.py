"""Email thread utilities ensuring identifier continuity across negotiation rounds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def _generate_thread_id() -> str:
    return str(uuid.uuid4())


def _generate_supplier_unique_id() -> str:
    return f"SUP-{uuid.uuid4().hex[:10].upper()}"


def _generate_hidden_identifier() -> str:
    return f"PROC-WF-{uuid.uuid4().hex[:12].upper()}"


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

    thread_id: str = field(default_factory=_generate_thread_id)
    supplier_id: str = ""
    supplier_unique_id: str = field(default_factory=_generate_supplier_unique_id)
    hidden_identifier: str = field(default_factory=_generate_hidden_identifier)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0

    def __post_init__(self) -> None:
        self.supplier_unique_id = self._normalise_supplier_unique_id(self.supplier_unique_id)
        self.hidden_identifier = self._normalise_hidden_identifier(self.hidden_identifier)

    @staticmethod
    def _normalise_supplier_unique_id(value: str) -> str:
        token = (value or "").upper()
        if not token.startswith("SUP-"):
            token = f"SUP-{token}" if token else _generate_supplier_unique_id()
        if len(token.split("-")) == 1:
            token = f"SUP-{token}"  # ensure prefix
        return token

    @staticmethod
    def _normalise_hidden_identifier(value: str) -> str:
        token = (value or "").upper()
        if not token.startswith("PROC-WF-"):
            token = f"PROC-WF-{token}" if token else _generate_hidden_identifier()
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
            "hidden_identifier": self.hidden_identifier,
            "supplier_id": self.supplier_id,
            "supplier_unique_id": self.supplier_unique_id,
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
            f"Supplier ID: {self.supplier_id}\n"
            f"Supplier Unique ID: {self.supplier_unique_id}\n"
            f"Hidden Identifier: {self.hidden_identifier}\n"
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
    start_time: str
    current_round: int = 0
    supplier_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
