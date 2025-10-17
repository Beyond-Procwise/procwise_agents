from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
import uuid


@dataclass
class EmailThread:
    """ONE thread per supplier, maintained throughout entire workflow"""

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    supplier_id: str = ""
    supplier_unique_id: str = field(default_factory=lambda: f"SUP-{uuid.uuid4().hex[:10].upper()}")
    hidden_identifier: str = field(default_factory=lambda: f"PROC-WF-{uuid.uuid4().hex[:12].upper()}")
    messages: List[Dict] = field(default_factory=list)
    current_round: int = 0

    def add_message(self, message_type: str, content: str, action_id: str):
        self.messages.append({
            "message_type": message_type,
            "content": content,
            "action_id": action_id,
            "timestamp": datetime.now().isoformat(),
            "round": self.current_round
        })

    def get_full_thread(self) -> str:
        thread_content = f"Thread: {self.hidden_identifier}\nSupplier: {self.supplier_id}\n\n"
        for msg in self.messages:
            thread_content += f"[Round {msg['round']}] {msg['message_type']}:\n{msg['content']}\n\n"
        return thread_content


@dataclass
class NegotiationSession:
    session_id: str
    start_time: str
    current_round: int = 0
    supplier_states: Dict[str, Dict] = field(default_factory=dict)
