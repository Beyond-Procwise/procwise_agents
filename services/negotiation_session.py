from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

@dataclass
class SupplierNegotiation:
    supplier_id: str
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    last_response: Optional[Dict[str, Any]] = None

@dataclass
class NegotiationSession:
    session_id: str
    workflow_id: str
    supplier_negotiations: Dict[str, SupplierNegotiation] = field(default_factory=dict)
    current_round: int = 1
    max_rounds: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.redis_key = f"negotiation_session:{self.workflow_id}"

    def add_supplier(self, supplier_id: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        if supplier_id not in self.supplier_negotiations:
            self.supplier_negotiations[supplier_id] = SupplierNegotiation(
                supplier_id=supplier_id,
                parameters=parameters or {}
            )

    def record_response(self, supplier_id: str, response: Dict[str, Any]) -> None:
        if supplier_id not in self.supplier_negotiations:
            self.add_supplier(supplier_id)
        
        negotiation = self.supplier_negotiations[supplier_id]
        negotiation.round_history.append({
            "round": self.current_round,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        negotiation.last_response = response
        negotiation.status = "responded"

    def can_advance_round(self) -> bool:
        if self.current_round >= self.max_rounds:
            return False
        return all(
            neg.status == "responded" 
            for neg in self.supplier_negotiations.values()
        )

    def advance_round(self) -> bool:
        if not self.can_advance_round():
            return False
        self.current_round += 1
        for negotiation in self.supplier_negotiations.values():
            negotiation.status = "pending"
        return True

    def is_complete(self) -> bool:
        return (
            self.current_round >= self.max_rounds 
            or all(neg.status == "responded" for neg in self.supplier_negotiations.values())
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "parameters": self.parameters,
            "supplier_negotiations": {
                sid: {
                    "supplier_id": neg.supplier_id,
                    "round_history": neg.round_history,
                    "parameters": neg.parameters,
                    "status": neg.status,
                    "last_response": neg.last_response
                }
                for sid, neg in self.supplier_negotiations.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NegotiationSession:
        session = cls(
            session_id=data["session_id"],
            workflow_id=data["workflow_id"],
            current_round=data["current_round"],
            max_rounds=data["max_rounds"],
            parameters=data["parameters"]
        )
        
        for sid, neg_data in data["supplier_negotiations"].items():
            negotiation = SupplierNegotiation(
                supplier_id=neg_data["supplier_id"],
                round_history=neg_data["round_history"],
                parameters=neg_data["parameters"],
                status=neg_data["status"],
                last_response=neg_data["last_response"]
            )
            session.supplier_negotiations[sid] = negotiation
            
        return session

    def save(self, redis_client: redis.Redis) -> None:
        redis_client.set(
            self.redis_key,
            json.dumps(self.to_dict())
        )

    @classmethod
    def load(cls, workflow_id: str, redis_client: redis.Redis) -> Optional[NegotiationSession]:
        key = f"negotiation_session:{workflow_id}"
        data = redis_client.get(key)
        if not data:
            return None
        try:
            return cls.from_dict(json.loads(data))
        except Exception:
            logger.exception(f"Failed to load negotiation session for workflow {workflow_id}")
            return None
