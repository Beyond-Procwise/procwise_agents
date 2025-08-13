import logging
import os
from typing import Dict, List

import torch
from psycopg2 import errors

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)

# Ensure GPU is accessible when available and downstream libs use it
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_DEFAULT_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
if torch.cuda.is_available():  # pragma: no cover - hardware dependent
    torch.set_default_device("cuda")
else:  # pragma: no cover - hardware dependent
    logger.warning("CUDA not available; defaulting to CPU.")


class DiscrepancyDetectionAgent(BaseAgent):
    """Agent that validates extracted documents and logs mismatches."""

    REQUIRED_INVOICE_FIELDS = ["vendor_name", "invoice_date", "total_amount"]

    def run(self, context: AgentContext) -> AgentOutput:
        mismatches: List[Dict[str, Dict[str, str]]] = []
        docs: List[Dict] = context.input_data.get("extracted_docs", [])

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for doc in docs:
                        doc_type = doc.get("doc_type")
                        pk = doc.get("id")
                        if doc_type != "Invoice":
                            continue  # Only invoices supported for now
                        try:
                            cursor.execute(
                                """
                                SELECT vendor_name, invoice_date, total_amount
                                FROM proc.invoice_agent
                                WHERE invoice_id = %s
                                """,
                                (pk,),
                            )
                        except errors.UndefinedColumn:
                            cursor.execute(
                                """
                                SELECT vendor, invoice_date, total_amount
                                FROM proc.invoice_agent
                                WHERE invoice_id = %s
                                """,
                                (pk,),
                            )
                        row = cursor.fetchone()
                        if not row:
                            mismatches.append(
                                {
                                    "doc_type": doc_type,
                                    "id": pk,
                                    "checks": {"record": "missing"},
                                }
                            )
                            continue
                        vendor_name, invoice_date, total_amount = row
                        checks = {}
                        if not vendor_name:
                            checks["vendor_name"] = "missing"
                        if invoice_date is None:
                            checks["invoice_date"] = "missing"
                        if total_amount is None or total_amount <= 0:
                            checks["total_amount"] = "invalid"
                        if checks:
                            mismatches.append({"doc_type": doc_type, "id": pk, "checks": checks})
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("DiscrepancyDetectionAgent failed: %s", exc)
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(exc),
            )

        return AgentOutput(status=AgentStatus.SUCCESS, data={"mismatches": mismatches})

