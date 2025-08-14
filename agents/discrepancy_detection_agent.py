import logging
import os
from typing import Dict, Iterable, List, Optional

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

        def _detect_column(cur, connection, candidates: Iterable[str]) -> Optional[str]:
            """Return the first candidate column present in the invoice table."""

            for col in candidates:
                try:
                    cur.execute(f"SELECT {col} FROM proc.invoice_agent LIMIT 0")
                    return col
                except errors.UndefinedColumn:
                    connection.rollback()
            return None

        def _detect_vendor_column(cur, connection) -> Optional[str]:
            return _detect_column(cur, connection, ("vendor_name", "vendor"))

        def _detect_date_column(cur, connection) -> Optional[str]:
            return _detect_column(cur, connection, ("invoice_date", "invoice_dt", "date"))

        def _detect_amount_column(cur, connection) -> Optional[str]:
            return _detect_column(
                cur,
                connection,
                (
                    "total_amount",
                    "total_with_tax",
                    "invoice_total_incl_tax",
                    "invoice_total",
                    "total",
                    "invoice_amount",
                    "tax_amount",
                ),
            )

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    vendor_col = _detect_vendor_column(cursor, conn)
                    date_col = _detect_date_column(cursor, conn)
                    amount_col = _detect_amount_column(cursor, conn)
                    for doc in docs:
                        doc_type = doc.get("doc_type")
                        pk = doc.get("id")
                        if doc_type != "Invoice":
                            continue  # Only invoices supported for now
                        try:
                            columns = []
                            if vendor_col:
                                columns.append(vendor_col)
                            if date_col:
                                columns.append(date_col)
                            if amount_col:
                                columns.append(amount_col)

                            if columns:
                                cursor.execute(
                                    f"SELECT {', '.join(columns)} FROM proc.invoice_agent WHERE invoice_id = %s",
                                    (pk,),
                                )
                                row = cursor.fetchone()

                                idx = 0
                                if vendor_col:
                                    vendor_name = row[idx]
                                    idx += 1
                                else:
                                    vendor_name = None
                                if date_col:
                                    invoice_date = row[idx]
                                    idx += 1
                                else:
                                    invoice_date = None
                                if amount_col:
                                    total_amount = row[idx]
                                else:
                                    total_amount = None
                            else:
                                cursor.execute(
                                    "SELECT 1 FROM proc.invoice_agent WHERE invoice_id = %s",
                                    (pk,),
                                )
                                row = cursor.fetchone()
                                vendor_name = None
                                invoice_date = None
                                total_amount = None
                        except Exception as db_exc:
                            conn.rollback()
                            logger.error(
                                "DB error during discrepancy check: %s", db_exc
                            )
                            mismatches.append(
                                {
                                    "doc_type": doc_type,
                                    "id": pk,
                                    "checks": {"db_error": str(db_exc)},
                                }
                            )
                            continue
                        if not row:
                            mismatches.append(
                                {
                                    "doc_type": doc_type,
                                    "id": pk,
                                    "checks": {"record": "missing"},
                                }
                            )
                            continue
                        checks = {}
                        if not vendor_name:
                            checks["vendor_name"] = "missing"
                        if invoice_date is None:
                            checks["invoice_date"] = "missing"
                        if total_amount is None or total_amount <= 0:
                            checks["total_amount"] = "invalid"
                        if checks:
                            mismatches.append(
                                {"doc_type": doc_type, "id": pk, "checks": checks}
                            )
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("DiscrepancyDetectionAgent failed: %s", exc)
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(exc),
            )

        return AgentOutput(status=AgentStatus.SUCCESS, data={"mismatches": mismatches})
