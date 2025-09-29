import json
import logging
from typing import Dict, Iterable, List, Optional

from psycopg2 import errors

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

configure_gpu()


class DiscrepancyDetectionAgent(BaseAgent):
    """Agent that validates extracted documents and logs mismatches."""

    REQUIRED_INVOICE_FIELDS = ["vendor_name", "invoice_date", "total_amount"]

    def run(self, context: AgentContext) -> AgentOutput:
        mismatches: List[Dict[str, Dict[str, str]]] = []
        docs: List[Dict] = context.input_data.get("extracted_docs", [])
        processing_issues: List[Dict[str, str]] = context.input_data.get(
            "processing_issues", []
        )

        if processing_issues:
            for issue in processing_issues:
                mismatches.append(
                    {
                        "doc_type": issue.get("doc_type", "Unknown"),
                        "id": issue.get("record_id", "unknown"),
                        "checks": {"processing_issue": issue.get("reason", "unspecified")},
                    }
                )

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
                                if row is None:
                                    mismatches.append(
                                        {
                                            "doc_type": doc_type,
                                            "id": pk,
                                            "checks": {"record": "missing"},
                                        }
                                    )
                                    continue

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
                                if not row:
                                    mismatches.append(
                                        {
                                            "doc_type": doc_type,
                                            "id": pk,
                                            "checks": {"record": "missing"},
                                        }
                                    )
                                    continue
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

        self._persist_mismatches(mismatches)

        processed_ids: set[str] = set()
        for doc in docs:
            identifier = doc.get("id") or doc.get("record_id") or doc.get("invoice_id") or doc.get("po_id") or doc.get("contract_id") or doc.get("quote_id")
            if identifier is None:
                continue
            processed_ids.add(str(identifier))

        failed_ids: set[str] = set()
        for mis in mismatches:
            mid = mis.get("id")
            if mid is None:
                continue
            sid = str(mid)
            if sid in processed_ids:
                failed_ids.add(sid)

        summary = {
            "documents_processed": len(processed_ids) if processed_ids else len(docs),
            "documents_successful": max(0, (len(processed_ids) if processed_ids else len(docs)) - len(failed_ids)),
            "documents_with_issues": len(failed_ids),
        }
        if processing_issues:
            summary["processing_issues"] = len(processing_issues)

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={"mismatches": mismatches, "summary": summary},
        )

    def _persist_mismatches(self, mismatches: List[Dict]) -> None:
        """Store discrepancies in proc.data_discrepancy for auditing."""
        if not mismatches:
            return
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.data_discrepancy (
                            id SERIAL PRIMARY KEY,
                            doc_type TEXT NOT NULL,
                            record_id TEXT NOT NULL,
                            details JSONB NOT NULL,
                            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                        )
                        """
                    )
                    for mis in mismatches:
                        cur.execute(
                            "INSERT INTO proc.data_discrepancy (doc_type, record_id, details) VALUES (%s, %s, %s)",
                            (
                                mis.get("doc_type"),
                                mis.get("id"),
                                json.dumps(mis.get("checks", {})),
                            ),
                        )
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.error("Failed to persist discrepancies: %s", exc)
