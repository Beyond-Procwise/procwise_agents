# ProcWise/engines/query_engine.py
"""Light‑weight database access layer used by the orchestrator.

Only a single method is required for the exercises in this repository –
``fetch_supplier_data`` which collects information about suppliers by
combining the supplier master data with aggregated information from invoice
and purchase order tables.  The returned ``pandas.DataFrame`` is consumed by
:class:`SupplierRankingAgent`.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .base_engine import BaseEngine

logger = logging.getLogger(__name__)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        super().__init__()
        self.agent_nick = agent_nick

    def _has_column(self, conn, schema: str, table: str, column: str) -> bool:
        """Return ``True`` if the given column exists on the table."""
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = %s
                    """,
                    (schema, table, column),
                )
                return cur.fetchone() is not None
        except Exception:
            logger.exception("column existence check failed")
            return False

    def fetch_supplier_data(self, input_data: dict = None) -> pd.DataFrame:
        """Return up-to-date supplier metrics."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                po_price = (
                    "COALESCE(unit_price_gbp, unit_price)"
                    if self._has_column(conn, "proc", "purchase_order_agent", "unit_price_gbp")
                    else "unit_price"
                )
                inv_price = (
                    "COALESCE(unit_price_gbp, unit_price)"
                    if self._has_column(conn, "proc", "invoice_agent", "unit_price_gbp")
                    else "unit_price"
                )

                sql = f"""
                WITH po AS (
                    SELECT supplier_id,
                           SUM({po_price} * COALESCE(quantity, 1)) AS po_spend
                    FROM proc.purchase_order_agent
                    WHERE supplier_id IS NOT NULL
                    GROUP BY supplier_id
                ), inv AS (
                    SELECT supplier_id,
                           SUM({inv_price} * COALESCE(quantity, 1)) AS invoice_spend,
                           COUNT(DISTINCT invoice_id) AS invoice_count,
                           AVG(CASE WHEN on_time = TRUE THEN 1.0 ELSE 0.0 END) AS on_time_pct
                    FROM proc.invoice_agent
                    WHERE supplier_id IS NOT NULL
                    GROUP BY supplier_id
                )
                SELECT
                    s.supplier_id,
                    s.supplier_name,
                    COALESCE(po.po_spend, 0.0) AS po_spend,
                    COALESCE(inv.invoice_spend, 0.0) AS invoice_spend,
                    COALESCE(po.po_spend, 0.0) + COALESCE(inv.invoice_spend, 0.0) AS total_spend,
                    COALESCE(inv.invoice_count, 0) AS invoice_count,
                    COALESCE(inv.on_time_pct, 0.0) AS on_time_pct,
                    -- include other supplier fields if present
                    s.*
                FROM proc.supplier s
                LEFT JOIN po ON s.supplier_id = po.supplier_id
                LEFT JOIN inv ON s.supplier_id = inv.supplier_id
                """
                df = pd.read_sql(sql, conn)

            if "supplier_id" in df.columns:
                df["supplier_id"] = df["supplier_id"].astype(str)
            return df
        except Exception:
            logger.exception("fetch_supplier_data failed")
            return None

    def fetch_invoice_data(self, intent: dict | None = None) -> pd.DataFrame:
        """Return invoice headers from ``proc.invoice_agent``."""
        sql = "SELECT * FROM proc.invoice_agent;"
        with self.agent_nick.get_db_connection() as conn:
            return pd.read_sql(sql, conn)

    def fetch_purchase_order_data(self, intent: dict | None = None) -> pd.DataFrame:
        """Return purchase order headers from ``proc.purchase_order_agent``."""
        sql = "SELECT * FROM proc.purchase_order_agent;"
        with self.agent_nick.get_db_connection() as conn:
            return pd.read_sql(sql, conn)
