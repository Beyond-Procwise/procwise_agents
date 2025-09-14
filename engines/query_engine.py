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
import pandas as pd

from .base_engine import BaseEngine
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

# Ensure GPU-related environment variables are set even for DB-heavy agents.
configure_gpu()


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        super().__init__()
        self.agent_nick = agent_nick

    def _price_expression(self, conn, schema: str, table: str) -> str:
        """Return SQL snippet for the unit price column in ``table``.

        Databases in different environments expose price information under
        various column names. This helper inspects ``information_schema`` and
        returns a suitable expression. If no price-related column is found a
        constant ``0.0`` is returned to avoid runtime SQL errors.
        """
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (schema, table),
                )
                cols = [r[0] for r in cur.fetchall()]

            price_cols = [c for c in cols if "price" in c]
            if "unit_price_gbp" in price_cols and "unit_price" in price_cols:
                return "COALESCE(unit_price_gbp, unit_price)"
            if "unit_price_gbp" in price_cols:
                return "unit_price_gbp"
            if "unit_price" in price_cols:
                return "unit_price"
            if price_cols:
                # Use the first price-like column as a last resort
                return price_cols[0]

            logger.warning("no price column found on %s.%s; defaulting to zero", schema, table)
        except Exception:
            logger.exception("price column detection failed")
        return "0.0"

    def _quantity_expression(self, conn, schema: str, table: str) -> str:
        """Return SQL snippet for the quantity column in ``table``.

        Similar to :meth:`_price_expression`, this helper inspects the
        ``information_schema`` to locate a column representing quantity or
        count of items.  If no such column exists a constant ``1`` is
        returned so that spend calculations still succeed without raising
        a ``DatabaseError``.
        """
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (schema, table),
                )
                cols = [r[0] for r in cur.fetchall()]

            qty_cols = [c for c in cols if "qty" in c or "quantity" in c]
            if qty_cols:
                # Use the first match and coalesce to 1 in case of NULLs
                return f"COALESCE({qty_cols[0]}, 1)"

            logger.warning("no quantity column found on %s.%s; defaulting to 1", schema, table)
        except Exception:
            logger.exception("quantity column detection failed")
        return "1"

    def fetch_supplier_data(self, input_data: dict = None) -> pd.DataFrame:
        """Return up-to-date supplier metrics."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                # Line item tables carry the price/quantity information we need
                po_price = self._price_expression(conn, "proc", "po_line_items_agent")
                inv_price = self._price_expression(conn, "proc", "invoice_line_items_agent")
                po_qty = self._quantity_expression(conn, "proc", "po_line_items_agent")
                inv_qty = self._quantity_expression(conn, "proc", "invoice_line_items_agent")

                sql = f"""
                WITH po AS (
                    SELECT p.supplier_id,
                           SUM({po_price} * {po_qty}) AS po_spend
                    FROM proc.po_line_items_agent li
                    JOIN proc.purchase_order_agent p ON p.po_id = li.po_id
                    WHERE p.supplier_id IS NOT NULL
                    GROUP BY p.supplier_id
                ), inv AS (
                    SELECT i.supplier_id,
                           SUM({inv_price} * {inv_qty}) AS invoice_spend,
                           COUNT(DISTINCT i.invoice_id) AS invoice_count,
                           AVG(CASE WHEN i.on_time = TRUE THEN 1.0 ELSE 0.0 END) AS on_time_pct
                    FROM proc.invoice_agent i
                    LEFT JOIN proc.invoice_line_items_agent li ON i.invoice_id = li.invoice_id
                    WHERE i.supplier_id IS NOT NULL
                    GROUP BY i.supplier_id
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
        except Exception as exc:
            # Surface the original exception so callers can handle it explicitly
            logger.exception("fetch_supplier_data failed")
            raise RuntimeError("fetch_supplier_data failed") from exc

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
