# ProcWise/engines/query_engine.py
"""Light‑weight database access layer used by the orchestrator.

Only a single method is required for the exercises in this repository –
``fetch_supplier_data`` which collects information about suppliers by
combining the supplier master data with aggregated information from invoice
and purchase order tables.  The returned ``pandas.DataFrame`` is consumed by
:class:`SupplierRankingAgent`.
"""
from __future__ import annotations

import pandas as pd
from .base_engine import BaseEngine


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        super().__init__()
        self.agent_nick = agent_nick

    def fetch_supplier_data(self, intent: dict) -> pd.DataFrame:
        """Return supplier information required for ranking.

        The query joins ``proc.supplier`` with invoice and purchase order
        tables.  Only suppliers that appear in either table are returned.  Raw
        numeric metrics are suffixed with ``_score_raw`` so that the ranking
        agent can normalise them according to the defined policies.
        """
        sql = """
            SELECT s.supplier_id,
                   s.supplier_name,
                   s.payment_terms     AS payment_terms_raw,
                   s.price_score       AS price_score_raw,
                   s.delivery_score    AS delivery_score_raw,
                   s.risk_score        AS risk_score_raw,
                   COALESCE(inv.total_invoiced, 0) AS total_invoiced_score_raw,
                   COALESCE(po.total_ordered, 0)  AS total_ordered_score_raw
            FROM proc.supplier s
            LEFT JOIN (
                SELECT supplier_id, SUM(total_amount) AS total_invoiced
                FROM proc.invoice
                GROUP BY supplier_id
            ) inv ON inv.supplier_id = s.supplier_id
            LEFT JOIN (
                SELECT supplier_id, SUM(total_amount) AS total_ordered
                FROM proc.purchase_order
                GROUP BY supplier_id
            ) po ON po.supplier_id = s.supplier_id
            WHERE inv.supplier_id IS NOT NULL OR po.supplier_id IS NOT NULL;
        """
        with self.agent_nick.get_db_connection() as conn:
            return pd.read_sql(sql, conn)
