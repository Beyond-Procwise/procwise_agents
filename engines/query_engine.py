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

    def fetch_supplier_data(self, intent: dict | None = None) -> pd.DataFrame:
        """Return supplier information required for ranking.

        The query joins ``proc.supplier`` with invoice and purchase order
        tables.  Only suppliers that appear in either table are returned.  In
        addition to scoring fields, key attributes such as ``trading_name``,
        ``legal_structure`` and ``is_preferred_supplier`` are returned so that
        prompts can reference these values.  Raw numeric metrics are suffixed
        with ``_score_raw`` so that the ranking agent can normalise them
        according to the defined policies.
        """
        sql = """SELECT supplier_id, supplier_name, trading_name, supplier_type, 
                    registered_country, is_preferred_supplier, risk_score 
                    FROM proc.supplier;""",
        with self.agent_nick.get_db_connection() as conn:
            return pd.read_sql(sql, conn)
