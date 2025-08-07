# ProcWise/engines/query_engine.py

import pandas as pd
from typing import Dict, Any, List
from .base_engine import BaseEngine  # <-- CORRECTED IMPORT


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

    def _build_supplier_query(self, parameters: Dict[str, Any]) -> tuple[str, List[Any]]:
        """Builds a SQL query for fetching supplier metrics based on intent parameters."""

        # Map criteria to database columns (more can be added as needed)
        criteria_column_map = {
            "price": "m.avg_price",
            "delivery": "m.delivery_performance",
            "delivery performance": "m.delivery_performance",
            "risk": "m.risk_rating",
            "payment_terms": "s.payment_terms",
        }

        # Build SELECT clause mapping DB columns to *_score_raw fields
        select_clause = ["s.supplier_name"]
        for criterion, db_col in criteria_column_map.items():
            alias = f"{criterion}_score_raw"
            if " " in alias:
                select_clause.append(f"{db_col} AS \"{alias}\"")
            else:
                select_clause.append(f"{db_col} AS {alias}")

        query = f"""
            SELECT {', '.join(select_clause)}
            FROM proc.supplier s
            LEFT JOIN proc.supplier_metrics m ON m.supplier_id = s.supplier_id
            WHERE 1=1
        """

        query_params: List[Any] = []

        # Filtering by supplier list
        suppliers = parameters.get("suppliers_list")
        if suppliers:
            query += " AND s.supplier_name = ANY(%s)"
            query_params.append(suppliers)

        # Filtering by category
        category = parameters.get("category")
        if category:
            query += " AND s.category = %s"
            query_params.append(category)

        # Filtering by arbitrary additional filters (column=value)
        filters = parameters.get("filters")
        if isinstance(filters, dict):
            for col, val in filters.items():
                query += f" AND s.{col} = %s"
                query_params.append(val)

        # Filtering by time period
        time_period = parameters.get("time_period")
        if isinstance(time_period, dict):
            start = time_period.get("start_date") or time_period.get("start")
            end = time_period.get("end_date") or time_period.get("end")
            if start:
                query += " AND m.metric_date >= %s"
                query_params.append(start)
            if end:
                query += " AND m.metric_date <= %s"
                query_params.append(end)

        return query, query_params

    def fetch_supplier_data(self, intent: dict) -> pd.DataFrame:
        """Fetches supplier metrics from the database based on the intent."""

        parameters = intent.get("parameters", {})
        query, params = self._build_supplier_query(parameters)

        print(f"QueryEngine: Executing supplier data query with parameters -> {parameters}")

        with self.agent_nick.get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df
