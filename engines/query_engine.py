# ProcWise/engines/query_engine.py

import pandas as pd
from .base_engine import BaseEngine  # <-- CORRECTED IMPORT

class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

    def fetch_supplier_data(self, intent: dict) -> pd.DataFrame:
        print(f"QueryEngine: Fetching data for intent -> {intent} (from mock source)")
        # In a real system, this would build a dynamic SQL query
        mock_data = {
            'supplier_name': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D'],
            'price_score_raw': [100, 120, 110, 95],
            'delivery_score_raw': [98, 95, 99, 92],
            'risk_score_raw': [3, 5, 2, 4],
            'payment_terms_raw': [30, 60, 30, 90]
        }
        return pd.DataFrame(mock_data)
