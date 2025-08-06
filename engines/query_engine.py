# ProcWise/engines/query_engine.py

import pandas as pd
import numpy as np
import logging
from .base_engine import BaseEngine

logger = logging.getLogger(__name__)

class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

    def fetch_supplier_data(self, intent: dict) -> pd.DataFrame:
        """
        --- UPGRADE: Dynamic Data Querying ---
        Fetches and aggregates supplier data directly from the PostgreSQL database,
        replacing the previous mock data implementation.
        """
        logger.info(f"QueryEngine: Fetching dynamic data for intent -> {intent}")

        # In a real system, you would build a more complex query based on the intent.
        # For this example, we'll aggregate core metrics from the invoice table.
        # This query calculates average invoice amount (as a proxy for price) and
        # gets the most recent payment terms for each vendor.
        sql_query = """
        SELECT
            vendor_name,
            AVG(total_amount) as price_score_raw,
            (array_agg(payment_terms ORDER BY invoice_date DESC))[1] as payment_terms_raw
        FROM
            proc.invoice_agent
        WHERE
            vendor_name IS NOT NULL
        GROUP BY
            vendor_name;
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                df = pd.read_sql_query(sql_query, conn)

            if df.empty:
                logger.warning("QueryEngine: No supplier data found in the database.")
                return pd.DataFrame()

            # --- Placeholder Data for Metrics Not in Invoices ---
            # In a production system, this data would come from other tables or APIs
            # (e.g., a supplier risk management system or delivery tracking system).
            np.random.seed(42) # for reproducible results
            df['delivery_score_raw'] = np.random.uniform(90, 99.5, df.shape[0]).round(2)
            df['risk_score_raw'] = np.random.randint(1, 6, df.shape[0])

            logger.info(f"Successfully fetched and prepared data for {len(df)} suppliers.")
            return df

        except Exception as e:
            logger.error(f"QueryEngine: Failed to execute query against database: {e}")
            return pd.DataFrame()
