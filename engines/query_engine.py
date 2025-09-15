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


# Columns expected on ``proc.supplier``. The list mirrors the schema so that the
# query can explicitly project each field rather than relying on ``s.*`` which
# tends to be brittle across database revisions.
SUPPLIER_FIELDS = [
    "supplier_id",
    "supplier_name",
    "trading_name",
    "supplier_type",
    "legal_structure",
    "tax_id",
    "vat_number",
    "duns_number",
    "parent_company_id",
    "registered_country",
    "registration_number",
    "is_preferred_supplier",
    "risk_score",
    "credit_limit_amount",
    "esg_cert_iso14001",
    "esg_cert_sa8000",
    "esg_cert_ecovadis",
    "diversity_women_owned",
    "diversity_minority_owned",
    "diversity_veteran_owned",
    "insurance_coverage_type",
    "insurance_coverage_amount",
    "insurance_expiry_date",
    "bank_name",
    "bank_account_number",
    "bank_swift",
    "bank_iban",
    "default_currency",
    "incoterms",
    "delivery_lead_time_days",
    "address_line1",
    "address_line2",
    "city",
    "postal_code",
    "country",
    "website_url",
    "edi_enabled",
    "api_enabled",
    "ariba_integrated",
    "contact_name_1",
    "contact_role_1",
    "contact_email_1",
    "contact_phone_1",
    "contact_name_2",
    "contact_role_2",
    "contact_email_2",
    "contact_phone_2",
    "created_date",
    "created_by",
    "last_modified_by",
    "last_modified_date",
]


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        super().__init__()
        self.agent_nick = agent_nick

    def _get_columns(self, conn, schema: str, table: str) -> list[str]:
        """Return list of column names for ``schema.table``.

        Accessing ``information_schema`` can raise errors when databases are
        unavailable or the caller lacks permissions.  In those situations an
        empty list is returned so that callers can gracefully fall back to
        safe defaults instead of propagating the failure to SQL execution.
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
                return [r[0] for r in cur.fetchall()]
        except Exception:
            logger.exception("column introspection failed for %s.%s", schema, table)
            return []

    def _price_expression(self, conn, schema: str, table: str, alias: str) -> str:
        """Return SQL snippet for the unit price column in ``table``.

        ``alias`` is applied to any discovered column names so that the returned
        expression can safely be embedded into queries that join multiple tables
        which might expose similarly named fields, avoiding "ambiguous column"
        errors in PostgreSQL.

        Databases in different environments expose price information under
        various column names. This helper inspects ``information_schema`` and
        returns a suitable expression. If no price-related column is found a
        constant ``0.0`` is returned to avoid runtime SQL errors.
        """
        cols = self._get_columns(conn, schema, table)
        price_cols = [c for c in cols if "price" in c]
        try:
            if "unit_price_gbp" in price_cols and "unit_price" in price_cols:
                return f"COALESCE({alias}.unit_price_gbp, {alias}.unit_price)"
            if "unit_price_gbp" in price_cols:
                return f"{alias}.unit_price_gbp"
            if "unit_price" in price_cols:
                return f"{alias}.unit_price"
            if price_cols:
                # Use the first price-like column as a last resort
                return f"{alias}.{price_cols[0]}"

            logger.warning(
                "no price column found on %s.%s; defaulting to zero", schema, table
            )
        except Exception:
            logger.exception("price column detection failed")
        return "0.0"

    def _quantity_expression(self, conn, schema: str, table: str, alias: str) -> str:
        """Return SQL snippet for the quantity column in ``table``.

        ``alias`` is applied to discovered columns to avoid ambiguity when the
        target table is joined with other tables exposing a column of the same
        name. If no quantity column exists a constant ``1`` is returned so that
        spend calculations still succeed without raising a ``DatabaseError``.
        """
        cols = self._get_columns(conn, schema, table)
        try:
            qty_cols = [c for c in cols if "qty" in c or "quantity" in c]
            if qty_cols:
                # Use the first match and coalesce to 1 in case of NULLs
                return f"COALESCE({alias}.{qty_cols[0]}, 1)"

            logger.warning(
                "no quantity column found on %s.%s; defaulting to 1", schema, table
            )
        except Exception:
            logger.exception("quantity column detection failed")
        return "1"

    def fetch_supplier_data(self, input_data: dict = None) -> pd.DataFrame:
        """Return up-to-date supplier metrics."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                # Line item tables carry the price/quantity information we need
                po_price = self._price_expression(
                    conn, "proc", "po_line_items_agent", "li"
                )
                inv_price = self._price_expression(
                    conn, "proc", "invoice_line_items_agent", "ili"
                )
                po_qty = self._quantity_expression(
                    conn, "proc", "po_line_items_agent", "li"
                )
                inv_qty = self._quantity_expression(
                    conn, "proc", "invoice_line_items_agent", "ili"
                )

                supplier_cols = self._get_columns(conn, "proc", "supplier")

                # Build expression for on-time performance using the delivery
                # lead-time column if present. The column is stored as
                # ``VARCHAR`` in some databases, so we defensively cast only
                # when the value looks numeric to avoid ``DatatypeMismatch``
                # errors.  Non-existent columns result in a constant ``0.0`` to
                # keep the query resilient across database variants.
                if "delivery_lead_time_days" in supplier_cols:
                    on_time_expr = (
                        "CASE "
                        "WHEN s.delivery_lead_time_days ~ '^-?\\d+(\\.\\d+)?$' "
                        "THEN CASE "
                        "WHEN s.delivery_lead_time_days::numeric <= 0 "
                        "THEN 1.0 ELSE 0.0 END "
                        "ELSE 0.0 END AS on_time_pct"
                    )
                else:
                    on_time_expr = "0.0 AS on_time_pct"

                # Select only supplier fields that actually exist in the
                # database. ``supplier_id`` and ``supplier_name`` are already
                # projected earlier in the query, so they are skipped here to
                # avoid duplicate columns in the result set.
                additional = [
                    f"s.{c}" for c in SUPPLIER_FIELDS
                    if c in supplier_cols and c not in {"supplier_id", "supplier_name"}
                ]
                additional_fields = (
                    ",\n                    " + ",\n                    ".join(additional)
                    if additional
                    else ""
                )

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
                           COUNT(DISTINCT i.invoice_id) AS invoice_count
                    FROM proc.invoice_agent i
                    LEFT JOIN proc.invoice_line_items_agent ili ON i.invoice_id = ili.invoice_id

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
                    {on_time_expr}{additional_fields}
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
