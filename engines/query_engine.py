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
from contextlib import contextmanager
from difflib import SequenceMatcher

import pandas as pd

from .base_engine import BaseEngine
from utils.db import read_sql_compat
from utils.gpu import configure_gpu
import services.data_flow_manager as data_flow_module
import services.rag_service as rag_module
import services.supplier_relationship_service as supplier_rel_module

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


PROCUREMENT_CATEGORY_FIELDS = [
    "product",
    "category_level_1",
    "category_level_2",
    "category_level_3",
    "category_level_4",
    "category_level_5",
]

_PRODUCT_SIMILARITY_THRESHOLD = 0.35

# Canonical procurement tables that should be embedded for supplier-aware RAG
_PROCUREMENT_TABLE_SOURCES: dict[str, tuple[str, str]] = {
    "contracts": ("proc", "contracts"),
    "supplier_master": ("proc", "supplier"),
    "purchase_orders": ("proc", "purchase_order_agent"),
    "purchase_order_lines": ("proc", "po_line_items_agent"),
    "invoices": ("proc", "invoice_agent"),
    "invoice_lines": ("proc", "invoice_line_items_agent"),
    "product_mapping": ("proc", "cat_product_mapping"),
    "quotes": ("proc", "quote_agent"),
    "quote_lines": ("proc", "quote_line_items_agent"),
}

_TABLE_SAMPLE_LIMIT = 10000


class QueryEngine(BaseEngine):
    def __init__(self, agent_nick):
        super().__init__()
        self.agent_nick = agent_nick

    @contextmanager
    def _pandas_reader(self):
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        if callable(pandas_conn):
            with pandas_conn() as conn:
                yield conn
                return
        with self.agent_nick.get_db_connection() as conn:
            yield conn

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
        """Return up-to-date supplier metrics.

        ``input_data`` may contain ``supplier_candidates`` and
        ``supplier_directory`` entries originating from the
        ``OpportunityMinerAgent``.  When provided the result set is filtered so
        that downstream agents only receive rows linked to the opportunity flow
        and any missing suppliers are synthesised from the directory payload.
        """

        candidate_ids: set[str] = set()
        directory_lookup: dict[str, dict] = {}
        if input_data:
            raw_candidates = input_data.get("supplier_candidates") or []
            for cand in raw_candidates:
                if cand is None:
                    continue
                cand_str = str(cand).strip()
                if cand_str:
                    candidate_ids.add(cand_str)

            directory_entries = input_data.get("supplier_directory") or []
            for entry in directory_entries:
                if not isinstance(entry, dict):
                    continue
                supplier_id = entry.get("supplier_id")
                if supplier_id is None:
                    continue
                sid = str(supplier_id).strip()
                if not sid:
                    continue
                directory_lookup[sid] = entry

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
                supplier_cols_set = set(supplier_cols)

                # Determine which supplier columns can safely be projected from
                # ``proc.supplier``. ``supplier_id`` and ``supplier_name`` are
                # required for downstream joins. Additional fields are only
                # included when they exist in the current database so we avoid
                # referencing stale or renamed columns.
                supplier_projection: list[str] = []
                for mandatory in ("supplier_id", "supplier_name"):
                    if mandatory not in supplier_projection:
                        supplier_projection.append(mandatory)
                for field in SUPPLIER_FIELDS:
                    if (
                        field in supplier_cols_set
                        and field not in {"supplier_id", "supplier_name"}
                        and field not in supplier_projection
                    ):
                        supplier_projection.append(field)

                supplier_lookup_select_parts = [
                    f"src.{field}" for field in supplier_projection
                ]
                supplier_lookup_select_parts.append(
                    "LOWER(NULLIF(BTRIM(src.supplier_name), '')) AS supplier_name_norm"
                )
                supplier_lookup_select = ",\n                           ".join(
                    supplier_lookup_select_parts
                )

                # Build expression for on-time performance using the delivery
                # lead-time column if present. The column is stored as
                # ``VARCHAR`` in some databases, so we defensively cast only
                # when the value looks numeric to avoid ``DatatypeMismatch``
                # errors.  Non-existent columns result in a constant ``0.0`` to
                # keep the query resilient across database variants.
                if "delivery_lead_time_days" in supplier_cols_set:
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
                extra_output_fields = [
                    f"s.{field}"
                    for field in supplier_projection
                    if field not in {"supplier_id", "supplier_name"}
                ]
                additional_fields = (
                    ",\n                    " + ",\n                    ".join(extra_output_fields)
                    if extra_output_fields
                    else ""
                )

                sql = f"""
                WITH supplier_lookup AS (
                    SELECT {supplier_lookup_select}
                    FROM proc.supplier src
                ), po AS (
                    SELECT LOWER(NULLIF(BTRIM(p.supplier_name), '')) AS supplier_name_norm,
                           SUM({po_price} * {po_qty}) AS po_spend
                    FROM proc.po_line_items_agent li
                    JOIN proc.purchase_order_agent p ON p.po_id = li.po_id
                    WHERE NULLIF(BTRIM(p.supplier_name), '') IS NOT NULL
                    GROUP BY LOWER(NULLIF(BTRIM(p.supplier_name), ''))
                ), inv AS (
                    SELECT LOWER(NULLIF(BTRIM(i.supplier_name), '')) AS supplier_name_norm,
                           SUM({inv_price} * {inv_qty}) AS invoice_spend,
                           COUNT(DISTINCT i.invoice_id) AS invoice_count
                    FROM proc.invoice_agent i
                    LEFT JOIN proc.invoice_line_items_agent ili ON i.invoice_id = ili.invoice_id
                    WHERE NULLIF(BTRIM(i.supplier_name), '') IS NOT NULL
                    GROUP BY LOWER(NULLIF(BTRIM(i.supplier_name), ''))
                )
                SELECT
                    s.supplier_id,
                    s.supplier_name,
                    COALESCE(po.po_spend, 0.0) AS po_spend,
                    COALESCE(inv.invoice_spend, 0.0) AS invoice_spend,
                    COALESCE(po.po_spend, 0.0) + COALESCE(inv.invoice_spend, 0.0) AS total_spend,
                    COALESCE(inv.invoice_count, 0) AS invoice_count,
                    {on_time_expr}{additional_fields}
                FROM supplier_lookup s
                LEFT JOIN po ON po.supplier_name_norm = s.supplier_name_norm
                LEFT JOIN inv ON inv.supplier_name_norm = s.supplier_name_norm
                """
            with self._pandas_reader() as reader:
                df = read_sql_compat(sql, reader)

            base_columns = list(df.columns)

            if "supplier_id" in df.columns:
                df["supplier_id"] = df["supplier_id"].astype(str).str.strip()
            if "supplier_name" in df.columns:
                df["supplier_name"] = df["supplier_name"].astype(str).str.strip()

            if candidate_ids and "supplier_id" in df.columns:
                df = df[df["supplier_id"].isin(candidate_ids)].copy()

            if candidate_ids:
                present = (
                    set(df["supplier_id"].astype(str).str.strip())
                    if "supplier_id" in df.columns
                    else set()
                )
                missing = [cid for cid in candidate_ids if cid not in present]
                if missing:
                    fallback_rows = []
                    for cid in missing:
                        entry = directory_lookup.get(cid, {})
                        row = dict(entry) if isinstance(entry, dict) else {}
                        row["supplier_id"] = cid
                        row.setdefault("supplier_name", entry.get("supplier_name") if isinstance(entry, dict) else None)
                        fallback_rows.append(row)
                    if fallback_rows:
                        fallback_df = pd.DataFrame(fallback_rows)
                        if base_columns:
                            for col in base_columns:
                                if col not in fallback_df.columns:
                                    fallback_df[col] = pd.NA
                            fallback_df = fallback_df[base_columns]
                        df = pd.concat([df, fallback_df], ignore_index=True, sort=False)


            return df
        except Exception as exc:
            # Surface the original exception so callers can handle it explicitly
            logger.exception("fetch_supplier_data failed")
            raise RuntimeError("fetch_supplier_data failed") from exc

    def fetch_invoice_data(self, intent: dict | None = None) -> pd.DataFrame:
        """Return invoice headers from ``proc.invoice_agent`` with supplier IDs."""

        sql = """
            WITH supplier_lookup AS (
                SELECT supplier_id,
                       LOWER(NULLIF(BTRIM(supplier_name), '')) AS supplier_name_norm,
                       supplier_name AS supplier_name_master
                FROM proc.supplier
            )
            SELECT i.*,
                   sl.supplier_id AS supplier_id_lookup,
                   sl.supplier_name_master
            FROM proc.invoice_agent i
            LEFT JOIN supplier_lookup sl
              ON LOWER(NULLIF(BTRIM(i.supplier_name), '')) = sl.supplier_name_norm;
        """

        with self._pandas_reader() as conn:
            df = read_sql_compat(sql, conn)

        if "supplier_id_lookup" in df.columns and "supplier_id" not in df.columns:
            df = df.rename(columns={"supplier_id_lookup": "supplier_id"})
        elif "supplier_id_lookup" in df.columns:
            df["supplier_id"] = df["supplier_id"].combine_first(df["supplier_id_lookup"])
            df = df.drop(columns=["supplier_id_lookup"])

        if "supplier_name_master" in df.columns:
            df["supplier_name"] = df["supplier_name_master"].combine_first(
                df.get("supplier_name")
            )
            df = df.drop(columns=["supplier_name_master"])

        return df

    def fetch_purchase_order_data(self, intent: dict | None = None) -> pd.DataFrame:
        """Return purchase order headers from ``proc.purchase_order_agent`` with supplier IDs."""

        sql = """
            WITH supplier_lookup AS (
                SELECT supplier_id,
                       LOWER(NULLIF(BTRIM(supplier_name), '')) AS supplier_name_norm,
                       supplier_name AS supplier_name_master
                FROM proc.supplier
            )
            SELECT p.*,
                   sl.supplier_id AS supplier_id_lookup,
                   sl.supplier_name_master
            FROM proc.purchase_order_agent p
            LEFT JOIN supplier_lookup sl
              ON LOWER(NULLIF(BTRIM(p.supplier_name), '')) = sl.supplier_name_norm;
        """

        with self._pandas_reader() as conn:
            df = read_sql_compat(sql, conn)

        if "supplier_id_lookup" in df.columns and "supplier_id" not in df.columns:
            df = df.rename(columns={"supplier_id_lookup": "supplier_id"})
        elif "supplier_id_lookup" in df.columns:
            df["supplier_id"] = df["supplier_id"].combine_first(df["supplier_id_lookup"])
            df = df.drop(columns=["supplier_id_lookup"])

        if "supplier_name_master" in df.columns:
            df["supplier_name"] = df["supplier_name_master"].combine_first(
                df.get("supplier_name")
            )
            df = df.drop(columns=["supplier_name_master"])

        return df

    def fetch_procurement_flow(self, embed: bool = False) -> pd.DataFrame:
        """Return enriched procurement data across multiple tables.

        The query implements the following flow:

        1. Retrieve supplier master data from ``proc.supplier`` and normalise
           ``supplier_name`` values for resilient joins.
        2. Map contracts and purchase orders to suppliers using the
           normalised ``supplier_name`` rather than legacy numeric
           identifiers.
        3. For the matching purchase orders, collect line items and invoices
           via ``po_id`` joins (``proc.po_line_items_agent`` and
           ``proc.invoice_agent``).
        4. Intelligently map purchase order line items to canonical product
           and category information from ``proc.cat_product_mapping`` using a
           fuzzy match between ``item_description`` and the mapping table's
           ``product`` field. All category levels ``1``‑``5`` are returned.
        5. Map invoices to ``proc.invoice_line_items_agent`` via ``invoice_id``.

        When ``embed`` is ``True`` a human‑readable summary of the returned
        rows is generated and upserted into the vector database so downstream
        RAG agents can reason over the procurement flow.

        Returns
        -------
        pandas.DataFrame
            A table containing supplier, purchase order, line item, category
            and invoice references for downstream agents.
        """

        sql = """
            WITH supplier_lookup AS (
                SELECT supplier_id,
                       supplier_name,
                       LOWER(NULLIF(BTRIM(supplier_name), '')) AS supplier_name_norm
                FROM proc.supplier
            ),
            contract_supplier AS (
                SELECT c.contract_id,
                       c.supplier_id,
                       sl.supplier_name,
                       sl.supplier_name_norm
                FROM proc.contracts c
                LEFT JOIN supplier_lookup sl ON c.supplier_id = sl.supplier_id
            ),
            contract_supplier_name AS (
                SELECT DISTINCT ON (supplier_name_norm)
                       supplier_name_norm,
                       supplier_id,
                       supplier_name
                FROM contract_supplier
                WHERE supplier_name_norm IS NOT NULL
                ORDER BY supplier_name_norm, contract_id DESC
            ),
            po_raw AS (
                SELECT p.po_id,
                       p.contract_id,
                       p.supplier_name,
                       LOWER(NULLIF(BTRIM(p.supplier_name), '')) AS supplier_name_norm
                FROM proc.purchase_order_agent p
            ),
            po_enriched AS (
                SELECT
                    pr.po_id,
                    COALESCE(cs.contract_id, pr.contract_id) AS contract_id,
                    COALESCE(sl.supplier_id, cs.supplier_id, csn.supplier_id) AS supplier_id,
                    COALESCE(sl.supplier_name, cs.supplier_name, csn.supplier_name, pr.supplier_name) AS supplier_name,
                    pr.supplier_name_norm
                FROM po_raw pr
                LEFT JOIN supplier_lookup sl ON pr.supplier_name_norm = sl.supplier_name_norm
                LEFT JOIN contract_supplier cs ON pr.contract_id = cs.contract_id
                LEFT JOIN contract_supplier_name csn ON pr.supplier_name_norm = csn.supplier_name_norm
            ),
            inv AS (
                SELECT ia.invoice_id, ia.po_id
                FROM proc.invoice_agent ia
            )
            SELECT
                pe.supplier_id,
                pe.supplier_name,
                pe.po_id,
                li.po_line_id,
                li.item_description,
                inv.invoice_id,
                ili.invoice_line_id
            FROM po_enriched pe
            LEFT JOIN proc.po_line_items_agent li ON pe.po_id = li.po_id
            LEFT JOIN inv ON pe.po_id = inv.po_id
            LEFT JOIN proc.invoice_line_items_agent ili
                ON inv.invoice_id = ili.invoice_id
        """

        category_sql = """
            SELECT
                product,
                category_level_1,
                category_level_2,
                category_level_3,
                category_level_4,
                category_level_5
            FROM proc.cat_product_mapping
        """

        with self._pandas_reader() as conn:
            df = read_sql_compat(sql, conn)
            category_df = read_sql_compat(category_sql, conn)

        df = self._assign_procurement_categories(df, category_df)

        if embed and not df.empty:
            self._embed_procurement_summary(df)

        return df

    def _assign_procurement_categories(
        self, df: pd.DataFrame, category_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Map ``item_description`` entries to the closest product categories."""

        if df is None:
            return pd.DataFrame(columns=PROCUREMENT_CATEGORY_FIELDS)

        for field in PROCUREMENT_CATEGORY_FIELDS:
            if field not in df.columns:
                df[field] = pd.NA

        if df.empty or category_df.empty:
            return df

        mapping_rows = []
        for row in category_df.itertuples(index=False, name="CategoryRow"):
            product = getattr(row, "product", None)
            if product is None:
                continue

            product_str = str(product).strip()
            if not product_str:
                continue

            mapping_rows.append((product_str.lower(), row))

        if not mapping_rows:
            return df

        matched_rows = []
        descriptions = df.get("item_description", pd.Series(dtype="object")).fillna("")

        for description in descriptions:
            description_str = str(description).strip()
            if not description_str:
                matched_rows.append(None)
                continue

            desc_key = description_str.lower()
            best_score = 0.0
            best_row = None

            for product_key, mapping_row in mapping_rows:
                score = SequenceMatcher(None, desc_key, product_key).ratio()
                if score > best_score:
                    best_score = score
                    best_row = mapping_row

            if best_score >= _PRODUCT_SIMILARITY_THRESHOLD:
                matched_rows.append(best_row)
            else:
                matched_rows.append(None)

        for field in PROCUREMENT_CATEGORY_FIELDS:
            values = []
            for match in matched_rows:
                if match is None:
                    values.append(None)
                else:
                    values.append(getattr(match, field, None))
            df[field] = values

        return df

    # ------------------------------------------------------------------
    # Procurement summarisation helpers
    # ------------------------------------------------------------------
    def _embed_procurement_summary(self, df: pd.DataFrame) -> None:
        """Create a textual summary of ``df`` and upsert it via ``RAGService``."""
        from services.rag_service import RAGService

        summaries = []
        for row in df.itertuples(index=False):
            categories = " > ".join(
                [
                    getattr(row, f"category_level_{i}")
                    for i in range(1, 6)
                    if getattr(row, f"category_level_{i}")
                ]
            )
            summaries.append(
                f"Supplier {row.supplier_name} (ID {row.supplier_id}) has purchase order "
                f"{row.po_id} item '{row.item_description}' mapped to product {row.product} "
                f"under categories {categories} with invoice {row.invoice_id} "
                f"line {row.invoice_line_id}."
            )

        rag = rag_module.RAGService(self.agent_nick)
        rag.upsert_texts(summaries, metadata={"record_id": "procurement_flow"})

    # ------------------------------------------------------------------
    # Agent training helpers
    # ------------------------------------------------------------------
    def train_procurement_context(self) -> pd.DataFrame:
        """Embed procurement tables, flows and knowledge graph for agent training."""

        rag = rag_module.RAGService(self.agent_nick)
        schema_descriptions: list[str] = []
        tables: dict[str, pd.DataFrame] = {}
        table_name_map: dict[str, str] = {}

        with self._pandas_reader() as conn:
            if conn is None:  # pragma: no cover - defensive safety net
                logger.warning("Database connection unavailable during training")
            for alias, (schema, table) in _PROCUREMENT_TABLE_SOURCES.items():
                canonical = f"{schema}.{table}"
                table_name_map[alias] = canonical

                try:
                    columns = self._get_columns(conn, schema, table) if conn else []
                except Exception:  # pragma: no cover - errors logged upstream
                    columns = []
                if columns:
                    unique_cols = sorted({str(col) for col in columns})
                    schema_descriptions.append(
                        f"Table {canonical} columns: {', '.join(unique_cols)}"
                    )

                sql = f"SELECT * FROM {schema}.{table} LIMIT {_TABLE_SAMPLE_LIMIT}"
                try:
                    df = read_sql_compat(sql, conn) if conn else pd.DataFrame()
                except Exception:
                    logger.exception("Failed to sample data from %s", canonical)
                    continue

                if isinstance(df, pd.DataFrame) and not df.empty:
                    tables[alias] = df

        if schema_descriptions:
            rag.upsert_texts(
                schema_descriptions,
                metadata={
                    "record_id": "procurement_schema",
                    "document_type": "procurement_schema",
                },
            )

        relations: list[dict[str, object]] = []
        graph: dict[str, object] | None = None
        if tables:
            try:
                manager = data_flow_module.DataFlowManager(self.agent_nick)
                relations, graph = manager.build_data_flow_map(
                    tables, table_name_map=table_name_map
                )
                manager.persist_knowledge_graph(relations, graph)
                supplier_flows = (graph or {}).get("supplier_flows") if graph else None
                if supplier_flows:
                    relationship_service = supplier_rel_module.SupplierRelationshipService(self.agent_nick)
                    relationship_service.index_supplier_flows(supplier_flows)
            except Exception:
                logger.exception("Failed to build or persist procurement knowledge graph")

        procurement_df = self.fetch_procurement_flow(embed=True)

        # Return flow data for downstream agents while ensuring training side-effects ran
        return procurement_df
