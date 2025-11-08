from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from retrying import retry
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import AppConfig, Neo4jConfig, OllamaConfig, PostgresConfig, QdrantConfig


SCHEMA = "proc"
DEFAULT_BATCH_SIZE = 5_000


@dataclass
class DataFrames:
    suppliers: pd.DataFrame
    contracts: pd.DataFrame
    purchase_orders: pd.DataFrame
    po_line_items: pd.DataFrame
    quotes: pd.DataFrame
    quote_line_items: pd.DataFrame
    invoices: pd.DataFrame
    negotiation_sessions: pd.DataFrame
    negotiation_session_state: pd.DataFrame
    supplier_responses: pd.DataFrame
    workflow_email_tracking: pd.DataFrame
    categories: pd.DataFrame


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10_000)
    def embed(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self._config.embedding_model, "input": texts}
        url = f"{self._config.base_url}/api/embeddings"
        response = requests.post(url, json=payload, timeout=self._config.timeout)
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            raise ValueError(f"Unexpected embedding response: {data}")
        return [item["embedding"] for item in data["data"]]

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10_000)
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "model": self._config.generate_model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload.update(options)
        url = f"{self._config.base_url}/api/generate"
        response = requests.post(url, json=payload, timeout=self._config.timeout)
        response.raise_for_status()
        data = response.json()
        text = data.get("response")
        if not text:
            raise ValueError(f"Empty response from Ollama: {data}")
        return text


class ProcurementKnowledgeGraph:
    def __init__(
        self,
        postgres_config: PostgresConfig,
        neo4j_config: Neo4jConfig,
        qdrant_config: QdrantConfig,
        ollama_config: OllamaConfig,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._postgres_config = postgres_config
        self._neo4j_config = neo4j_config
        self._qdrant_config = qdrant_config
        self._ollama_config = ollama_config
        self._batch_size = batch_size

        self._engine: Engine = create_engine(self._postgres_config.dsn, pool_size=10, max_overflow=5)
        self._neo4j_driver = GraphDatabase.driver(
            self._neo4j_config.uri,
            auth=(self._neo4j_config.username, self._neo4j_config.password),
            max_connection_pool_size=50,
        )
        self._qdrant = QdrantClient(host=self._qdrant_config.host, port=self._qdrant_config.port, api_key=self._qdrant_config.api_key)
        self._ollama = OllamaClient(self._ollama_config)
        logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])

    @classmethod
    def from_config(cls, config: AppConfig) -> "ProcurementKnowledgeGraph":
        return cls(config.postgres, config.neo4j, config.qdrant, config.ollama)

    def close(self) -> None:
        logger.info("Closing database connections")
        self._engine.dispose()
        self._neo4j_driver.close()

    # -----------------------------------------------------
    # Connection testing
    # -----------------------------------------------------
    def test_connections(self) -> None:
        logger.info("Testing PostgreSQL connection")
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Testing Neo4j connection")
        with self._neo4j_driver.session() as session:
            session.run("RETURN 1")
        logger.info("Testing Qdrant connection")
        self._qdrant.get_collections()
        logger.info("Testing Ollama embedding endpoint")
        self._ollama.embed(["connection test"])

    # -----------------------------------------------------
    # Data extraction
    # -----------------------------------------------------
    def load_dataframes(self, since: Optional[datetime] = None) -> DataFrames:
        filter_clause = ""
        params: Dict[str, Any] = {}
        if since:
            filter_clause = "WHERE updated_at >= :since"
            params["since"] = since

        def load(table: str, extra_where: str = "") -> pd.DataFrame:
            where_clause = extra_where or filter_clause
            query = f"SELECT * FROM {SCHEMA}.{table}"
            if where_clause:
                query = f"{query} {where_clause}"
            logger.info("Loading table {table}", table=table)
            with self._engine.connect() as conn:
                return pd.read_sql_query(text(query), conn, params=params)

        suppliers = load("supplier")
        contracts = load("contracts")
        purchase_orders = load("purchase_order_agent")
        po_line_items = load("po_line_items_agent")
        quotes = load("quote_agent")
        quote_line_items = load("quote_line_items_agent")
        invoices = load("invoice_agent")
        negotiation_sessions = load("negotiation_sessions")
        negotiation_session_state = load("negotiation_session_state")
        supplier_responses = load("supplier_response")
        workflow_email_tracking = load("workflow_email_tracking")
        categories = load("cat_product_mapping")

        return DataFrames(
            suppliers=suppliers,
            contracts=contracts,
            purchase_orders=purchase_orders,
            po_line_items=po_line_items,
            quotes=quotes,
            quote_line_items=quote_line_items,
            invoices=invoices,
            negotiation_sessions=negotiation_sessions,
            negotiation_session_state=negotiation_session_state,
            supplier_responses=supplier_responses,
            workflow_email_tracking=workflow_email_tracking,
            categories=categories,
        )

    # -----------------------------------------------------
    # Neo4j schema management
    # -----------------------------------------------------
    def _execute_cypher(self, statements: Iterable[str]) -> None:
        with self._neo4j_driver.session() as session:
            for statement in statements:
                logger.debug("Executing Cypher: {statement}", statement=statement)
                session.run(statement)

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.supplier_id IS UNIQUE",
            "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE",
            "CREATE CONSTRAINT po_id IF NOT EXISTS FOR (p:PurchaseOrder) REQUIRE p.po_id IS UNIQUE",
            "CREATE CONSTRAINT quote_id IF NOT EXISTS FOR (q:Quote) REQUIRE q.quote_id IS UNIQUE",
            "CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.invoice_id IS UNIQUE",
            "CREATE CONSTRAINT negotiation_key IF NOT EXISTS FOR (n:NegotiationSession) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT supplier_response_key IF NOT EXISTS FOR (r:SupplierResponse) REQUIRE r.unique_id IS UNIQUE",
            "CREATE CONSTRAINT category_key IF NOT EXISTS FOR (c:Category) REQUIRE c.category_path IS UNIQUE",
        ]
        self._execute_cypher(statements)
        self._create_indexes()

    def _create_indexes(self) -> None:
        statements = [
            "CREATE INDEX supplier_name IF NOT EXISTS FOR (s:Supplier) ON (s.supplier_name)",
            "CREATE INDEX supplier_risk IF NOT EXISTS FOR (s:Supplier) ON (s.risk_score)",
            "CREATE INDEX po_status IF NOT EXISTS FOR (p:PurchaseOrder) ON (p.po_status)",
            "CREATE INDEX po_date IF NOT EXISTS FOR (p:PurchaseOrder) ON (p.order_date)",
            "CREATE INDEX quote_date IF NOT EXISTS FOR (q:Quote) ON (q.quote_date)",
            "CREATE INDEX negotiation_status IF NOT EXISTS FOR (n:NegotiationSession) ON (n.status)",
            "CREATE INDEX negotiation_round IF NOT EXISTS FOR (n:NegotiationSession) ON (n.round)",
            "CREATE INDEX supplier_response_workflow IF NOT EXISTS FOR (r:SupplierResponse) ON (r.workflow_id)",
            "CREATE INDEX supplier_response_processed IF NOT EXISTS FOR (r:SupplierResponse) ON (r.processed)",
        ]
        self._execute_cypher(statements)

    def rebuild_indexes(self) -> None:
        statements = [
            "DROP INDEX supplier_name IF EXISTS",
            "DROP INDEX supplier_risk IF EXISTS",
            "DROP INDEX po_status IF EXISTS",
            "DROP INDEX po_date IF EXISTS",
            "DROP INDEX quote_date IF EXISTS",
            "DROP INDEX negotiation_status IF EXISTS",
            "DROP INDEX negotiation_round IF EXISTS",
            "DROP INDEX supplier_response_workflow IF EXISTS",
            "DROP INDEX supplier_response_processed IF EXISTS",
        ]
        self._execute_cypher(statements)
        self._create_indexes()

    # -----------------------------------------------------
    # Graph building
    # -----------------------------------------------------
    def build_graph(self, frames: DataFrames) -> None:
        self.ensure_schema()
        logger.info("Building knowledge graph")
        with self._neo4j_driver.session() as session:
            self._merge_suppliers(session, frames.suppliers)
            self._merge_contracts(session, frames.contracts)
            self._merge_purchase_orders(session, frames.purchase_orders, frames.suppliers)
            self._merge_quotes(session, frames.quotes)
            self._merge_invoices(session, frames.invoices)
            self._merge_negotiations(session, frames.negotiation_sessions, frames.negotiation_session_state)
            self._merge_supplier_responses(session, frames.supplier_responses)
            self._merge_categories(session, frames.categories, frames.po_line_items)
            self._merge_relationships(session, frames)

    def _merge_suppliers(self, session, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (s:Supplier {supplier_id: toString(row.supplier_id)})
        SET s += {
            supplier_name: row.supplier_name,
            risk_score: coalesce(row.risk_score, 0),
            is_preferred_supplier: coalesce(row.is_preferred_supplier, false),
            esg_certifications: row.esg_certifications,
            diversity_flags: row.diversity_flags,
            contact_email: row.contact_email,
            contact_phone: row.contact_phone,
            country: row.country,
            updated_at: datetime(coalesce(row.updated_at, row.created_at))
        }
        """
        session.run(query, rows=records)

    def _merge_contracts(self, session, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (c:Contract {contract_id: toString(row.contract_id)})
        SET c += {
            contract_type: row.contract_type,
            total_contract_value: toFloat(row.total_contract_value),
            contract_lifecycle_status: row.contract_lifecycle_status,
            payment_terms: row.payment_terms,
            start_date: datetime(row.start_date),
            end_date: datetime(row.end_date),
            updated_at: datetime(coalesce(row.updated_at, row.created_at))
        }
        WITH row, c
        MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
        MERGE (s)-[:HAS_CONTRACT]->(c)
        """
        session.run(query, rows=records)

    def _resolve_supplier_id(self, suppliers: pd.DataFrame, supplier_name: str) -> Optional[str]:
        if "supplier_name" not in suppliers.columns or suppliers.empty:
            return None
        matches = suppliers[suppliers["supplier_name"].str.lower() == supplier_name.lower()]
        if not matches.empty:
            return str(matches.iloc[0]["supplier_id"])
        from rapidfuzz import process

        choices = suppliers["supplier_name"].tolist()
        result = process.extractOne(supplier_name, choices, score_cutoff=80)
        if result:
            idx = suppliers[suppliers["supplier_name"] == result[0]].index[0]
            return str(suppliers.iloc[idx]["supplier_id"])
        return None

    def _merge_purchase_orders(self, session, df: pd.DataFrame, suppliers: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        if "supplier_id" not in df.columns:
            df["supplier_id"] = df["supplier_name"].apply(lambda name: self._resolve_supplier_id(suppliers, name) if isinstance(name, str) else None)
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (p:PurchaseOrder {po_id: toString(row.po_id)})
        SET p += {
            supplier_name: row.supplier_name,
            total_amount: toFloat(row.total_amount),
            po_status: row.po_status,
            contract_id: toString(row.contract_id),
            order_date: datetime(row.order_date),
            updated_at: datetime(coalesce(row.updated_at, row.created_at))
        }
        WITH row, p
        OPTIONAL MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:RECEIVED_PO]->(p)
        )
        """
        session.run(query, rows=records)

    def _merge_quotes(self, session, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (q:Quote {quote_id: toString(row.quote_id)})
        SET q += {
            supplier_id: toString(row.supplier_id),
            quote_date: datetime(row.quote_date),
            total_amount_incl_tax: toFloat(row.total_amount_incl_tax),
            validity_date: datetime(row.validity_date),
            updated_at: datetime(coalesce(row.updated_at, row.created_at))
        }
        WITH row, q
        OPTIONAL MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:SUBMITTED_QUOTE]->(q)
        )
        WITH row, q
        OPTIONAL MATCH (p:PurchaseOrder {po_id: toString(row.po_id)})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (q)-[:AWARDED_AS]->(p)
        )
        """
        session.run(query, rows=records)

    def _merge_invoices(self, session, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (i:Invoice {invoice_id: toString(row.invoice_id)})
        SET i += {
            po_id: toString(row.po_id),
            invoice_total_incl_tax: toFloat(row.invoice_total_incl_tax),
            invoice_status: row.invoice_status,
            payment_terms: row.payment_terms,
            issued_on: datetime(row.issued_on),
            updated_at: datetime(coalesce(row.updated_at, row.created_at))
        }
        WITH row, i
        OPTIONAL MATCH (p:PurchaseOrder {po_id: toString(row.po_id)})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p)-[:HAS_INVOICE]->(i)
        )
        """
        session.run(query, rows=records)

    def _merge_negotiations(self, session, sessions_df: pd.DataFrame, state_df: pd.DataFrame) -> None:
        if sessions_df.empty and state_df.empty:
            return
        combined = sessions_df.copy()
        if not state_df.empty:
            state_df = state_df.rename(columns={"current_round": "current_round_state"})
            combined = combined.merge(state_df, how="left", on=["rfq_id", "supplier_id"], suffixes=("", "_state"))
        records = combined.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        WITH row,
            toString(row.rfq_id) AS rfq_id,
            toString(row.supplier_id) AS supplier_id,
            toInteger(coalesce(row.round, row.current_round_state, 0)) AS round
        WITH row, rfq_id, supplier_id, round,
            rfq_id + '|' + supplier_id + '|' + toString(round) AS key
        MERGE (n:NegotiationSession {key: key})
        SET n += {
            rfq_id: rfq_id,
            supplier_id: supplier_id,
            round: round,
            counter_offer: toFloat(row.counter_offer),
            created_on: datetime(row.created_on),
            supplier_reply_count: toInteger(coalesce(row.supplier_reply_count, 0)),
            status: coalesce(row.status, 'pending'),
            awaiting_response: coalesce(row.awaiting_response, false)
        }
        WITH row, n
        OPTIONAL MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:NEGOTIATED_IN]->(n)
        )
        """
        session.run(query, rows=records)

    def _merge_supplier_responses(self, session, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.fillna(value=pd.NA).to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        MERGE (r:SupplierResponse {unique_id: row.unique_id})
        SET r += {
            workflow_id: row.workflow_id,
            supplier_email: row.supplier_email,
            price: toFloat(row.price),
            payment_terms: row.payment_terms,
            lead_time: row.lead_time,
            round_number: toInteger(row.round_number),
            processed: coalesce(row.processed, false),
            message_id: row.message_id,
            responded_at: datetime(row.responded_at)
        }
        WITH row, r
        MATCH (n:NegotiationSession {rfq_id: toString(row.rfq_id), supplier_id: toString(row.supplier_id)})
        MERGE (r)-[:RESPONDS_TO]->(n)
        WITH row, r
        OPTIONAL MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:SENT_RESPONSE]->(r)
        )
        """
        session.run(query, rows=records)

    def _merge_categories(self, session, categories: pd.DataFrame, items: pd.DataFrame) -> None:
        if categories.empty:
            return
        cat_records = categories.fillna(value="").to_dict(orient="records")
        query = """
        UNWIND $rows AS row
        WITH row,
            [row.category_level_1, row.category_level_2, row.category_level_3, row.category_level_4, row.category_level_5] AS levels,
            row.product AS product
        WITH row, [level IN levels WHERE level IS NOT NULL AND level <> ''] AS filtered, product
        WITH row, filtered, product,
            [category IN filtered | toLower(category)] AS normalized
        WITH row, product, normalized,
            apoc.text.join(normalized, '>') AS path
        MERGE (root:Category {category_path: path})
        SET root += {levels: normalized}
        WITH row, product, root
        MERGE (i:Item {item_description: row.product})
        MERGE (i)-[:BELONGS_TO_CATEGORY]->(root)
        """
        session.run(query, rows=cat_records)

    def _merge_relationships(self, session, frames: DataFrames) -> None:
        if not frames.po_line_items.empty:
            po_items = frames.po_line_items.fillna(value=pd.NA).to_dict(orient="records")
            query = """
            UNWIND $rows AS row
            MATCH (p:PurchaseOrder {po_id: toString(row.po_id)})
            MERGE (i:Item {item_description: row.item_description})
            MERGE (p)-[rel:CONTAINS_ITEM {line_number: toInteger(row.line_number)}]->(i)
            SET rel.quantity = toFloat(row.quantity), rel.unit_price = toFloat(row.unit_price), rel.total_amount = toFloat(row.total_amount)
            """
            session.run(query, rows=po_items)

        if not frames.workflow_email_tracking.empty:
            wf_records = frames.workflow_email_tracking.fillna(value=pd.NA).to_dict(orient="records")
            query = """
            UNWIND $rows AS row
            MERGE (w:Workflow {workflow_id: row.workflow_id})
            SET w.unique_id = row.unique_id, w.dispatched_at = datetime(row.dispatched_at), w.responded_at = datetime(row.responded_at)
            WITH row, w
            OPTIONAL MATCH (s:Supplier {supplier_id: toString(row.supplier_id)})
            FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
                MERGE (s)-[:PART_OF_WORKFLOW]->(w)
            )
            WITH row, w
            OPTIONAL MATCH (r:SupplierResponse {unique_id: row.unique_id})
            FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
                MERGE (w)-[:TRACKS_RESPONSE]->(r)
            )
            """
            session.run(query, rows=wf_records)

    # -----------------------------------------------------
    # Embeddings + Qdrant
    # -----------------------------------------------------
    def _ensure_collections(self) -> None:
        collections = self._qdrant.get_collections().collections
        existing = {c.name for c in collections}
        schema = {
            "suppliers": qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
            "contracts": qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
            "negotiations": qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        }
        for name, params in schema.items():
            if name not in existing:
                logger.info("Creating Qdrant collection {name}", name=name)
                self._qdrant.create_collection(name=name, vectors_config=params)

    def _prepare_supplier_payloads(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        descriptions = []
        payloads = []
        for row in df.fillna(value="").to_dict(orient="records"):
            description = (
                f"Supplier: {row.get('supplier_name','')} | Type: {row.get('supplier_type','unknown')} | "
                f"Country: {row.get('country','')} | Risk Score: {row.get('risk_score', 0)} | "
                f"Certifications: {row.get('esg_certifications','')} | Diversity: {row.get('diversity_flags','')} | "
                f"Lead Time: {row.get('avg_lead_time_days', 'n/a')} days"
            )
            descriptions.append(description)
            payloads.append(
                {
                    "supplier_id": str(row.get("supplier_id")),
                    "name": row.get("supplier_name"),
                    "type": row.get("supplier_type"),
                    "country": row.get("country"),
                    "risk_score": row.get("risk_score"),
                    "certifications": row.get("esg_certifications"),
                    "diversity_flags": row.get("diversity_flags"),
                    "lead_time_days": row.get("avg_lead_time_days"),
                }
            )
        return descriptions, payloads

    def _prepare_contract_payloads(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        descriptions = []
        payloads = []
        for row in df.fillna(value="").to_dict(orient="records"):
            description = (
                f"Contract {row.get('contract_id')} for {row.get('contract_type')} worth {row.get('total_contract_value')} "
                f"status {row.get('contract_lifecycle_status')} terms {row.get('payment_terms')}"
            )
            descriptions.append(description)
            payloads.append(
                {
                    "contract_id": str(row.get("contract_id")),
                    "title": row.get("contract_type"),
                    "type": row.get("contract_type"),
                    "value": row.get("total_contract_value"),
                    "status": row.get("contract_lifecycle_status"),
                }
            )
        return descriptions, payloads

    def _prepare_negotiation_payloads(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        descriptions = []
        payloads = []
        for row in df.fillna(value="").to_dict(orient="records"):
            description = (
                f"Negotiation RFQ {row.get('rfq_id')} supplier {row.get('supplier_id')} round {row.get('round')} "
                f"counter {row.get('counter_offer')} status {row.get('status')} awaiting {row.get('awaiting_response')}"
            )
            descriptions.append(description)
            payloads.append(
                {
                    "rfq_id": str(row.get("rfq_id")),
                    "supplier_id": str(row.get("supplier_id")),
                    "round": row.get("round"),
                    "counter_offer": row.get("counter_offer"),
                    "status": row.get("status"),
                    "awaiting_response": row.get("awaiting_response"),
                }
            )
        return descriptions, payloads

    def sync_embeddings(self, frames: DataFrames) -> None:
        self._ensure_collections()
        collections = [
            ("suppliers", *self._prepare_supplier_payloads(frames.suppliers)),
            ("contracts", *self._prepare_contract_payloads(frames.contracts)),
            ("negotiations", *self._prepare_negotiation_payloads(frames.negotiation_sessions)),
        ]
        for name, descriptions, payloads in collections:
            if not descriptions:
                continue
            logger.info("Generating embeddings for {name}", name=name)
            vectors = self._ollama.embed(descriptions)
            points = []
            for idx, vector in enumerate(vectors):
                payload = payloads[idx]
                point_id = payload.get("supplier_id") or payload.get("contract_id") or f"{payload.get('rfq_id')}:{payload.get('supplier_id')}:{payload.get('round')}"
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
            logger.info("Upserting {count} vectors into {name}", count=len(points), name=name)
            self._qdrant.upsert(collection_name=name, points=points)

    # -----------------------------------------------------
    # Utility methods
    # -----------------------------------------------------
    def full_refresh(self) -> None:
        frames = self.load_dataframes()
        self.build_graph(frames)
        self.sync_embeddings(frames)

    def incremental_refresh(self, since: datetime) -> None:
        frames = self.load_dataframes(since=since)
        self.build_graph(frames)
        self.sync_embeddings(frames)

    # -----------------------------------------------------
    # CLI entry point
    # -----------------------------------------------------
    @staticmethod
    def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Procurement knowledge graph builder")
        parser.add_argument("--full-refresh", action="store_true", help="Execute a full rebuild")
        parser.add_argument("--incremental", action="store_true", help="Run incremental load")
        parser.add_argument("--since", type=str, help="Timestamp for incremental load in ISO format")
        parser.add_argument("--rebuild-indexes", action="store_true", help="Rebuild Neo4j indexes")
        return parser.parse_args(argv)

    @classmethod
    def cli(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        config = AppConfig.from_env()
        builder = cls.from_config(config)
        try:
            if args.rebuild_indexes:
                builder.rebuild_indexes()
                logger.info("Indexes rebuilt")
            elif args.full_refresh:
                logger.info("Running full refresh")
                builder.full_refresh()
            elif args.incremental:
                if not args.since:
                    raise ValueError("--since is required for incremental refresh")
                since = datetime.fromisoformat(args.since)
                logger.info("Running incremental refresh since {since}", since=since)
                builder.incremental_refresh(since)
            else:
                cls._parse_args(["-h"])
        finally:
            builder.close()


if __name__ == "__main__":
    ProcurementKnowledgeGraph.cli()
