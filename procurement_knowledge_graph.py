"""Procurement knowledge graph builder.

This module extracts procurement data from PostgreSQL, projects the
records into a Neo4j knowledge graph, and indexes rich textual payloads
in Qdrant for semantic search. It is designed to be idempotent and safe
for production use. Subsequent runs only process data that changed since
the last successful sync, enabling fast refresh cycles and resumability.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase, Transaction
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import requests

LOGGER = logging.getLogger("procurement_knowledge_graph")
logging.basicConfig(
    level=os.environ.get("PROCWISE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


SYNC_TABLE_NAME = "procwise_sync_state"
EMBED_COLLECTION_NAME = "procwise_procurement_embeddings"
EMBED_VECTOR_SIZE = 1536


class EnvironmentConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass
class SyncCheckpoint:
    entity: str
    updated_after: Optional[datetime]


@dataclass
class ProcurementRecord:
    """Represents a record loaded from Postgres for graph ingestion."""

    label: str
    key: str
    properties: Dict[str, Any]
    relationships: List[Tuple[str, str, str, Dict[str, Any]]] = field(default_factory=list)
    embedding_chunks: List[Tuple[str, str, str]] = field(default_factory=list)
    updated_at: Optional[datetime] = None


class PostgresSource:
    """Loads procurement records from PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
        self._conn.autocommit = True
        LOGGER.debug("Connected to Postgres using DSN %s", dsn)
        self._ensure_sync_table()

    def _ensure_sync_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SYNC_TABLE_NAME} (
                    entity TEXT PRIMARY KEY,
                    last_synced_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        LOGGER.debug("Ensured sync table %s exists", SYNC_TABLE_NAME)

    def fetch_checkpoint(self, entity: str) -> SyncCheckpoint:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT last_synced_at FROM {SYNC_TABLE_NAME} WHERE entity = %s",
                (entity,),
            )
            row = cur.fetchone()
        checkpoint = SyncCheckpoint(entity=entity, updated_after=row["last_synced_at"] if row else None)
        LOGGER.debug("Checkpoint for %s -> %s", entity, checkpoint.updated_after)
        return checkpoint

    def update_checkpoint(self, entity: str, timestamp: datetime) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SYNC_TABLE_NAME} (entity, last_synced_at, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (entity) DO UPDATE SET
                    last_synced_at = EXCLUDED.last_synced_at,
                    updated_at = NOW();
                """,
                (entity, timestamp),
            )
        LOGGER.debug("Updated checkpoint for %s to %s", entity, timestamp)

    def _format_timestamp_condition(self, checkpoint: SyncCheckpoint) -> Tuple[str, Tuple[Any, ...]]:
        if checkpoint.updated_after is None:
            return "", tuple()
        return "WHERE updated_at > %s", (checkpoint.updated_after,)

    def _iter_query(self, query: str, params: Tuple[Any, ...]) -> Iterator[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur:
                yield dict(row)

    def suppliers(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT supplier_id, name, description, risk_rating, performance_score,
                   coverage_regions, preferred, updated_at
            FROM suppliers
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"supplier:{row['supplier_id']}"
            props = {
                "supplier_id": row["supplier_id"],
                "name": row["name"],
                "description": row.get("description"),
                "risk_rating": row.get("risk_rating"),
                "performance_score": row.get("performance_score"),
                "coverage_regions": row.get("coverage_regions"),
                "preferred": bool(row.get("preferred")),
                "updated_at": row.get("updated_at"),
            }
            chunks = []
            if row.get("name"):
                chunks.append((key, "name", row["name"]))
            if row.get("description"):
                chunks.append((key, "description", row["description"]))
            yield ProcurementRecord(
                label="Supplier",
                key=key,
                properties=props,
                relationships=[],
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def contracts(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT contract_id, supplier_id, contract_number, start_date, end_date,
                   total_value, terms_summary, updated_at
            FROM contracts
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"contract:{row['contract_id']}"
            props = {
                "contract_id": row["contract_id"],
                "contract_number": row["contract_number"],
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "total_value": float(row["total_value"]) if row.get("total_value") is not None else None,
                "terms_summary": row.get("terms_summary"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("supplier_id"):
                relationships.append(
                    (
                        key,
                        "HAS_CONTRACT",
                        f"supplier:{row['supplier_id']}",
                        {
                            "start_date": row.get("start_date"),
                            "end_date": row.get("end_date"),
                            "total_value": props["total_value"],
                        },
                    )
                )
            chunks = []
            if row.get("contract_number"):
                chunks.append((key, "contract_number", row["contract_number"]))
            if row.get("terms_summary"):
                chunks.append((key, "terms_summary", row["terms_summary"]))
            yield ProcurementRecord(
                label="Contract",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def purchase_orders(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT po_id, supplier_id, contract_id, order_number, category_id,
                   product_id, status, total_amount, issued_at, delivery_due,
                   updated_at
            FROM purchase_orders
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"po:{row['po_id']}"
            props = {
                "po_id": row["po_id"],
                "order_number": row["order_number"],
                "status": row.get("status"),
                "total_amount": float(row["total_amount"]) if row.get("total_amount") is not None else None,
                "issued_at": row.get("issued_at"),
                "delivery_due": row.get("delivery_due"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("supplier_id"):
                relationships.append((key, "ISSUED_TO", f"supplier:{row['supplier_id']}", {"issued_at": row.get("issued_at")}))
            if row.get("contract_id"):
                relationships.append((key, "FULFILLS", f"contract:{row['contract_id']}", {"status": row.get("status")}))
            if row.get("category_id"):
                relationships.append((key, "ORDERED_UNDER", f"category:{row['category_id']}", {}))
            if row.get("product_id"):
                relationships.append((key, "INCLUDES", f"product:{row['product_id']}", {}))
            chunks = []
            if row.get("order_number"):
                chunks.append((key, "order_number", row["order_number"]))
            yield ProcurementRecord(
                label="PurchaseOrder",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def quotes(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT quote_id, supplier_id, negotiation_session_id, total_amount,
                   currency, validity_date, notes, updated_at
            FROM quotes
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"quote:{row['quote_id']}"
            props = {
                "quote_id": row["quote_id"],
                "total_amount": float(row["total_amount"]) if row.get("total_amount") is not None else None,
                "currency": row.get("currency"),
                "validity_date": row.get("validity_date"),
                "notes": row.get("notes"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("supplier_id"):
                relationships.append((key, "PROVIDED_BY", f"supplier:{row['supplier_id']}", {}))
            if row.get("negotiation_session_id"):
                relationships.append(
                    (
                        key,
                        "NEGOTIATED_IN",
                        f"negotiation:{row['negotiation_session_id']}",
                        {"validity_date": row.get("validity_date")},
                    )
                )
            chunks = []
            if row.get("notes"):
                chunks.append((key, "notes", row["notes"]))
            yield ProcurementRecord(
                label="Quote",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def invoices(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT invoice_id, po_id, invoice_number, amount_due, currency,
                   due_date, paid_at, status, updated_at
            FROM invoices
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"invoice:{row['invoice_id']}"
            props = {
                "invoice_id": row["invoice_id"],
                "invoice_number": row["invoice_number"],
                "amount_due": float(row["amount_due"]) if row.get("amount_due") is not None else None,
                "currency": row.get("currency"),
                "due_date": row.get("due_date"),
                "paid_at": row.get("paid_at"),
                "status": row.get("status"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("po_id"):
                relationships.append((key, "INVOICES", f"po:{row['po_id']}", {"status": row.get("status")}))
            chunks = []
            if row.get("invoice_number"):
                chunks.append((key, "invoice_number", row["invoice_number"]))
            yield ProcurementRecord(
                label="Invoice",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def negotiation_sessions(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT negotiation_session_id, workflow_id, rfq_reference, round_number,
                   started_at, closed_at, status, updated_at
            FROM negotiation_sessions
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"negotiation:{row['negotiation_session_id']}"
            props = {
                "negotiation_session_id": row["negotiation_session_id"],
                "workflow_id": row.get("workflow_id"),
                "rfq_reference": row.get("rfq_reference"),
                "round_number": row.get("round_number"),
                "started_at": row.get("started_at"),
                "closed_at": row.get("closed_at"),
                "status": row.get("status"),
                "updated_at": row.get("updated_at"),
            }
            chunks = []
            if row.get("rfq_reference"):
                chunks.append((key, "rfq_reference", row["rfq_reference"]))
            yield ProcurementRecord(
                label="NegotiationSession",
                key=key,
                properties=props,
                relationships=[],
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def supplier_responses(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT response_id, negotiation_session_id, supplier_id, round_number,
                   response_text, proposed_amount, received_at, updated_at
            FROM supplier_responses
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"response:{row['response_id']}"
            props = {
                "response_id": row["response_id"],
                "round_number": row.get("round_number"),
                "proposed_amount": float(row["proposed_amount"]) if row.get("proposed_amount") is not None else None,
                "received_at": row.get("received_at"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("negotiation_session_id"):
                relationships.append(
                    (
                        key,
                        "RESPONDS_TO",
                        f"negotiation:{row['negotiation_session_id']}",
                        {"received_at": row.get("received_at")},
                    )
                )
            if row.get("supplier_id"):
                relationships.append(
                    (key, "FROM_SUPPLIER", f"supplier:{row['supplier_id']}", {}),
                )
            chunks = []
            if row.get("response_text"):
                chunks.append((key, "response_text", row["response_text"]))
            yield ProcurementRecord(
                label="SupplierResponse",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def categories(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT category_id, name, description, parent_category_id, updated_at
            FROM categories
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"category:{row['category_id']}"
            props = {
                "category_id": row["category_id"],
                "name": row["name"],
                "description": row.get("description"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("parent_category_id"):
                relationships.append((key, "CHILD_OF", f"category:{row['parent_category_id']}", {}))
            chunks = []
            if row.get("name"):
                chunks.append((key, "name", row["name"]))
            if row.get("description"):
                chunks.append((key, "description", row["description"]))
            yield ProcurementRecord(
                label="Category",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )

    def products(self, checkpoint: SyncCheckpoint) -> Iterator[ProcurementRecord]:
        condition, params = self._format_timestamp_condition(checkpoint)
        query = f"""
            SELECT product_id, name, description, category_id, sku, unit_price,
                   currency, updated_at
            FROM products
            {condition}
            ORDER BY updated_at ASC;
        """
        for row in self._iter_query(query, params):
            key = f"product:{row['product_id']}"
            props = {
                "product_id": row["product_id"],
                "name": row["name"],
                "description": row.get("description"),
                "sku": row.get("sku"),
                "unit_price": float(row["unit_price"]) if row.get("unit_price") is not None else None,
                "currency": row.get("currency"),
                "updated_at": row.get("updated_at"),
            }
            relationships = []
            if row.get("category_id"):
                relationships.append((key, "BELONGS_TO", f"category:{row['category_id']}", {}))
            chunks = []
            if row.get("name"):
                chunks.append((key, "name", row["name"]))
            if row.get("description"):
                chunks.append((key, "description", row["description"]))
            yield ProcurementRecord(
                label="Product",
                key=key,
                properties=props,
                relationships=relationships,
                embedding_chunks=chunks,
                updated_at=row.get("updated_at"),
            )


class Neo4jSink:
    """Handles Neo4j writes."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        LOGGER.debug("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    def upsert_record(self, record: ProcurementRecord) -> None:
        with self._driver.session() as session:
            session.write_transaction(self._merge_record, record)
            for rel in record.relationships:
                session.write_transaction(self._merge_relationship, rel)

    @staticmethod
    def _merge_record(tx: Transaction, record: ProcurementRecord) -> None:
        merge_statement = (
            f"MERGE (n:{record.label} {{key: $key}}) "
            "SET n += $properties, n.key = $key"
        )
        tx.run(merge_statement, key=record.key, properties=_sanitize_properties(record.properties))

    @staticmethod
    def _merge_relationship(tx: Transaction, rel: Tuple[str, str, str, Dict[str, Any]]) -> None:
        start_key, rel_type, end_key, rel_props = rel
        statement = (
            "MATCH (start {key: $start_key})"
            "MATCH (end {key: $end_key})"
            f"MERGE (start)-[r:{rel_type}]->(end)"
            "SET r += $properties"
        )
        tx.run(statement, start_key=start_key, end_key=end_key, properties=_sanitize_properties(rel_props))


class OllamaEmbeddingService:
    """Uses Ollama's embedding API."""

    def __init__(self, host: str, model: str = "nomic-embed-text") -> None:
        self._host = host.rstrip("/")
        self._model = model

    def embed(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Cannot embed empty text")
        response = requests.post(
            f"{self._host}/api/embeddings",
            json={"model": self._model, "input": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        vector = data.get("embedding") or data.get("data", [{}])[0].get("embedding")
        if not vector:
            raise RuntimeError(f"Embedding response missing vector: {data}")
        return vector


class QdrantVectorStore:
    """Manages Qdrant collection and embeddings."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in collections:
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(size=EMBED_VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            optimizers_config=qmodels.OptimizersConfigDiff(memmap_threshold=20000),
        )
        LOGGER.info("Created Qdrant collection %s", self._collection_name)

    def upsert_embeddings(self, payloads: List[qmodels.PointStruct]) -> None:
        if not payloads:
            return
        self._client.upsert(collection_name=self._collection_name, points=payloads)


class ProcurementKnowledgeGraphBuilder:
    """Coordinates data extraction, graph updates, and embedding refresh."""

    ENTITY_METHODS = {
        "suppliers": PostgresSource.suppliers,
        "contracts": PostgresSource.contracts,
        "purchase_orders": PostgresSource.purchase_orders,
        "quotes": PostgresSource.quotes,
        "invoices": PostgresSource.invoices,
        "negotiation_sessions": PostgresSource.negotiation_sessions,
        "supplier_responses": PostgresSource.supplier_responses,
        "categories": PostgresSource.categories,
        "products": PostgresSource.products,
    }

    def __init__(
        self,
        postgres: PostgresSource,
        graph: Neo4jSink,
        qdrant: QdrantVectorStore,
        embed_service: OllamaEmbeddingService,
        batch_size: int = 128,
    ) -> None:
        self._postgres = postgres
        self._graph = graph
        self._qdrant = qdrant
        self._embed_service = embed_service
        self._batch_size = batch_size

    def close(self) -> None:
        self._graph.close()

    def refresh(self, entities: Optional[List[str]] = None) -> None:
        targets = entities or list(self.ENTITY_METHODS.keys())
        for entity in targets:
            LOGGER.info("Refreshing entity %s", entity)
            checkpoint = self._postgres.fetch_checkpoint(entity)
            loader = self.ENTITY_METHODS[entity]
            last_processed: Optional[datetime] = checkpoint.updated_after
            embedding_batch: List[qmodels.PointStruct] = []
            count = 0
            for record in loader(self._postgres, checkpoint):
                self._graph.upsert_record(record)
                count += 1
                last_processed = record.updated_at or last_processed
                embedding_batch.extend(self._build_points(record))
                if len(embedding_batch) >= self._batch_size:
                    self._qdrant.upsert_embeddings(embedding_batch)
                    embedding_batch = []
            if embedding_batch:
                self._qdrant.upsert_embeddings(embedding_batch)
            if last_processed:
                self._postgres.update_checkpoint(entity, last_processed)
            LOGGER.info("Entity %s refresh complete (%d records)", entity, count)

    def _build_points(self, record: ProcurementRecord) -> List[qmodels.PointStruct]:
        points: List[qmodels.PointStruct] = []
        for idx, (key, field_name, text) in enumerate(record.embedding_chunks):
            try:
                vector = self._embed_service.embed(text)
            except Exception:
                LOGGER.exception("Failed to embed %s field %s", key, field_name)
                continue
            payload = {
                "entity_key": key,
                "field": field_name,
                "label": record.label,
                "text": text,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
                "public_name": record.properties.get("name") or record.properties.get("contract_number") or record.properties.get("order_number"),
            }
            points.append(
                qmodels.PointStruct(
                    id=self._stable_point_id(key, field_name, idx),
                    vector=vector,
                    payload=payload,
                )
            )
        return points

    @staticmethod
    def _stable_point_id(key: str, field_name: str, idx: int) -> int:
        composite = f"{key}|{field_name}|{idx}"
        digest = hashlib.sha256(composite.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big')


def _sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in properties.items():
        if isinstance(v, datetime):
            sanitized[k] = v.astimezone(timezone.utc).isoformat()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            sanitized[k] = v
        elif isinstance(v, (list, tuple, set)):
            sanitized[k] = list(v)
        else:
            sanitized[k] = json.dumps(v, default=str)
    return sanitized


def build_default_builder() -> ProcurementKnowledgeGraphBuilder:
    postgres_dsn = os.environ.get("POSTGRES_DSN")
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    ollama_host = os.environ.get("OLLAMA_HOST")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    if not all([postgres_dsn, neo4j_uri, neo4j_user, neo4j_password, qdrant_url, ollama_host]):
        raise EnvironmentConfigError("Missing required environment variables for KG build")

    postgres = PostgresSource(postgres_dsn)
    graph = Neo4jSink(neo4j_uri, neo4j_user, neo4j_password)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    prefix = os.environ.get("QDRANT_COLLECTION_PREFIX", "")
    collection_name = prefix + EMBED_COLLECTION_NAME
    qdrant_store = QdrantVectorStore(qdrant_client, collection_name)
    embed_service = OllamaEmbeddingService(ollama_host, embed_model)
    return ProcurementKnowledgeGraphBuilder(postgres, graph, qdrant_store, embed_service)


def main(argv: Optional[List[str]] = None) -> None:
    args = argv or sys.argv[1:]
    start = time.time()
    builder = build_default_builder()
    if args:
        builder.refresh(args)
    else:
        builder.refresh()
    builder.close()
    LOGGER.info("Knowledge graph refresh finished in %.2fs", time.time() - start)


if __name__ == "__main__":
    main()
