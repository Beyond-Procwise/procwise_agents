"""Data flow management and knowledge graph persistence for ProcWise agents."""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from qdrant_client import models

from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


# Canonical procurement relationships derived from ``docs/procurement_table_reference.md``.
# ``source``/``target`` reference the table identifiers used by ``OpportunityMinerAgent.TABLE_MAP``
# (and other services) so that the relationships can be evaluated directly against the
# pandas DataFrames loaded at runtime.
PROCUREMENT_RELATIONSHIPS: Tuple[Dict[str, Any], ...] = (
    {
        "source": "contracts",
        "source_column": "supplier_id",
        "target": "supplier_master",
        "target_column": "supplier_id",
        "relationship": "references",
        "description": "Contracts are linked to supplier master records via supplier_id.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "purchase_orders",
        "source_column": "contract_id",
        "target": "contracts",
        "target_column": "contract_id",
        "relationship": "references",
        "description": "Purchase orders reference the contract they were raised against.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "purchase_orders",
        "source_column": "supplier_id",
        "target": "supplier_master",
        "target_column": "supplier_id",
        "relationship": "references",
        "description": "Purchase orders inherit supplier attributes from the supplier master.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "purchase_order_lines",
        "source_column": "po_id",
        "target": "purchase_orders",
        "target_column": "po_id",
        "relationship": "belongs_to",
        "description": "Line items roll up to their parent purchase order via po_id.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "invoice_lines",
        "source_column": "invoice_id",
        "target": "invoices",
        "target_column": "invoice_id",
        "relationship": "belongs_to",
        "description": "Invoice line items aggregate into invoices via invoice_id.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "invoice_lines",
        "source_column": "po_id",
        "target": "purchase_orders",
        "target_column": "po_id",
        "relationship": "reconciles",
        "description": "Invoice lines reconcile back to the originating purchase order.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "invoices",
        "source_column": "po_id",
        "target": "purchase_orders",
        "target_column": "po_id",
        "relationship": "reconciles",
        "description": "Invoices reconcile to the purchase order that triggered them.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "product_mapping",
        "source_column": "product",
        "target": "purchase_order_lines",
        "target_column": "item_description",
        "relationship": "categorises",
        "description": "Product taxonomy enriches PO line descriptions for category analytics.",
        "source_normalizer": "text",
        "target_normalizer": "text",
    },
    {
        "source": "quote_lines",
        "source_column": "quote_id",
        "target": "quotes",
        "target_column": "quote_id",
        "relationship": "belongs_to",
        "description": "Quote line items attach to their header quote records.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
    {
        "source": "quotes",
        "source_column": "po_id",
        "target": "purchase_orders",
        "target_column": "po_id",
        "relationship": "compares",
        "description": "Quotes can be compared against the purchase order ultimately issued.",
        "source_normalizer": "id",
        "target_normalizer": "id",
    },
)

# High level flow paths for reporting and vector persistence.
PROCUREMENT_FLOW_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("contracts", "supplier_master", "purchase_orders", "purchase_order_lines"),
    ("contracts", "purchase_orders", "invoices", "invoice_lines"),
    ("purchase_orders", "purchase_order_lines", "product_mapping"),
    ("purchase_orders", "quotes", "quote_lines"),
    ("purchase_orders", "invoices", "invoice_lines"),
)


@dataclass
class RelationAnalysis:
    """Result of analysing a single relationship between two tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str
    description: Optional[str]
    status: str
    confidence: float
    source_column_resolved: Optional[str] = None
    target_column_resolved: Optional[str] = None
    source_unique_values: int = 0
    target_unique_values: int = 0
    intersection_count: int = 0
    source_coverage: float = 0.0
    target_coverage: float = 0.0
    sample_values: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "source_column_resolved": self.source_column_resolved,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "target_column_resolved": self.target_column_resolved,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "status": self.status,
            "confidence": self.confidence,
            "source_unique_values": self.source_unique_values,
            "target_unique_values": self.target_unique_values,
            "intersection_count": self.intersection_count,
            "source_coverage": self.source_coverage,
            "target_coverage": self.target_coverage,
            "sample_values": list(self.sample_values),
        }


class DataFlowManager:
    """Constructs data flow mappings and persists them as a knowledge graph."""

    def __init__(self, agent_nick: Any, collection_name: str = "Knowledge Graph") -> None:
        self.agent_nick = agent_nick
        self.collection_name = collection_name
        self.qdrant_client = getattr(agent_nick, "qdrant_client", None)
        self.embedding_model = getattr(agent_nick, "embedding_model", None)
        self.settings = getattr(agent_nick, "settings", None)
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        self._collection_ready: bool = False
        self._cached_vector_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_data_flow_map(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyse relationships across ``tables`` and build a knowledge graph."""

        relations: List[RelationAnalysis] = []
        for config in PROCUREMENT_RELATIONSHIPS:
            try:
                relation = self._analyse_relationship(config, tables)
            except Exception:  # pragma: no cover - defensive guardrail
                logger.exception("Failed to analyse relationship %s", config)
                relation = RelationAnalysis(
                    source_table=config["source"],
                    source_column=config["source_column"],
                    target_table=config["target"],
                    target_column=config["target_column"],
                    relationship_type=config.get("relationship", "related_to"),
                    description=config.get("description"),
                    status="error",
                    confidence=0.0,
                )
            relations.append(relation)

        graph = self._build_graph(relations, tables)
        relations_dicts = [relation.as_dict() for relation in relations]
        return relations_dicts, graph

    def persist_knowledge_graph(
        self,
        relations: Iterable[Dict[str, Any]],
        graph: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Vectorise ``relations`` and persist them to Qdrant."""

        if not self.qdrant_client or not self.embedding_model:
            logger.debug(
                "Knowledge graph persistence skipped – Qdrant or embedding model unavailable."
            )
            return

        try:
            if not self._ensure_collection():
                return
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to ensure Qdrant collection for knowledge graph")
            return

        points: List[models.PointStruct] = []
        for relation in relations:
            text = self._relation_to_text(relation)
            vector = self._embed_text(text)
            if vector is None:
                continue
            payload = {
                "document_type": "knowledge_graph_relation",
                "source_table": relation.get("source_table"),
                "target_table": relation.get("target_table"),
                "relationship_type": relation.get("relationship_type"),
                "status": relation.get("status"),
                "confidence": relation.get("confidence"),
                "sample_values": relation.get("sample_values", []),
            }
            if relation.get("description"):
                payload["description"] = relation.get("description")
            point_id = self._deterministic_id(text)
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        if graph:
            for path in graph.get("paths", []):
                if not path:
                    continue
                path_text = self._path_to_text(path)
                vector = self._embed_text(path_text)
                if vector is None:
                    continue
                payload = {
                    "document_type": "knowledge_graph_path",
                    "path": list(path),
                }
                point_id = self._deterministic_id(path_text)
                points.append(
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                )

        if not points:
            logger.debug("No knowledge graph points generated for persistence")
            return

        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception:  # pragma: no cover - network/runtime issues
            logger.exception("Failed to upsert knowledge graph points into Qdrant")

    # ------------------------------------------------------------------
    # Relationship analysis helpers
    # ------------------------------------------------------------------
    def _analyse_relationship(
        self, config: Dict[str, Any], tables: Dict[str, pd.DataFrame]
    ) -> RelationAnalysis:
        source_table = config["source"]
        target_table = config["target"]
        relation_type = config.get("relationship", "related_to")
        description = config.get("description")

        source_df = tables.get(source_table, pd.DataFrame())
        target_df = tables.get(target_table, pd.DataFrame())

        if source_df.empty or target_df.empty:
            status = "missing_data"
            if source_df.empty and target_df.empty:
                status = "missing_tables"
            return RelationAnalysis(
                source_table=source_table,
                source_column=config["source_column"],
                target_table=target_table,
                target_column=config["target_column"],
                relationship_type=relation_type,
                description=description,
                status=status,
                confidence=0.0,
            )

        source_column = self._resolve_column(source_df, config["source_column"], config.get("source_aliases"))
        target_column = self._resolve_column(target_df, config["target_column"], config.get("target_aliases"))

        if source_column is None or target_column is None:
            missing = source_column is None
            status = "missing_column_source" if missing else "missing_column_target"
            return RelationAnalysis(
                source_table=source_table,
                source_column=config["source_column"],
                source_column_resolved=source_column,
                target_table=target_table,
                target_column=config["target_column"],
                target_column_resolved=target_column,
                relationship_type=relation_type,
                description=description,
                status=status,
                confidence=0.0,
            )

        source_series = self._normalise_series(
            source_df[source_column], config.get("source_normalizer", "id")
        )
        target_series = self._normalise_series(
            target_df[target_column], config.get("target_normalizer", "id")
        )

        if source_series.empty or target_series.empty:
            return RelationAnalysis(
                source_table=source_table,
                source_column=config["source_column"],
                source_column_resolved=source_column,
                target_table=target_table,
                target_column=config["target_column"],
                target_column_resolved=target_column,
                relationship_type=relation_type,
                description=description,
                status="insufficient_values",
                confidence=0.0,
            )

        source_unique = set(source_series.unique().tolist())
        target_unique = set(target_series.unique().tolist())
        if not source_unique or not target_unique:
            return RelationAnalysis(
                source_table=source_table,
                source_column=config["source_column"],
                source_column_resolved=source_column,
                target_table=target_table,
                target_column=config["target_column"],
                target_column_resolved=target_column,
                relationship_type=relation_type,
                description=description,
                status="insufficient_values",
                confidence=0.0,
            )

        intersection = source_unique & target_unique
        intersection_count = len(intersection)
        source_coverage = intersection_count / max(1, len(source_unique))
        target_coverage = intersection_count / max(1, len(target_unique))
        confidence = min(1.0, (source_coverage + target_coverage) / 2.0)
        status = "linked" if intersection else "no_overlap"
        samples = tuple(sorted(list(intersection))[:5])

        return RelationAnalysis(
            source_table=source_table,
            source_column=config["source_column"],
            source_column_resolved=source_column,
            target_table=target_table,
            target_column=config["target_column"],
            target_column_resolved=target_column,
            relationship_type=relation_type,
            description=description,
            status=status,
            confidence=confidence,
            source_unique_values=len(source_unique),
            target_unique_values=len(target_unique),
            intersection_count=intersection_count,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            sample_values=samples,
        )

    def _resolve_column(
        self,
        df: pd.DataFrame,
        canonical: str,
        aliases: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        if df.empty:
            return None
        candidates = [canonical]
        if aliases:
            candidates.extend(list(aliases))
        canonical_lower = {str(value).strip().lower(): str(value) for value in candidates}
        for column in df.columns:
            key = str(column).strip().lower()
            if key in canonical_lower:
                return column
        return None

    def _normalise_series(self, series: pd.Series, mode: str) -> pd.Series:
        def _normalise(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            text = str(value).strip()
            if not text:
                return None
            if mode == "id":
                return text.upper()
            if mode == "text":
                return text.lower()
            return text

        normalised = series.map(_normalise)
        normalised = normalised.dropna()
        if normalised.empty:
            return normalised
        if len(normalised) > 5000:
            normalised = normalised.sample(n=5000, random_state=42)
        return normalised

    def _build_graph(
        self,
        relations: Iterable[RelationAnalysis],
        tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        for name, df in tables.items():
            metadata = {
                "type": "table",
                "row_count": int(df.shape[0]),
                "columns": [str(col) for col in df.columns],
            }
            nodes[name] = metadata

        edges = [relation.as_dict() for relation in relations]

        paths: List[Tuple[str, ...]] = []
        for path in PROCUREMENT_FLOW_PATHS:
            if all(table in tables and not tables[table].empty for table in path):
                paths.append(path)

        return {"nodes": nodes, "edges": edges, "paths": paths}

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> bool:
        if not self.qdrant_client or not self.embedding_model:
            return False
        if self._collection_ready:
            return True

        vector_size = self._determine_vector_size()
        if vector_size is None:
            logger.debug("Unable to determine embedding vector size; skipping Qdrant init")
            return False

        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
            self._collection_ready = True
            return True
        except Exception:
            logger.info("Creating Qdrant collection '%s' for knowledge graph", self.collection_name)

        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            for field in ("document_type", "source_table", "target_table", "relationship_type"):
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                        wait=True,
                    )
                except Exception:
                    logger.debug("Failed to create payload index for %s", field, exc_info=True)
            self._collection_ready = True
        except Exception:
            logger.exception("Failed to create knowledge graph collection")
            return False
        return True

    def _determine_vector_size(self) -> Optional[int]:
        if self._cached_vector_size:
            return self._cached_vector_size
        model = self.embedding_model
        if not model:
            return None
        size: Optional[int] = None
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                size = int(getter())
            except Exception:
                logger.debug("Embedding model did not provide dimension", exc_info=True)
        if size is None:
            try:
                vector = self._embed_text("procurement knowledge graph warmup")
                size = len(vector) if vector else None
            except Exception:
                logger.debug("Failed to probe embedding vector size", exc_info=True)
                size = None
        self._cached_vector_size = size
        return size

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        model = self.embedding_model
        if not model or not hasattr(model, "encode"):
            return None
        try:
            vector = model.encode(text, normalize_embeddings=True)
        except TypeError:
            vector = model.encode(text)  # type: ignore[call-arg]
        except Exception:
            logger.exception("Embedding generation failed for knowledge graph text")
            return None
        if vector is None:
            return None
        if isinstance(vector, list):
            return [float(value) for value in vector]
        if hasattr(vector, "tolist"):
            return [float(value) for value in vector.tolist()]
        return None

    def _deterministic_id(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _relation_to_text(self, relation: Dict[str, Any]) -> str:
        source = relation.get("source_table")
        target = relation.get("target_table")
        source_column = relation.get("source_column_resolved") or relation.get("source_column")
        target_column = relation.get("target_column_resolved") or relation.get("target_column")
        relation_type = relation.get("relationship_type", "related_to")
        description = relation.get("description")
        status = relation.get("status")
        confidence = relation.get("confidence")
        samples = relation.get("sample_values") or []
        parts = [
            f"{source}.{source_column} {relation_type} {target}.{target_column}",
        ]
        if description:
            parts.append(description)
        if confidence is not None:
            parts.append(f"confidence {confidence:.2f}")
        if samples:
            joined = ", ".join(str(value) for value in samples)
            parts.append(f"shared keys: {joined}")
        if status:
            parts.append(f"status: {status}")
        return ". ".join(parts)

    def _path_to_text(self, path: Sequence[str]) -> str:
        sequence = " -> ".join(path)
        return f"Procurement data flows through {sequence}."
