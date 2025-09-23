"""Data flow management and knowledge graph persistence for ProcWise agents."""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
from qdrant_client import models

from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationshipConfig:
    """Configuration describing how two procurement tables relate."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship: str
    description: str
    source_normalizer: str = "id"
    target_normalizer: str = "id"
    source_aliases: Tuple[str, ...] = ()
    target_aliases: Tuple[str, ...] = ()
    source_column_aliases: Tuple[str, ...] = ()
    target_column_aliases: Tuple[str, ...] = ()


@dataclass
class TableAliasIndex:
    """Index translating canonical procurement tables to runtime aliases."""

    canonical_to_aliases: Dict[str, Tuple[str, ...]]
    alias_to_canonical: Dict[str, str]

    def resolve_alias(self, canonical: str, tables: Mapping[str, pd.DataFrame]) -> Optional[str]:
        for alias in self.canonical_to_aliases.get(canonical, ()):  # pragma: no branch - deterministic order
            if alias in tables:
                return alias
        return None

    def canonical_name(self, alias: str) -> str:
        return self.alias_to_canonical.get(alias, alias)


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
    source_table_alias: Optional[str] = None
    target_table_alias: Optional[str] = None
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
            "source_table_alias": self.source_table_alias,
            "source_column": self.source_column,
            "source_column_resolved": self.source_column_resolved,
            "target_table": self.target_table,
            "target_table_alias": self.target_table_alias,
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


# Canonical procurement relationships derived from ``docs/procurement_table_reference.md``.
PROCUREMENT_TABLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "proc.contracts": ("contracts",),
    "proc.supplier": ("supplier_master", "suppliers"),
    "proc.purchase_order_agent": ("purchase_orders", "purchase_order_agent"),
    "proc.po_line_items_agent": ("purchase_order_lines", "po_line_items"),
    "proc.invoice_agent": ("invoices", "invoice_agent"),
    "proc.invoice_line_items_agent": ("invoice_lines", "invoice_line_items"),
    "proc.cat_product_mapping": ("product_mapping", "cat_product_mapping"),
    "proc.quote_agent": ("quotes", "quote_agent"),
    "proc.quote_line_items_agent": ("quote_lines", "quote_line_items"),
}

PROCUREMENT_RELATIONSHIPS: Tuple[RelationshipConfig, ...] = (
    RelationshipConfig(
        source_table="proc.contracts",
        source_column="supplier_id",
        target_table="proc.supplier",
        target_column="supplier_id",
        relationship="references",
        description="Contracts are linked to supplier master records via supplier_id.",
        source_aliases=("contracts",),
        target_aliases=("supplier_master", "suppliers"),
    ),
    RelationshipConfig(
        source_table="proc.purchase_order_agent",
        source_column="contract_id",
        target_table="proc.contracts",
        target_column="contract_id",
        relationship="references",
        description="Purchase orders reference the contract they were raised against.",
        source_aliases=("purchase_orders",),
        target_aliases=("contracts",),
    ),
    RelationshipConfig(
        source_table="proc.purchase_order_agent",
        source_column="supplier_id",
        target_table="proc.supplier",
        target_column="supplier_id",
        relationship="references",
        description="Purchase orders inherit supplier attributes from the supplier master.",
        source_aliases=("purchase_orders",),
        target_aliases=("supplier_master", "suppliers"),
    ),
    RelationshipConfig(
        source_table="proc.po_line_items_agent",
        source_column="po_id",
        target_table="proc.purchase_order_agent",
        target_column="po_id",
        relationship="belongs_to",
        description="Line items roll up to their parent purchase order via po_id.",
        source_aliases=("purchase_order_lines",),
        target_aliases=("purchase_orders",),
    ),
    RelationshipConfig(
        source_table="proc.invoice_line_items_agent",
        source_column="invoice_id",
        target_table="proc.invoice_agent",
        target_column="invoice_id",
        relationship="belongs_to",
        description="Invoice line items aggregate into invoices via invoice_id.",
        source_aliases=("invoice_lines",),
        target_aliases=("invoices",),
    ),
    RelationshipConfig(
        source_table="proc.invoice_line_items_agent",
        source_column="po_id",
        target_table="proc.purchase_order_agent",
        target_column="po_id",
        relationship="reconciles",
        description="Invoice lines reconcile back to the originating purchase order.",
        source_aliases=("invoice_lines",),
        target_aliases=("purchase_orders",),
    ),
    RelationshipConfig(
        source_table="proc.invoice_agent",
        source_column="po_id",
        target_table="proc.purchase_order_agent",
        target_column="po_id",
        relationship="reconciles",
        description="Invoices reconcile to the purchase order that triggered them.",
        source_aliases=("invoices",),
        target_aliases=("purchase_orders",),
    ),
    RelationshipConfig(
        source_table="proc.cat_product_mapping",
        source_column="product",
        target_table="proc.po_line_items_agent",
        target_column="item_description",
        relationship="categorises",
        description="Product taxonomy enriches PO line descriptions for category analytics.",
        source_aliases=("product_mapping",),
        target_aliases=("purchase_order_lines",),
        source_normalizer="text",
        target_normalizer="text",
    ),
    RelationshipConfig(
        source_table="proc.quote_line_items_agent",
        source_column="quote_id",
        target_table="proc.quote_agent",
        target_column="quote_id",
        relationship="belongs_to",
        description="Quote line items attach to their header quote records.",
        source_aliases=("quote_lines",),
        target_aliases=("quotes",),
    ),
    RelationshipConfig(
        source_table="proc.quote_agent",
        source_column="po_id",
        target_table="proc.purchase_order_agent",
        target_column="po_id",
        relationship="compares",
        description="Quotes can be compared against the purchase order ultimately issued.",
        source_aliases=("quotes",),
        target_aliases=("purchase_orders",),
    ),
)

# High level flow paths for reporting and vector persistence (canonical table names).
PROCUREMENT_FLOW_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("proc.contracts", "proc.supplier", "proc.purchase_order_agent", "proc.po_line_items_agent"),
    ("proc.contracts", "proc.purchase_order_agent", "proc.invoice_agent", "proc.invoice_line_items_agent"),
    ("proc.purchase_order_agent", "proc.po_line_items_agent", "proc.cat_product_mapping"),
    ("proc.purchase_order_agent", "proc.quote_agent", "proc.quote_line_items_agent"),
    ("proc.purchase_order_agent", "proc.invoice_agent", "proc.invoice_line_items_agent"),
)


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
        self,
        tables: Dict[str, pd.DataFrame],
        table_name_map: Optional[Mapping[str, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyse relationships across ``tables`` and build a knowledge graph."""

        alias_index = self._build_alias_index(tables, table_name_map)
        relations: List[RelationAnalysis] = []
        for config in PROCUREMENT_RELATIONSHIPS:
            try:
                relation = self._analyse_relationship(config, tables, alias_index)
            except Exception:  # pragma: no cover - defensive guardrail
                logger.exception("Failed to analyse relationship %s", config)
                relation = RelationAnalysis(
                    source_table=config.source_table,
                    source_column=config.source_column,
                    target_table=config.target_table,
                    target_column=config.target_column,
                    relationship_type=config.relationship,
                    description=config.description,
                    status="error",
                    confidence=0.0,
                )
            relations.append(relation)

        graph = self._build_graph(relations, tables, alias_index)
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
            if relation.get("source_table_alias"):
                payload["source_table_alias"] = relation.get("source_table_alias")
            if relation.get("target_table_alias"):
                payload["target_table_alias"] = relation.get("target_table_alias")
            point_id = self._deterministic_id(text)
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        if graph:
            for path_entry in graph.get("paths", []):
                if isinstance(path_entry, dict):
                    canonical_path = tuple(path_entry.get("canonical", ()))
                    resolved_aliases = tuple(path_entry.get("resolved", ()))
                else:  # pragma: no cover - legacy format compatibility
                    canonical_path = tuple(path_entry)
                    resolved_aliases = ()
                if not canonical_path:
                    continue
                path_text = self._path_to_text(canonical_path, resolved_aliases or None)
                vector = self._embed_text(path_text)
                if vector is None:
                    continue
                payload = {
                    "document_type": "knowledge_graph_path",
                    "path": list(canonical_path),
                }
                if resolved_aliases and resolved_aliases != canonical_path:
                    payload["resolved_aliases"] = list(resolved_aliases)

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
        self,
        config: RelationshipConfig,
        tables: Dict[str, pd.DataFrame],
        alias_index: TableAliasIndex,
    ) -> RelationAnalysis:
        source_alias = alias_index.resolve_alias(config.source_table, tables)
        target_alias = alias_index.resolve_alias(config.target_table, tables)
        source_df = tables.get(source_alias, pd.DataFrame()) if source_alias else pd.DataFrame()
        target_df = tables.get(target_alias, pd.DataFrame()) if target_alias else pd.DataFrame()

        if source_df.empty or target_df.empty:
            status = "missing_data"
            if (source_df.empty and source_alias is None) or (target_df.empty and target_alias is None):
                status = "missing_tables"
            return RelationAnalysis(
                source_table=config.source_table,
                source_table_alias=source_alias,
                source_column=config.source_column,
                target_table=config.target_table,
                target_table_alias=target_alias,
                target_column=config.target_column,
                relationship_type=config.relationship,
                description=config.description,
                status=status,
                confidence=0.0,
            )

        source_column = self._resolve_column(
            source_df, config.source_column, config.source_column_aliases
        )
        target_column = self._resolve_column(
            target_df, config.target_column, config.target_column_aliases
        )


        if source_column is None or target_column is None:
            missing = source_column is None
            status = "missing_column_source" if missing else "missing_column_target"
            return RelationAnalysis(
                source_table=config.source_table,
                source_table_alias=source_alias,
                source_column=config.source_column,
                source_column_resolved=source_column,
                target_table=config.target_table,
                target_table_alias=target_alias,
                target_column=config.target_column,
                target_column_resolved=target_column,
                relationship_type=config.relationship,
                description=config.description,
                status=status,
                confidence=0.0,
            )

        source_series = self._normalise_series(
            source_df[source_column], config.source_normalizer
        )
        target_series = self._normalise_series(
            target_df[target_column], config.target_normalizer
        )

        if source_series.empty or target_series.empty:
            return RelationAnalysis(
                source_table=config.source_table,
                source_table_alias=source_alias,
                source_column=config.source_column,
                source_column_resolved=source_column,
                target_table=config.target_table,
                target_table_alias=target_alias,
                target_column=config.target_column,
                target_column_resolved=target_column,
                relationship_type=config.relationship,
                description=config.description,
                status="insufficient_values",
                confidence=0.0,
            )

        source_unique = set(source_series.unique().tolist())
        target_unique = set(target_series.unique().tolist())
        if not source_unique or not target_unique:
            return RelationAnalysis(
                source_table=config.source_table,
                source_table_alias=source_alias,
                source_column=config.source_column,
                source_column_resolved=source_column,
                target_table=config.target_table,
                target_table_alias=target_alias,
                target_column=config.target_column,
                target_column_resolved=target_column,
                relationship_type=config.relationship,
                description=config.description,
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
            source_table=config.source_table,
            source_table_alias=source_alias,
            source_column=config.source_column,
            source_column_resolved=source_column,
            target_table=config.target_table,
            target_table_alias=target_alias,
            target_column=config.target_column,
            target_column_resolved=target_column,
            relationship_type=config.relationship,
            description=config.description,
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
        alias_index: TableAliasIndex,
    ) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        for canonical, aliases in alias_index.canonical_to_aliases.items():
            alias_key = alias_index.resolve_alias(canonical, tables)
            df = tables.get(alias_key, pd.DataFrame()) if alias_key else pd.DataFrame()
            metadata = {
                "type": "table",
                "row_count": int(df.shape[0]) if not df.empty else 0,
                "columns": [str(col) for col in df.columns] if not df.empty else [],
            }
            if alias_key and alias_key != canonical:
                metadata["alias"] = alias_key
            nodes[canonical] = metadata

        for alias, df in tables.items():
            canonical = alias_index.canonical_name(alias)
            if canonical in nodes and nodes[canonical].get("alias") == alias:
                continue
            if canonical in nodes and canonical != alias:
                continue
            nodes.setdefault(
                canonical,
                {
                    "type": "table",
                    "row_count": int(df.shape[0]),
                    "columns": [str(col) for col in df.columns],
                },
            )

        edges = [relation.as_dict() for relation in relations]

        paths: List[Dict[str, Tuple[str, ...]]] = []
        for path in PROCUREMENT_FLOW_PATHS:
            resolved: List[str] = []
            include = True
            for canonical in path:
                alias_key = alias_index.resolve_alias(canonical, tables)
                df = tables.get(alias_key, pd.DataFrame()) if alias_key else pd.DataFrame()
                if alias_key is None or df.empty:
                    include = False
                    break
                resolved.append(alias_key)
            if include:
                paths.append({"canonical": path, "resolved": tuple(resolved)})

        return {"nodes": nodes, "edges": edges, "paths": paths}

    def _build_alias_index(
        self,
        tables: Mapping[str, pd.DataFrame],
        table_name_map: Optional[Mapping[str, str]] = None,
    ) -> TableAliasIndex:
        canonical_to_aliases: Dict[str, set[str]] = {
            canonical: set(aliases) | {canonical}
            for canonical, aliases in PROCUREMENT_TABLE_ALIASES.items()
        }
        alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in canonical_to_aliases.items():
            for alias in aliases:
                alias_to_canonical[alias] = canonical

        for config in PROCUREMENT_RELATIONSHIPS:
            canonical_to_aliases.setdefault(config.source_table, set()).add(config.source_table)
            canonical_to_aliases.setdefault(config.target_table, set()).add(config.target_table)
            for alias in config.source_aliases:
                canonical_to_aliases[config.source_table].add(alias)
                alias_to_canonical[alias] = config.source_table
            for alias in config.target_aliases:
                canonical_to_aliases[config.target_table].add(alias)
                alias_to_canonical[alias] = config.target_table

        if table_name_map:
            for alias, canonical in table_name_map.items():
                canonical_to_aliases.setdefault(canonical, set()).add(alias)
                alias_to_canonical[alias] = canonical

        for canonical in list(canonical_to_aliases.keys()):
            canonical_to_aliases[canonical].add(canonical)
            alias_to_canonical.setdefault(canonical, canonical)

        for alias in tables.keys():
            canonical = alias_to_canonical.get(alias)
            if canonical:
                canonical_to_aliases.setdefault(canonical, set()).add(alias)
            else:
                alias_to_canonical[alias] = alias
                canonical_to_aliases.setdefault(alias, set()).add(alias)

        canonical_to_aliases_sorted = {
            canonical: tuple(sorted(aliases))
            for canonical, aliases in canonical_to_aliases.items()
        }
        return TableAliasIndex(
            canonical_to_aliases=canonical_to_aliases_sorted,
            alias_to_canonical=alias_to_canonical,
        )


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
            for field in (
                "document_type",
                "source_table",
                "target_table",
                "relationship_type",
            ):

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
        source_alias = relation.get("source_table_alias")
        target_alias = relation.get("target_table_alias")

        source_column = relation.get("source_column_resolved") or relation.get("source_column")
        target_column = relation.get("target_column_resolved") or relation.get("target_column")
        relation_type = relation.get("relationship_type", "related_to")
        description = relation.get("description")
        status = relation.get("status")
        confidence = relation.get("confidence")
        samples = relation.get("sample_values") or []

        def _table_label(canonical: Optional[str], alias: Optional[str]) -> str:
            if canonical and alias and canonical != alias:
                return f"{canonical} [{alias}]"
            return canonical or alias or "unknown"

        source_label = _table_label(source, source_alias)
        target_label = _table_label(target, target_alias)
        parts = [
            f"{source_label}.{source_column} {relation_type} {target_label}.{target_column}",
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

    def _path_to_text(
        self,
        canonical_path: Sequence[str],
        resolved_path: Optional[Sequence[str]] = None,
    ) -> str:
        canonical_sequence = " -> ".join(canonical_path)
        if resolved_path:
            resolved_sequence = " -> ".join(resolved_path)
            if tuple(canonical_path) != tuple(resolved_path):
                return (
                    f"Procurement data flows through {canonical_sequence} "
                    f"(resolved via {resolved_sequence})."
                )
        return f"Procurement data flows through {canonical_sequence}."

