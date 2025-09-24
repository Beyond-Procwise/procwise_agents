"""Data flow management and knowledge graph persistence for ProcWise agents."""
from __future__ import annotations

import hashlib
import logging
import math
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

_SUPPLIER_ID_ALIASES: Tuple[str, ...] = (
    "supplier",
    "supplier_identifier",
    "supplier_code",
    "supplier_reference",
    "vendor",
    "vendor_id",
    "vendor_identifier",
)

_SUPPLIER_NAME_ALIASES: Tuple[str, ...] = (
    "suppliername",
    "vendor_name",
    "vendorname",
    "trading_name",
)

_PO_ID_ALIASES: Tuple[str, ...] = (
    "purchase_order_id",
    "purchaseorderid",
    "po_number",
    "purchase_order_number",
)

_INVOICE_ID_ALIASES: Tuple[str, ...] = (
    "invoice_number",
    "invoice_reference",
    "invoiceid",
)

_QUOTE_ID_ALIASES: Tuple[str, ...] = (
    "quote_number",
    "quote_reference",
    "quoteid",
)

_CONTRACT_ID_ALIASES: Tuple[str, ...] = (
    "contract_number",
    "contract_reference",
    "contractid",
)

_ITEM_ID_ALIASES: Tuple[str, ...] = (
    "item",
    "itemcode",
    "product_id",
    "productid",
    "sku",
)

_ITEM_DESC_ALIASES: Tuple[str, ...] = (
    "item_name",
    "itemdesc",
    "product_name",
    "product_description",
    "service_description",
)

_PURCHASE_VALUE_COLUMNS: Tuple[str, ...] = (
    "line_amount_gbp",
    "line_total_gbp",
    "total_amount_gbp",
    "line_total",
    "total_amount",
)

_INVOICE_VALUE_COLUMNS: Tuple[str, ...] = (
    "total_amount_incl_tax_gbp",
    "invoice_total_incl_tax",
    "invoice_amount_gbp",
    "invoice_amount",
    "total_amount_incl_tax",
)

_PO_TOTAL_COLUMNS: Tuple[str, ...] = (
    "total_amount_gbp",
    "total_amount",
    "converted_amount_usd",
)

_QUOTE_TOTAL_COLUMNS: Tuple[str, ...] = (
    "total_amount_gbp",
    "total_amount",
    "quote_total",
)

_INVOICE_TOTAL_COLUMNS: Tuple[str, ...] = (
    "invoice_total_incl_tax",
    "invoice_amount",
    "total_amount_incl_tax",
)

_SUPPLIER_FLOW_ID_LIMIT = 5
_SUPPLIER_FLOW_ITEM_LIMIT = 5
_SUPPLIER_SAMPLE_LIMIT = 200


class DataFlowManager:
    """Constructs data flow mappings and persists them as a knowledge graph."""

    def __init__(self, agent_nick: Any, collection_name: Optional[str] = None) -> None:
        self.agent_nick = agent_nick
        settings = getattr(agent_nick, "settings", None)
        if collection_name is None:
            collection_name = getattr(settings, "qdrant_collection_name", None) or "Knowledge Graph"
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
        supplier_flows = self._extract_supplier_flows(tables, alias_index)
        graph["supplier_flows"] = supplier_flows
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

            for flow in graph.get("supplier_flows", []):
                if not isinstance(flow, Mapping):
                    continue
                text = self._supplier_flow_to_text(flow)
                vector = self._embed_text(text)
                if vector is None:
                    continue
                payload: Dict[str, Any] = {
                    "document_type": "supplier_flow",
                    "supplier_id": flow.get("supplier_id"),
                    "supplier_name": flow.get("supplier_name"),
                    "coverage_ratio": flow.get("coverage_ratio"),
                }
                for key in ("contracts", "purchase_orders", "invoices", "quotes", "products"):
                    value = flow.get(key)
                    if isinstance(value, dict):
                        payload[key] = value
                point_id = self._deterministic_id(text)
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

    # ------------------------------------------------------------------
    # Supplier flow helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
        else:
            text = str(value).strip().lower()
        return text or None

    @staticmethod
    def _clean_text_series(series: Optional[pd.Series]) -> pd.Series:
        if series is None:
            return pd.Series(dtype="string")
        cleaned = series.astype("string").str.strip()
        cleaned = cleaned.replace({"": pd.NA, "None": pd.NA, "null": pd.NA})
        cleaned = cleaned.replace({"nan": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
        return cleaned

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric) or math.isinf(numeric):
            return 0.0
        return numeric

    def _sum_numeric(self, series: Optional[pd.Series]) -> float:
        if series is None or series.empty:
            return 0.0
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        total = float(numeric.sum())
        if math.isnan(total) or math.isinf(total):
            return 0.0
        return total

    def _collect_ids(self, series: Optional[pd.Series], limit: int = _SUPPLIER_FLOW_ID_LIMIT) -> List[str]:
        if series is None or series.empty:
            return []
        cleaned = self._clean_text_series(series).dropna()
        results: List[str] = []
        for value in cleaned:
            text = str(value).strip()
            if not text or text in results:
                continue
            results.append(text)
            if len(results) >= limit:
                break
        return results

    def _format_timestamp(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.to_pydatetime().isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            parsed = None
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.to_pydatetime().isoformat()
        text = str(value).strip()
        return text or None

    def _select_first_existing(self, df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _derive_supplier_series(
        self, df: pd.DataFrame, name_map: Mapping[str, str]
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype="string")
        id_col = self._resolve_column(df, "supplier_id", _SUPPLIER_ID_ALIASES)
        if id_col:
            series = self._clean_text_series(df[id_col])
            return series
        name_col = self._resolve_column(df, "supplier_name", _SUPPLIER_NAME_ALIASES)
        if not name_col:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
        names = self._clean_text_series(df[name_col])
        keys = names.map(self._normalise_key)
        mapped = keys.map(lambda key: name_map.get(key) if key else None)
        resolved = pd.Series(mapped, index=df.index, dtype="object")
        fallback = names
        resolved = resolved.where(resolved.notna(), fallback)
        resolved = pd.Series(resolved, index=df.index, dtype="string").str.strip()
        resolved = resolved.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
        return resolved

    def _extract_supplier_flows(
        self, tables: Dict[str, pd.DataFrame], alias_index: TableAliasIndex
    ) -> List[Dict[str, Any]]:
        supplier_alias = alias_index.resolve_alias("proc.supplier", tables)
        supplier_df = tables.get(supplier_alias, pd.DataFrame()) if supplier_alias else pd.DataFrame()
        if supplier_df.empty:
            return []

        id_col = self._resolve_column(supplier_df, "supplier_id", _SUPPLIER_ID_ALIASES)
        if id_col is None:
            return []

        base_df = supplier_df.copy()
        id_series = self._clean_text_series(base_df[id_col])
        base_df = base_df.assign(__supplier_id=id_series)
        base_df = base_df[base_df["__supplier_id"].notna()].copy()
        if base_df.empty:
            return []

        if base_df.shape[0] > _SUPPLIER_SAMPLE_LIMIT:
            base_df = base_df.head(_SUPPLIER_SAMPLE_LIMIT).copy()

        name_col = self._resolve_column(base_df, "supplier_name", _SUPPLIER_NAME_ALIASES)
        name_series = self._clean_text_series(base_df[name_col]) if name_col else pd.Series(dtype="string")

        flows: Dict[str, Dict[str, Any]] = {}
        name_map: Dict[str, str] = {}
        for idx, supplier_id in base_df["__supplier_id"].items():
            if supplier_id is None:
                continue
            sid = str(supplier_id).strip()
            if not sid:
                continue
            entry = flows.setdefault(sid, {"supplier_id": sid})
            supplier_name = None
            if name_col and idx in name_series.index:
                raw_name = name_series.loc[idx]
                if pd.notna(raw_name):
                    supplier_name = str(raw_name).strip()
            if supplier_name:
                entry.setdefault("supplier_name", supplier_name)
                key = self._normalise_key(supplier_name)
                if key:
                    name_map.setdefault(key, sid)

        supplier_lookup = {
            sid: data.get("supplier_name") for sid, data in flows.items() if isinstance(data, dict)
        }

        def _ensure_entry(supplier_id: Any) -> Dict[str, Any]:
            sid = str(supplier_id).strip()
            if not sid:
                sid = str(supplier_id)
            entry = flows.setdefault(sid, {"supplier_id": sid})
            if not entry.get("supplier_name") and sid in supplier_lookup and supplier_lookup[sid]:
                entry["supplier_name"] = supplier_lookup[sid]
            return entry

        # Contracts
        contract_alias = alias_index.resolve_alias("proc.contracts", tables)
        contracts = tables.get(contract_alias, pd.DataFrame()) if contract_alias else pd.DataFrame()
        if not contracts.empty:
            supplier_series = self._derive_supplier_series(contracts, name_map)
            contract_df = contracts.assign(__supplier_id=supplier_series)
            contract_df = contract_df.dropna(subset=["__supplier_id"])
            if not contract_df.empty:
                contract_id_col = self._resolve_column(contract_df, "contract_id", _CONTRACT_ID_ALIASES) or "contract_id"
                end_date_col = self._resolve_column(
                    contract_df, "contract_end_date", ("contract_end_date", "end_date", "expiry_date")
                )
                for supplier_id, group in contract_df.groupby("__supplier_id"):
                    entry = _ensure_entry(supplier_id)
                    contract_ids = (
                        self._collect_ids(group[contract_id_col]) if contract_id_col in group.columns else []
                    )
                    latest_end: Optional[str] = None
                    if end_date_col and end_date_col in group.columns:
                        dates = pd.to_datetime(group[end_date_col], errors="coerce")
                        dates = dates.dropna()
                        if not dates.empty:
                            latest_end = self._format_timestamp(dates.max())
                    entry["contracts"] = {
                        "count": int(group.shape[0]),
                        "contract_ids": contract_ids,
                    }
                    if latest_end:
                        entry["contracts"]["latest_end_date"] = latest_end

        # Purchase orders
        po_alias = alias_index.resolve_alias("proc.purchase_order_agent", tables)
        purchase_orders = tables.get(po_alias, pd.DataFrame()) if po_alias else pd.DataFrame()
        po_supplier_series = None
        if not purchase_orders.empty:
            po_supplier_series = self._derive_supplier_series(purchase_orders, name_map)
            po_df = purchase_orders.assign(__supplier_id=po_supplier_series)
            po_df = po_df.dropna(subset=["__supplier_id"])
            if not po_df.empty:
                po_id_col = self._resolve_column(po_df, "po_id", _PO_ID_ALIASES) or "po_id"
                value_col = self._select_first_existing(po_df, _PO_TOTAL_COLUMNS)
                order_date_col = self._resolve_column(
                    po_df, "order_date", ("order_date", "requested_date", "created_date")
                )
                po_summary: Dict[str, Dict[str, Any]] = {}
                for supplier_id, group in po_df.groupby("__supplier_id"):
                    entry = _ensure_entry(supplier_id)
                    total_value = self._sum_numeric(group[value_col]) if value_col and value_col in group.columns else 0.0
                    po_ids = self._collect_ids(group[po_id_col]) if po_id_col in group.columns else []
                    latest_order = None
                    if order_date_col and order_date_col in group.columns:
                        dates = pd.to_datetime(group[order_date_col], errors="coerce").dropna()
                        if not dates.empty:
                            latest_order = self._format_timestamp(dates.max())
                    po_summary[supplier_id] = {
                        "count": int(group.shape[0]),
                        "po_ids": po_ids,
                        "total_value_gbp": round(total_value, 2),
                    }
                    if latest_order:
                        po_summary[supplier_id]["latest_order_date"] = latest_order
                    entry["purchase_orders"] = po_summary[supplier_id]

        # Purchase order line items (product coverage)
        po_lines_alias = alias_index.resolve_alias("proc.po_line_items_agent", tables)
        po_lines = tables.get(po_lines_alias, pd.DataFrame()) if po_lines_alias else pd.DataFrame()
        if po_supplier_series is not None and not po_lines.empty and not purchase_orders.empty:
            po_id_col = self._resolve_column(purchase_orders, "po_id", _PO_ID_ALIASES) or "po_id"
            line_po_id_col = self._resolve_column(po_lines, "po_id", _PO_ID_ALIASES) or "po_id"
            po_lookup = purchase_orders[[po_id_col]].copy()
            po_lookup["__supplier_id"] = po_supplier_series
            join_df = po_lines.merge(po_lookup, left_on=line_po_id_col, right_on=po_id_col, how="left")
            join_df = join_df.dropna(subset=["__supplier_id"])
            if not join_df.empty:
                value_col = self._select_first_existing(join_df, _PURCHASE_VALUE_COLUMNS)
                if value_col:
                    join_df[value_col] = pd.to_numeric(join_df[value_col], errors="coerce").fillna(0.0)
                item_desc_col = self._resolve_column(join_df, "item_description", _ITEM_DESC_ALIASES)
                item_id_col = self._resolve_column(join_df, "item_id", _ITEM_ID_ALIASES)
                for supplier_id, group in join_df.groupby("__supplier_id"):
                    entry = _ensure_entry(supplier_id)
                    descriptions: Dict[str, float] = {}
                    if item_desc_col and item_desc_col in group.columns:
                        desc_series = self._clean_text_series(group[item_desc_col]).dropna()
                        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0) if value_col in group.columns else pd.Series(0.0, index=group.index)
                        for idx, desc in desc_series.items():
                            spend = self._safe_float(values.loc[idx]) if idx in values.index else 0.0
                            descriptions[desc] = descriptions.get(desc, 0.0) + spend
                    elif item_id_col and item_id_col in group.columns:
                        id_series = self._clean_text_series(group[item_id_col]).dropna()
                        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0) if value_col in group.columns else pd.Series(0.0, index=group.index)
                        for idx, desc in id_series.items():
                            spend = self._safe_float(values.loc[idx]) if idx in values.index else 0.0
                            descriptions[desc] = descriptions.get(desc, 0.0) + spend
                    if descriptions:
                        top_items = sorted(descriptions.items(), key=lambda item: item[1], reverse=True)
                        entry["products"] = {
                            "top_items": [
                                {"description": desc, "spend_gbp": round(self._safe_float(value), 2)}
                                for desc, value in top_items[: _SUPPLIER_FLOW_ITEM_LIMIT]
                            ],
                            "unique_items": len(descriptions),
                        }

        # Invoices
        invoice_alias = alias_index.resolve_alias("proc.invoice_agent", tables)
        invoices = tables.get(invoice_alias, pd.DataFrame()) if invoice_alias else pd.DataFrame()
        invoice_supplier_series = None
        if not invoices.empty:
            invoice_supplier_series = self._derive_supplier_series(invoices, name_map)
            inv_df = invoices.assign(__supplier_id=invoice_supplier_series)
            inv_df = inv_df.dropna(subset=["__supplier_id"])
            if not inv_df.empty:
                invoice_id_col = self._resolve_column(inv_df, "invoice_id", _INVOICE_ID_ALIASES) or "invoice_id"
                total_col = self._select_first_existing(inv_df, _INVOICE_TOTAL_COLUMNS)
                invoice_date_col = self._resolve_column(inv_df, "invoice_date", ("invoice_date", "created_date"))
                for supplier_id, group in inv_df.groupby("__supplier_id"):
                    entry = _ensure_entry(supplier_id)
                    total_value = self._sum_numeric(group[total_col]) if total_col and total_col in group.columns else 0.0
                    invoice_ids = self._collect_ids(group[invoice_id_col]) if invoice_id_col in group.columns else []
                    latest_invoice = None
                    if invoice_date_col and invoice_date_col in group.columns:
                        dates = pd.to_datetime(group[invoice_date_col], errors="coerce").dropna()
                        if not dates.empty:
                            latest_invoice = self._format_timestamp(dates.max())
                    entry["invoices"] = {
                        "count": int(group.shape[0]),
                        "invoice_ids": invoice_ids,
                        "total_value_gbp": round(total_value, 2),
                    }
                    if latest_invoice:
                        entry["invoices"]["latest_invoice_date"] = latest_invoice

        # Quotes
        quote_alias = alias_index.resolve_alias("proc.quote_agent", tables)
        quotes = tables.get(quote_alias, pd.DataFrame()) if quote_alias else pd.DataFrame()
        if not quotes.empty:
            quote_supplier_series = self._derive_supplier_series(quotes, name_map)
            quote_df = quotes.assign(__supplier_id=quote_supplier_series)
            quote_df = quote_df.dropna(subset=["__supplier_id"])
            if not quote_df.empty:
                quote_id_col = self._resolve_column(quote_df, "quote_id", _QUOTE_ID_ALIASES) or "quote_id"
                total_col = self._select_first_existing(quote_df, _QUOTE_TOTAL_COLUMNS)
                for supplier_id, group in quote_df.groupby("__supplier_id"):
                    entry = _ensure_entry(supplier_id)
                    total_value = self._sum_numeric(group[total_col]) if total_col and total_col in group.columns else 0.0
                    quote_ids = self._collect_ids(group[quote_id_col]) if quote_id_col in group.columns else []
                    entry["quotes"] = {
                        "count": int(group.shape[0]),
                        "quote_ids": quote_ids,
                        "total_value_gbp": round(total_value, 2),
                    }

        flows_list: List[Dict[str, Any]] = []
        for supplier_id, payload in flows.items():
            entry = dict(payload)
            coverage_components = 0
            coverage_hits = 0
            for key in ("contracts", "purchase_orders", "invoices", "quotes"):
                details = entry.get(key)
                if isinstance(details, dict):
                    coverage_components += 1
                    if details.get("count"):
                        coverage_hits += 1
            coverage_ratio = (coverage_hits / coverage_components) if coverage_components else 0.0
            entry["coverage_ratio"] = round(coverage_ratio, 3)
            flows_list.append(entry)

        flows_list.sort(key=lambda item: (item.get("supplier_name") or "", item["supplier_id"]))
        return flows_list

    def _supplier_flow_to_text(self, flow: Mapping[str, Any]) -> str:
        supplier_id = flow.get("supplier_id")
        supplier_name = flow.get("supplier_name")
        label = str(supplier_name).strip() if supplier_name else None
        if label:
            label = f"{label} ({supplier_id})"
        else:
            label = f"Supplier {supplier_id}"

        parts = [f"{label} data flow summary."]

        coverage = flow.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            parts.append(f"Coverage ratio across contracts, purchase orders, quotes and invoices is {coverage:.2f}.")

        contracts = flow.get("contracts") if isinstance(flow.get("contracts"), dict) else {}
        if contracts:
            contract_count = contracts.get("count", 0)
            if contract_count:
                detail = f"{int(contract_count)} contract{'s' if contract_count != 1 else ''}"
                contract_ids = contracts.get("contract_ids")
                if isinstance(contract_ids, list) and contract_ids:
                    detail += f" ({', '.join(contract_ids[:_SUPPLIER_FLOW_ID_LIMIT])})"
                if contracts.get("latest_end_date"):
                    detail += f" ending by {contracts['latest_end_date']}"
                parts.append(detail + ".")

        purchase_orders = flow.get("purchase_orders") if isinstance(flow.get("purchase_orders"), dict) else {}
        if purchase_orders:
            po_count = purchase_orders.get("count", 0)
            if po_count:
                detail = f"{int(po_count)} purchase order{'s' if po_count != 1 else ''}"
                total_value = purchase_orders.get("total_value_gbp")
                if isinstance(total_value, (int, float)) and total_value:
                    detail += f" totalling {total_value:,.2f} GBP"
                if purchase_orders.get("latest_order_date"):
                    detail += f" (latest order {purchase_orders['latest_order_date']})"
                parts.append(detail + ".")

        invoices = flow.get("invoices") if isinstance(flow.get("invoices"), dict) else {}
        if invoices:
            invoice_count = invoices.get("count", 0)
            if invoice_count:
                detail = f"{int(invoice_count)} invoice{'s' if invoice_count != 1 else ''}"
                total_value = invoices.get("total_value_gbp")
                if isinstance(total_value, (int, float)) and total_value:
                    detail += f" worth {total_value:,.2f} GBP"
                if invoices.get("latest_invoice_date"):
                    detail += f" (latest invoice {invoices['latest_invoice_date']})"
                parts.append(detail + ".")

        quotes = flow.get("quotes") if isinstance(flow.get("quotes"), dict) else {}
        if quotes:
            quote_count = quotes.get("count", 0)
            if quote_count:
                detail = f"{int(quote_count)} quote{'s' if quote_count != 1 else ''}"
                total_value = quotes.get("total_value_gbp")
                if isinstance(total_value, (int, float)) and total_value:
                    detail += f" evaluated at {total_value:,.2f} GBP"
                parts.append(detail + ".")

        products = flow.get("products") if isinstance(flow.get("products"), dict) else {}
        if products:
            top_items = products.get("top_items")
            if isinstance(top_items, list) and top_items:
                descriptions = []
                for item in top_items[:_SUPPLIER_FLOW_ITEM_LIMIT]:
                    if not isinstance(item, dict):
                        continue
                    desc = str(item.get("description") or "").strip()
                    spend = item.get("spend_gbp")
                    if desc and isinstance(spend, (int, float)):
                        descriptions.append(f"{desc} ({spend:,.2f} GBP)")
                    elif desc:
                        descriptions.append(desc)
                if descriptions:
                    parts.append(
                        "Key products include " + ", ".join(descriptions) + "."
                    )
            unique = products.get("unique_items")
            if isinstance(unique, (int, float)) and unique:
                parts.append(f"Product catalogue covers {int(unique)} unique items.")

        return " ".join(parts)

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

