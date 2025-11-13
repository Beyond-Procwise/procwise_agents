"""Hybrid query engine combining Neo4j, Qdrant, and Ollama."""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
import requests
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline

LOGGER = logging.getLogger("hybrid_query_engine")
logging.basicConfig(
    level=os.environ.get("PROCWISE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@dataclass
class RetrievalResult:
    source: str
    title: str
    snippet: str
    weight: float
    payload: Dict[str, Any]
    collection: Optional[str] = None
    document_id: Optional[str] = None
    rerank_score: Optional[float] = None


class OllamaChatClient:
    """Simple HTTP client for Ollama chat and completion APIs."""

    def __init__(self, host: str, model: str = "phi4:latest", temperature: float = 0.35) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature

    def chat(self, prompt: str, system: Optional[str] = None, format: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "format": format,
                "temperature": self._temperature,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content") or data.get("response", "")

    def embed(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        response = requests.post(
            f"{self._host}/api/embeddings",
            json={"model": model, "input": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding") or data.get("data", [{}])[0].get("embedding")


ALLOWED_SPEND_DIMENSIONS = {
    "category": "category_id",
    "supplier": "supplier_id",
    "product": "product_id",
    "department": "department_id",
    "business_unit": "business_unit_id",
}


class Neo4jQueryService:
    """Wraps Neo4j access with curated queries."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def supplier_performance(self, supplier_id: str) -> Dict[str, Any]:
        query = """
        MATCH (s:Supplier {supplier_id: $supplier_id})
        OPTIONAL MATCH (s)<-[:ISSUED_TO]-(po:PurchaseOrder)
        OPTIONAL MATCH (s)<-[:PROVIDED_BY]-(q:Quote)
        OPTIONAL MATCH (s)<-[:FROM_SUPPLIER]-(resp:SupplierResponse)
        RETURN s.name AS name,
               s.risk_rating AS risk_rating,
               s.performance_score AS performance_score,
               avg(po.total_amount) AS avg_po_value,
               count(DISTINCT po) AS po_count,
               count(DISTINCT q) AS quotes,
               avg(resp.proposed_amount) AS avg_response_amount
        """
        with self._driver.session() as session:
            record = session.run(query, supplier_id=supplier_id).single()
        if not record:
            raise ValueError(f"Supplier {supplier_id} not found")
        return {k: record[k] for k in record.keys()}

    def supplier_response_counts(self, workflow_id: str) -> Dict[str, int]:
        query = """
        MATCH (n:NegotiationSession {workflow_id: $workflow_id})<-[:RESPONDS_TO]-(resp:SupplierResponse)
        RETURN count(DISTINCT resp) AS responses
        """
        with self._driver.session() as session:
            record = session.run(query, workflow_id=workflow_id).single()
        return {"responses": record["responses"] if record else 0}

    def spend_by_dimension(self, dimension: str, value: Optional[str] = None, period: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        dimension_key = dimension.lower().strip()
        if dimension_key not in ALLOWED_SPEND_DIMENSIONS:
            raise ValueError(
                f"Unsupported spend dimension '{dimension}'. Allowed values: {', '.join(sorted(ALLOWED_SPEND_DIMENSIONS))}"
            )
        property_name = ALLOWED_SPEND_DIMENSIONS[dimension_key]
        filters = []
        params: Dict[str, Any] = {}
        if value:
            filters.append(f"po.{property_name} = $value")
            params["value"] = value
        if period:
            filters.append("po.issued_at >= $start AND po.issued_at <= $end")
            params.update({"start": period[0], "end": period[1]})
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        query = f"""
        MATCH (po:PurchaseOrder)-[:ISSUED_TO]->(s:Supplier)
        OPTIONAL MATCH (po)-[:ORDERED_UNDER]->(c:Category)
        OPTIONAL MATCH (po)-[:INCLUDES]->(p:Product)
        {where_clause}
        WITH po, s, c, p
        RETURN coalesce(c.name, 'Uncategorized') AS category,
               coalesce(s.name, 'Unknown Supplier') AS supplier,
               coalesce(p.name, 'Mixed Products') AS product,
               sum(po.total_amount) AS total_spend,
               count(DISTINCT po) AS order_count
        ORDER BY total_spend DESC
        LIMIT 50
        """
        with self._driver.session() as session:
            result = session.run(query, **params)
        return [dict(record) for record in result]

    def negotiation_summary(self, rfq_reference: Optional[str], workflow_id: Optional[str]) -> List[Dict[str, Any]]:
        filters = []
        params: Dict[str, Any] = {}
        if rfq_reference:
            filters.append("n.rfq_reference = $rfq_reference")
            params["rfq_reference"] = rfq_reference
        if workflow_id:
            filters.append("n.workflow_id = $workflow_id")
            params["workflow_id"] = workflow_id
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        query = f"""
        MATCH (n:NegotiationSession)
        OPTIONAL MATCH (n)<-[:RESPONDS_TO]-(resp:SupplierResponse)
        OPTIONAL MATCH (resp)-[:FROM_SUPPLIER]->(s:Supplier)
        {where_clause}
        WITH n, s, resp
        RETURN n.negotiation_session_id AS session_id,
               n.workflow_id AS workflow_id,
               n.rfq_reference AS rfq_reference,
               n.round_number AS round,
               n.status AS status,
               collect({supplier: s.name, amount: resp.proposed_amount, received_at: resp.received_at}) AS responses
        ORDER BY n.started_at DESC
        LIMIT 25
        """
        with self._driver.session() as session:
            result = session.run(query, **params)
        return [dict(record) for record in result]

    def run_cypher(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, **params)
        return [dict(record) for record in result]


class QdrantRetriever:
    """Hybrid dense+lexical retriever with collection routing and reranking."""

    def __init__(
        self,
        client: QdrantClient,
        collections: Sequence[Dict[str, str]],
        embed_model: str,
        rerank_model: str,
        router_model: str,
        route_top_k: int = 3,
        per_collection_k: int = 15,
    ) -> None:
        if not collections:
            raise ValueError("QdrantRetriever requires at least one collection configuration")
        self._client = client
        self._collections = list(collections)
        self._route_top_k = max(1, route_top_k)
        self._per_collection_k = max(1, per_collection_k)
        self._label_to_name = {cfg.get("label", cfg["name"]): cfg["name"] for cfg in self._collections}
        self._name_to_label = {cfg["name"]: cfg.get("label", cfg["name"]) for cfg in self._collections}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = SentenceTransformer(embed_model, device=device)
        self._reranker = CrossEncoder(rerank_model, device=device)
        router_device = 0 if device == "cuda" else -1
        self._router = pipeline(
            "zero-shot-classification",
            model=router_model,
            device=router_device,
        )

    def search(self, text: str, filters: Optional[Dict[str, Any]] = None, limit: int = 12) -> List[RetrievalResult]:
        if not text.strip():
            return []
        filters = dict(filters) if filters else {}
        forced_collections = filters.pop("collection", None)
        selected_collections = self._route_collections(text, forced_collections)
        query_vector = self._encode_query(text)
        per_collection_results: List[List[Dict[str, Any]]] = []
        for collection in selected_collections:
            per_collection_results.append(
                self._search_collection(collection, query_vector, filters, limit=self._per_collection_k)
            )
        fused_candidates = self._reciprocal_rank_fusion(per_collection_results)
        if fused_candidates:
            reranked = self._cross_encoder_rerank(text, fused_candidates, top_k=limit)
            results = [self._to_retrieval_result(doc) for doc in reranked]
            if results:
                return results
        LOGGER.info("Falling back to sparse retrieval for query '%s'", text)
        return self._bm25_fallback(text, selected_collections, filters, limit)

    def _encode_query(self, query: str) -> np.ndarray:
        return self._embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]

    def _route_collections(
        self, query: str, forced: Optional[Any] = None
    ) -> List[str]:
        if forced:
            if isinstance(forced, str):
                return [forced]
            if isinstance(forced, Sequence):
                return [str(item) for item in forced] or [cfg["name"] for cfg in self._collections]
        labels = list(self._label_to_name.keys())
        if not labels:
            return [cfg["name"] for cfg in self._collections]
        try:
            prediction = self._router(query, candidate_labels=labels, multi_label=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Router failed (%s); defaulting to all collections", exc)
            return [cfg["name"] for cfg in self._collections]
        pairs = list(zip(prediction["labels"], prediction["scores"]))
        pairs.sort(key=lambda item: item[1], reverse=True)
        thresholded = [label for label, score in pairs[: self._route_top_k] if score >= 0.15]
        if not thresholded:
            return [cfg["name"] for cfg in self._collections]
        return [self._label_to_name[label] for label in thresholded if label in self._label_to_name]

    def _search_collection(
        self,
        collection: str,
        query_vector: np.ndarray,
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        hits = self._client.search(
            collection_name=collection,
            query_vector=query_vector.astype(np.float32).tolist(),
            query_filter=qdrant_filter,
            with_payload=True,
            limit=limit,
            search_params=qmodels.SearchParams(hnsw_ef=128, exact=False),
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "collection": collection,
                    "id": payload.get("doc_id", hit.id),
                    "title": _safe_title(payload),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                    "payload": payload,
                }
            )
        return results

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[qmodels.Filter]:
        if not filters:
            return None
        conditions: List[qmodels.FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(value))))
            else:
                conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
        return qmodels.Filter(must=conditions) if conditions else None

    def _reciprocal_rank_fusion(self, ranked_lists: Sequence[List[Dict[str, Any]]], k: float = 60.0) -> List[Dict[str, Any]]:
        scores: Dict[Tuple[str, Any], float] = {}
        seen: Dict[Tuple[str, Any], Dict[str, Any]] = {}
        for lst in ranked_lists:
            for rank, item in enumerate(lst, start=1):
                key = (item["collection"], item["id"])
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                seen[key] = item
        fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [seen[key] for key, _ in fused]

    def _cross_encoder_rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        pairs = [[query, doc.get("text", "") or ""] for doc in docs]
        scores = self._reranker.predict(pairs).tolist()
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        docs.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return docs[:top_k]

    def _to_retrieval_result(self, doc: Dict[str, Any]) -> RetrievalResult:
        payload = doc.get("payload", {})
        snippet = doc.get("text") or payload.get("text", "")
        collection = doc.get("collection")
        label = self._name_to_label.get(collection, collection)
        weight = doc.get("rerank_score") if doc.get("rerank_score") is not None else doc.get("score", 0.0)
        document_id = doc.get("id")
        return RetrievalResult(
            source=label or "vector",
            title=doc.get("title") or _safe_title(payload),
            snippet=snippet,
            weight=weight,
            payload=payload,
            collection=collection,
            document_id=str(document_id) if document_id is not None else None,
            rerank_score=doc.get("rerank_score"),
        )

    def _bm25_fallback(
        self,
        text: str,
        collections: Sequence[str],
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[RetrievalResult]:
        payloads = self._snapshot_payloads(collections, filters)
        documents = [payload.get("text", "") for payload in payloads]
        if not documents:
            return []
        bm25 = BM25Okapi([doc.lower().split() for doc in documents])
        scores = bm25.get_scores(text.lower().split())
        ranked = sorted(zip(payloads, scores), key=lambda item: item[1], reverse=True)[:limit]
        results: List[RetrievalResult] = []
        for payload, score in ranked:
            collection = payload.get("collection")
            label = self._name_to_label.get(collection, collection)
            results.append(
                RetrievalResult(
                    source=label or "sparse",
                    title=_safe_title(payload),
                    snippet=payload.get("text", ""),
                    weight=float(score),
                    payload=payload,
                    collection=collection,
                    document_id=str(payload.get("doc_id")) if payload.get("doc_id") else None,
                )
            )
        return results

    def _snapshot_payloads(
        self, collections: Sequence[str], filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        snapshots: List[Dict[str, Any]] = []
        for collection in collections:
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=collection,
                    scroll_filter=qdrant_filter,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    payload.setdefault("collection", collection)
                    snapshots.append(payload)
                if offset is None:
                    break
        return snapshots


class HybridQueryEngine:
    """Combines graph, vectors, and LLM reasoning."""

    def __init__(
        self,
        graph_service: Neo4jQueryService,
        retriever: QdrantRetriever,
        ollama: OllamaChatClient,
    ) -> None:
        self._graph_service = graph_service
        self._retriever = retriever
        self._ollama = ollama

    def ask(self, text: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        LOGGER.info("Received hybrid query: %s", text)
        intent = self._interpret_intent(text)
        vector_results = self._retriever.search(text, filters=filters)
        graph_results = self._graph_lookup(intent)
        fused = self._rerank(text, vector_results, graph_results)
        answer_html = self._synthesise_answer(text, intent, fused)
        suggestions = self._follow_up_suggestions(intent)
        supporting_facts = [self._project_fact(result) for result in fused[:5]]
        return {
            "answer_html": answer_html,
            "supporting_facts": supporting_facts,
            "suggestions": suggestions,
        }

    def get_supplier_performance_metrics(self, supplier_id: str) -> Dict[str, Any]:
        metrics = self._graph_service.supplier_performance(supplier_id)
        return _mask_internal_ids(metrics)

    def find_similar_negotiations(self, rfq_id: Optional[str] = None, text: Optional[str] = None) -> List[Dict[str, Any]]:
        query_text = text or f"Negotiation similar to {rfq_id}"
        filters = {"label": "NegotiationSession"}
        if rfq_id:
            filters["rfq_reference"] = rfq_id
        results = self._retriever.search(query_text, filters=filters)
        ranked = [self._project_fact(res) for res in results[:10]]
        return ranked

    def _interpret_intent(self, text: str) -> Dict[str, Any]:
        prompt = (
            "Classify the procurement question and return JSON with keys 'type', "
            "'dimensions', and 'filters'. Types: supplier_intelligence, negotiation_analysis, spend_analysis, general. "
            "Include important numeric hints if present."
        )
        response = self._ollama.chat(
            prompt=f"Question: {text}\nReturn JSON only.",
            system=prompt,
            format="json",
        )
        try:
            intent = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse intent JSON, defaulting to general. Response: %s", response)
            intent = {"type": "general", "dimensions": [], "filters": {}}
        return intent

    def _graph_lookup(self, intent: Dict[str, Any]) -> List[RetrievalResult]:
        intent_type = intent.get("type", "general")
        if intent_type == "supplier_intelligence" and intent.get("filters", {}).get("supplier_id"):
            data = self._graph_service.supplier_performance(intent["filters"]["supplier_id"])
            return [RetrievalResult(source="graph", title=data.get("name", "Supplier"), snippet=json.dumps(data), weight=1.0, payload=data)]
        if intent_type == "negotiation_analysis":
            data = self._graph_service.negotiation_summary(
                rfq_reference=intent.get("filters", {}).get("rfq_reference"),
                workflow_id=intent.get("filters", {}).get("workflow_id"),
            )
            return [
                RetrievalResult(
                    source="graph",
                    title=f"Negotiation {item.get('rfq_reference') or item.get('session_id')}",
                    snippet=json.dumps(item),
                    weight=1.0,
                    payload=item,
                )
                for item in data
            ]
        if intent_type == "spend_analysis":
            dims = intent.get("dimensions") or ["category"]
            dimension = dims[0]
            period = None
            filters = intent.get("filters") or {}
            if "start" in filters and "end" in filters:
                period = (filters["start"], filters["end"])
            rows = self._graph_service.spend_by_dimension(dimension, value=filters.get(dimension + "_id"), period=period)
            return [
                RetrievalResult(
                    source="graph",
                    title=f"Spend {row.get('category')} / {row.get('supplier')}",
                    snippet=json.dumps(row),
                    weight=1.0,
                    payload=row,
                )
                for row in rows
            ]
        return []

    def _rerank(
        self,
        query: str,
        vector_results: Sequence[RetrievalResult],
        graph_results: Sequence[RetrievalResult],
    ) -> List[RetrievalResult]:
        combined = list(graph_results) + list(vector_results)
        if not combined:
            return []
        tokens = query.lower().split()
        for result in combined:
            text_tokens = result.snippet.lower().split()
            overlap = len(set(tokens) & set(text_tokens))
            result.weight = result.weight + (overlap / (len(tokens) + 1))
        combined.sort(key=lambda r: r.weight, reverse=True)
        return combined

    def _synthesise_answer(self, query: str, intent: Dict[str, Any], results: Sequence[RetrievalResult]) -> str:
        context_text = pack_context(results)
        system_prompt = (
            "You are a procurement intelligence assistant. Use the provided context to answer the question in HTML with headings, "
            "bullet lists, and tables when appropriate. Do not mention internal identifiers, collection names, or table names."
        )
        prompt = (
            f"Question: {query}\n\nContext:\n{context_text}\n\nRespond with:\n"
            "<section>\n  <h2>Answer</h2>...\n</section> structure."
        )
        response = self._ollama.chat(prompt=prompt, system=system_prompt)
        return response

    def _follow_up_suggestions(self, intent: Dict[str, Any]) -> List[str]:
        intent_type = intent.get("type", "general")
        if intent_type == "supplier_intelligence":
            return [
                "Ask for recent negotiation outcomes with this supplier",
                "Compare this supplier's lead time to category averages",
            ]
        if intent_type == "negotiation_analysis":
            return [
                "Request recommended counter-offers",
                "Identify suppliers needing follow-up reminders",
            ]
        if intent_type == "spend_analysis":
            return [
                "Drill into spend variance by month",
                "List savings opportunities for adjacent categories",
            ]
        return [
            "Ask about supplier coverage gaps",
            "Request negotiation readiness summary",
        ]

    def _project_fact(self, result: RetrievalResult) -> Dict[str, Any]:
        payload = _mask_internal_ids(result.payload)
        return {
            "title": result.title,
            "summary": result.snippet[:280],
            "source": result.source,
            "data": payload,
        }


def build_default_engine() -> HybridQueryEngine:
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_prefix = os.environ.get("QDRANT_COLLECTION_PREFIX", "")
    ollama_host = os.environ.get("OLLAMA_HOST")
    model = os.environ.get("LLM_DEFAULT_MODEL", "phi4:latest")
    if not all([neo4j_uri, neo4j_user, neo4j_password, qdrant_url, ollama_host]):
        raise RuntimeError("HybridQueryEngine missing configuration")
    graph_service = Neo4jQueryService(neo4j_uri, neo4j_user, neo4j_password)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    ollama = OllamaChatClient(ollama_host, model=model)
    collections_raw = os.environ.get("QDRANT_COLLECTIONS")
    collections: List[Dict[str, str]] = []
    if collections_raw:
        try:
            parsed = json.loads(collections_raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        name = _apply_prefix(collection_prefix, item["name"])
                        label = item.get("label") or item.get("title") or name
                        collections.append({"name": name, "label": label})
                    elif isinstance(item, str):
                        collections.append(_parse_collection_entry(collection_prefix, item))
        except json.JSONDecodeError:
            for entry in collections_raw.split(","):
                entry = entry.strip()
                if entry:
                    collections.append(_parse_collection_entry(collection_prefix, entry))
    if not collections:
        default_name = _apply_prefix(collection_prefix, "procwise_procurement_embeddings")
        collections = [{"name": default_name, "label": "Procurement"}]
    embed_model = os.environ.get("EMBED_MODEL", "intfloat/e5-large-v2")
    rerank_model = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    router_model = os.environ.get("ROUTER_MODEL", "facebook/bart-large-mnli")
    route_top_k = int(os.environ.get("RETRIEVAL_ROUTE_TOP_K", "3"))
    per_collection_k = int(os.environ.get("RETRIEVAL_PER_COLLECTION_K", "15"))
    retriever = QdrantRetriever(
        qdrant_client,
        collections=collections,
        embed_model=embed_model,
        rerank_model=rerank_model,
        router_model=router_model,
        route_top_k=route_top_k,
        per_collection_k=per_collection_k,
    )
    return HybridQueryEngine(graph_service, retriever, ollama)


def _apply_prefix(prefix: str, name: str) -> str:
    if not prefix:
        return name
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def _parse_collection_entry(prefix: str, entry: str) -> Dict[str, str]:
    if ":" in entry:
        name_part, label_part = entry.split(":", 1)
        name = _apply_prefix(prefix, name_part.strip())
        label = label_part.strip() or name
    else:
        name = _apply_prefix(prefix, entry.strip())
        label = name
    return {"name": name, "label": label}


def pack_context(results: Sequence[RetrievalResult], max_tokens: int = 3000) -> str:
    if not results:
        return ""

    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.2))

    def is_summary(result: RetrievalResult) -> bool:
        payload = result.payload or {}
        if payload.get("is_summary"):
            return True
        section = str(payload.get("section", "")).lower()
        return section in {"summary", "overview", "abstract"}

    tokens_left = max_tokens
    context_blocks: List[str] = []
    seen: Set[Tuple[Optional[str], Optional[str], str]] = set()

    def add_result(res: RetrievalResult) -> None:
        nonlocal tokens_left
        if tokens_left <= 0:
            return
        key = (res.collection, res.document_id, res.title)
        if key in seen:
            return
        seen.add(key)
        masked_payload = _mask_internal_ids(res.payload)
        doc_id = res.document_id or masked_payload.get("doc_id") or masked_payload.get("id") or "unknown"
        collection_label = res.collection or res.source or "document"
        snippet = (res.snippet or "")[:2000]
        block_parts = [f"[{collection_label}:{doc_id}] {res.title}"]
        if snippet:
            block_parts.append(f"Snippet: {snippet}")
        if masked_payload:
            block_parts.append(f"Metadata: {json.dumps(masked_payload, ensure_ascii=False)}")
        block_text = "\n".join(block_parts)
        context_blocks.append(block_text)
        tokens_left -= min(estimate_tokens(block_text), 300)

    for summary_result in results:
        if is_summary(summary_result):
            add_result(summary_result)
            if tokens_left <= 0:
                return "\n---\n".join(context_blocks)

    counter = Counter((res.collection or res.source or "document") for res in results[:30])
    total = sum(counter.values()) or 1
    for collection, freq in counter.most_common():
        quota = max(1, int((freq / total) * (max_tokens / 300)))
        added = 0
        for res in results:
            if (res.collection or res.source or "document") != collection:
                continue
            if tokens_left <= 0 or added >= quota:
                break
            add_result(res)
            added += 1
        if tokens_left <= 0:
            break

    return "\n---\n".join(context_blocks)


def _mask_internal_ids(payload: Dict[str, Any]) -> Dict[str, Any]:
    masked: Dict[str, Any] = {}
    for key, value in payload.items():
        if key.endswith("_id") and key not in {"supplier_id", "category_id", "product_id"}:
            continue
        masked[key] = value
    return masked


def _safe_title(payload: Dict[str, Any]) -> str:
    name = payload.get("public_name") or payload.get("title") or payload.get("entity_key")
    return str(name)
