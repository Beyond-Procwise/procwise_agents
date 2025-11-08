"""Hybrid query engine combining Neo4j, Qdrant, and Ollama."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
import requests

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
    """Helper that wraps Qdrant vector search and lexical fallback."""

    def __init__(self, client: QdrantClient, collection: str, ollama: OllamaChatClient) -> None:
        self._client = client
        self._collection = collection
        self._ollama = ollama

    def search(self, text: str, filters: Optional[Dict[str, Any]] = None, limit: int = 8) -> List[RetrievalResult]:
        qdrant_filter = None
        if filters:
            must = []
            for key, value in filters.items():
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
            qdrant_filter = qmodels.Filter(must=must)
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=self._ollama.embed(text),
            query_filter=qdrant_filter,
            with_payload=True,
            limit=limit,
        )
        results = [
            RetrievalResult(
                source="vector",
                title=_safe_title(hit.payload),
                snippet=hit.payload.get("text", ""),
                weight=1.0 - hit.score,
                payload=hit.payload,
            )
            for hit in hits
        ]
        if not results or all(r.weight < 0.2 for r in results):
            LOGGER.info("Falling back to sparse retrieval for query '%s'", text)
            results = self._bm25_fallback(text, filters)
        return results

    def _bm25_fallback(self, text: str, filters: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        payloads = self._snapshot_payloads(filters)
        documents = [p.get("text", "") for p in payloads]
        if not documents:
            return []
        bm25 = BM25Okapi([doc.lower().split() for doc in documents])
        scores = bm25.get_scores(text.lower().split())
        ranked = sorted(zip(payloads, scores), key=lambda item: item[1], reverse=True)[:8]
        return [
            RetrievalResult(
                source="sparse",
                title=_safe_title(payload),
                snippet=payload.get("text", ""),
                weight=score,
                payload=payload,
            )
            for payload, score in ranked
        ]

    def _snapshot_payloads(self, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scroll_filter = None
        if filters:
            must = [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in filters.items()]
            scroll_filter = qmodels.Filter(must=must)
        results: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            results.extend([p.payload for p in points if p.payload])
            if offset is None:
                break
        return results


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
        context_parts = []
        for result in results[:6]:
            masked_payload = _mask_internal_ids(result.payload)
            context_parts.append(f"Source: {result.source}\nTitle: {result.title}\nData: {json.dumps(masked_payload)}")
        context_text = "\n\n".join(context_parts)
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
    retriever = QdrantRetriever(qdrant_client, collection_prefix + "procwise_procurement_embeddings", ollama)
    return HybridQueryEngine(graph_service, retriever, ollama)


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
