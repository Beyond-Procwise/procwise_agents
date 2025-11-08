from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from loguru import logger
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from retrying import retry
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import AppConfig, Neo4jConfig, OllamaConfig, PostgresConfig, QdrantConfig

SCHEMA = "proc"


class OllamaJsonClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=8000)
    def generate_json(self, system: str, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self._config.generate_model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        url = f"{self._config.base_url}/api/generate"
        response = requests.post(url, json=payload, timeout=self._config.timeout)
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "{}").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON from Ollama, attempting repair")
            repair_payload = {
                "model": self._config.generate_model,
                "system": "You are a JSON repair assistant. Return valid JSON only.",
                "prompt": f"Fix the JSON string: ```{content}```",
                "format": "json",
                "stream": False,
            }
            repair_response = requests.post(url, json=repair_payload, timeout=self._config.timeout)
            repair_response.raise_for_status()
            fixed = repair_response.json().get("response", "{}").strip()
            return json.loads(fixed)

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=8000)
    def embed(self, text: str) -> List[float]:
        payload = {
            "model": self._config.embedding_model,
            "input": [text],
        }
        url = f"{self._config.base_url}/api/embeddings"
        response = requests.post(url, json=payload, timeout=self._config.timeout)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


@dataclass
class SimilarNegotiation:
    rfq_id: str
    supplier_id: str
    round: int
    counter_offer: Optional[float]
    status: str
    similarity: float
    supplier_name: Optional[str]
    final_price: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rfq_id": self.rfq_id,
            "supplier_id": self.supplier_id,
            "round": self.round,
            "counter_offer": self.counter_offer,
            "status": self.status,
            "similarity": self.similarity,
            "supplier_name": self.supplier_name,
            "final_price": self.final_price,
        }


class HybridProcurementQueryEngine:
    def __init__(
        self,
        postgres_config: PostgresConfig,
        neo4j_config: Neo4jConfig,
        qdrant_config: QdrantConfig,
        ollama_config: OllamaConfig,
    ) -> None:
        self._postgres_config = postgres_config
        self._neo4j_config = neo4j_config
        self._qdrant_config = qdrant_config
        self._ollama_config = ollama_config

        self._engine: Engine = create_engine(self._postgres_config.dsn, pool_size=10, max_overflow=5)
        self._neo4j_driver = GraphDatabase.driver(
            self._neo4j_config.uri,
            auth=(self._neo4j_config.username, self._neo4j_config.password),
        )
        self._qdrant = QdrantClient(host=self._qdrant_config.host, port=self._qdrant_config.port, api_key=self._qdrant_config.api_key)
        self._ollama = OllamaJsonClient(self._ollama_config)

    @classmethod
    def from_config(cls, config: AppConfig) -> "HybridProcurementQueryEngine":
        return cls(config.postgres, config.neo4j, config.qdrant, config.ollama)

    def close(self) -> None:
        self._engine.dispose()
        self._neo4j_driver.close()

    # ------------------------------------------------------------------
    # LLM utilities
    # ------------------------------------------------------------------
    def generate_llm_json(self, system: str, prompt: str) -> Dict[str, Any]:
        return self._ollama.generate_json(system, prompt)

    def embed_text(self, text: str) -> List[float]:
        return self._ollama.embed(text)

    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        return self._fetch_dataframe(query, params)

    def run_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._run_cypher(query, parameters)

    def execute_scalar(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.scalar()

    def execute_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(query), params or {})

    def table_has_column(self, schema: str, table: str, column: str) -> bool:
        query = """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
              AND column_name = :column
            LIMIT 1
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text(query),
                {"schema": schema, "table": table, "column": column},
            )
            return result.scalar() is not None

    # ------------------------------------------------------------------
    # Helper query methods
    # ------------------------------------------------------------------
    def _fetch_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql_query(text(query), conn, params=params)

    def _run_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._neo4j_driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    # ------------------------------------------------------------------
    # Supplier performance metrics (Method 9)
    # ------------------------------------------------------------------
    def get_supplier_performance_metrics(self, supplier_id: str) -> Dict[str, Any]:
        logger.info("Fetching supplier performance metrics for {supplier_id}", supplier_id=supplier_id)
        quote_query = """
        MATCH (s:Supplier {supplier_id: $supplier_id})-[:SUBMITTED_QUOTE]->(q:Quote)
        OPTIONAL MATCH (q)-[:AWARDED_AS]->(po:PurchaseOrder)
        RETURN count(q) AS total_quotes,
               count(po) AS won_quotes,
               avg(q.total_amount_incl_tax) AS avg_quote_amount
        """
        quote_stats = self._run_cypher(quote_query, {"supplier_id": supplier_id})[0]

        contract_query = """
        MATCH (s:Supplier {supplier_id: $supplier_id})-[:HAS_CONTRACT]->(c:Contract)
        RETURN count(c) AS total_contracts, sum(c.total_contract_value) AS total_contract_value
        """
        contract_stats = self._run_cypher(contract_query, {"supplier_id": supplier_id})[0]

        negotiation_query = """
        MATCH (s:Supplier {supplier_id: $supplier_id})-[:NEGOTIATED_IN]->(n:NegotiationSession)
        RETURN count(n) AS total_negotiations, avg(n.round) AS avg_rounds
        """
        negotiation_stats = self._run_cypher(negotiation_query, {"supplier_id": supplier_id})[0]

        supplier_query = """
        MATCH (s:Supplier {supplier_id: $supplier_id})
        RETURN s.supplier_name AS supplier_name,
               s.risk_score AS risk_score,
               s.is_preferred_supplier AS is_preferred_supplier
        """
        supplier = self._run_cypher(supplier_query, {"supplier_id": supplier_id})[0]

        total_quotes = quote_stats.get("total_quotes", 0) or 0
        won_quotes = quote_stats.get("won_quotes", 0) or 0
        win_rate = (won_quotes / total_quotes * 100) if total_quotes else 0

        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier.get("supplier_name"),
            "quote_statistics": {
                "total_quotes": total_quotes,
                "won_quotes": won_quotes,
                "win_rate_percent": round(win_rate, 2),
                "avg_quote_amount": quote_stats.get("avg_quote_amount") or 0,
            },
            "contract_statistics": {
                "total_contracts": contract_stats.get("total_contracts", 0) or 0,
                "total_contract_value": contract_stats.get("total_contract_value") or 0,
            },
            "negotiation_statistics": {
                "total_negotiations": negotiation_stats.get("total_negotiations", 0) or 0,
                "avg_negotiation_rounds": negotiation_stats.get("avg_rounds") or 0,
            },
            "risk_indicators": {
                "risk_score": supplier.get("risk_score"),
                "is_preferred_supplier": supplier.get("is_preferred_supplier", False),
            },
        }

    # ------------------------------------------------------------------
    # Similar negotiations (Method 10)
    # ------------------------------------------------------------------
    def find_similar_negotiations(self, negotiation_context: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info("Searching for similar negotiations: {context}", context=negotiation_context)
        embedding = self._ollama.embed(negotiation_context)
        search_result = self._qdrant.search(
            collection_name="negotiations",
            query_vector=embedding,
            limit=top_k,
        )
        matches: List[SimilarNegotiation] = []
        for point in search_result:
            payload = point.payload or {}
            rfq_id = payload.get("rfq_id")
            supplier_id = payload.get("supplier_id")
            round_number = payload.get("round", 0)
            enrichment = self._run_cypher(
                """
                MATCH (n:NegotiationSession {rfq_id: $rfq_id, supplier_id: $supplier_id, round: $round})
                OPTIONAL MATCH (s:Supplier {supplier_id: $supplier_id})
                OPTIONAL MATCH (n)<-[:RESPONDS_TO]-(resp:SupplierResponse)
                RETURN s.supplier_name AS supplier_name,
                       n.counter_offer AS counter_offer,
                       n.status AS status,
                       resp.price AS final_price
                """,
                {"rfq_id": rfq_id, "supplier_id": supplier_id, "round": round_number},
            )
            supplier_name = enrichment[0]["supplier_name"] if enrichment else None
            counter_offer = enrichment[0]["counter_offer"] if enrichment else None
            status = enrichment[0]["status"] if enrichment else "unknown"
            final_price = enrichment[0]["final_price"] if enrichment else None
            matches.append(
                SimilarNegotiation(
                    rfq_id=rfq_id,
                    supplier_id=supplier_id,
                    round=round_number,
                    counter_offer=counter_offer,
                    status=status,
                    similarity=point.score,
                    supplier_name=supplier_name,
                    final_price=final_price,
                )
            )
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return [match.to_dict() for match in matches]

    # ------------------------------------------------------------------
    # Negotiation pattern analysis (Method 11)
    # ------------------------------------------------------------------
    def analyze_negotiation_pattern(self, supplier_id: str) -> Dict[str, Any]:
        history = self._run_cypher(
            """
            MATCH (s:Supplier {supplier_id: $supplier_id})-[:NEGOTIATED_IN]->(n:NegotiationSession)
            OPTIONAL MATCH (n)<-[:RESPONDS_TO]-(r:SupplierResponse)
            RETURN n.rfq_id AS rfq_id,
                   n.round AS round,
                   n.counter_offer AS counter_offer,
                   n.status AS status,
                   n.awaiting_response AS awaiting_response,
                   r.price AS supplier_price,
                   r.responded_at AS responded_at
            ORDER BY n.rfq_id, n.round
            """,
            {"supplier_id": supplier_id},
        )
        if not history:
            return {
                "supplier_id": supplier_id,
                "history": [],
                "llm_analysis": "No negotiation history available.",
            }
        prompt = """
        Summarise the negotiation behaviour using the provided JSON records.
        Focus on average number of rounds, pricing concessions, response speed, and recommendations for future strategy.
        Return keys: avg_rounds, price_patterns, response_patterns, recommendation.
        """
        llm_input = json.dumps(history)
        analysis = self._ollama.generate_json(
            system="You analyse supplier negotiation histories and respond with concise JSON reports.",
            prompt=f"Records: {llm_input}\n{prompt}",
        )
        return {
            "supplier_id": supplier_id,
            "history": history,
            "llm_analysis": analysis,
        }

    # ------------------------------------------------------------------
    # Natural language interface (Method 12)
    # ------------------------------------------------------------------
    def natural_language_query(self, question: str) -> Dict[str, Any]:
        classifier = self._ollama.generate_json(
            system="Classify procurement intelligence queries.",
            prompt=(
                "Return JSON with key query_type in {supplier_search, negotiation_analysis, spend_analysis, risk_insight}.\n"
                f"Question: {question}"
            ),
        )
        query_type = classifier.get("query_type", "negotiation_analysis")
        data: Any
        if query_type == "supplier_search":
            data = self._supplier_search(question)
        elif query_type == "negotiation_analysis":
            data = self.find_similar_negotiations(question)
        elif query_type == "spend_analysis":
            data = self._spend_analysis()
        else:
            data = self._risk_insight()

        summary = self._ollama.generate_json(
            system="You summarise procurement analytics in plain English.",
            prompt=f"Question: {question}\nData: {json.dumps(data)}\nReturn JSON with key answer",
        )
        return {
            "question": question,
            "query_type": query_type,
            "data": data,
            "answer": summary.get("answer", ""),
        }

    # ------------------------------------------------------------------
    # Support methods for agent integration
    # ------------------------------------------------------------------
    def get_supplier_quote_summary(self, supplier_id: str) -> Dict[str, Any]:
        query = """
        SELECT supplier_id,
               COUNT(*) FILTER (WHERE awarded) AS awarded_quotes,
               COUNT(*) AS total_quotes,
               AVG(total_amount_incl_tax) AS avg_quote
        FROM (
            SELECT qa.supplier_id,
                   qa.total_amount_incl_tax,
                   (qa.po_id IS NOT NULL) AS awarded
            FROM proc.quote_agent qa
            WHERE qa.supplier_id = :supplier_id
        ) sub
        GROUP BY supplier_id
        """
        df = self._fetch_dataframe(query, {"supplier_id": supplier_id})
        if df.empty:
            return {"awarded_quotes": 0, "total_quotes": 0, "avg_quote": None}
        row = df.iloc[0]
        return {
            "awarded_quotes": int(row["awarded_quotes"]),
            "total_quotes": int(row["total_quotes"]),
            "avg_quote": float(row["avg_quote"]) if row["avg_quote"] is not None else None,
        }

    def get_supplier_response_stats(self, supplier_id: str) -> Dict[str, Any]:
        query = """
        SELECT supplier_id,
               AVG(EXTRACT(EPOCH FROM (responded_at - dispatched_at)) / 3600) AS avg_response_hours,
               COUNT(*) FILTER (WHERE responded_at IS NOT NULL) AS responded,
               COUNT(*) AS total
        FROM proc.workflow_email_tracking
        WHERE supplier_id = :supplier_id
        GROUP BY supplier_id
        """
        df = self._fetch_dataframe(query, {"supplier_id": supplier_id})
        if df.empty:
            return {"avg_response_hours": None, "responded": 0, "total": 0}
        row = df.iloc[0]
        return {
            "avg_response_hours": float(row["avg_response_hours"]) if row["avg_response_hours"] is not None else None,
            "responded": int(row["responded"]),
            "total": int(row["total"]),
        }

    def get_workflow_overview(self, workflow_id: str) -> Dict[str, Any]:
        query = """
        SELECT supplier_id,
               workflow_id,
               unique_id,
               dispatched_at,
               responded_at,
               matched
        FROM proc.workflow_email_tracking
        WHERE workflow_id = :workflow_id
        """
        df = self._fetch_dataframe(query, {"workflow_id": workflow_id})
        responses = self._fetch_dataframe(
            """
            SELECT workflow_id,
                   unique_id,
                   supplier_id,
                   price,
                   responded_at
            FROM proc.supplier_response
            WHERE workflow_id = :workflow_id
            """,
            {"workflow_id": workflow_id},
        )
        response_lookup = {
            (row["supplier_id"], row["unique_id"]): row for _, row in responses.iterrows()
        }
        suppliers = []
        for _, row in df.iterrows():
            supplier_entry = response_lookup.get((row["supplier_id"], row["unique_id"]))
            suppliers.append(
                {
                    "supplier_id": row["supplier_id"],
                    "has_responded": row["responded_at"] is not None,
                    "quoted_price": float(supplier_entry["price"]) if supplier_entry is not None and supplier_entry["price"] is not None else None,
                    "response_time": row["responded_at"],
                }
            )
        responded_count = sum(1 for supplier in suppliers if supplier["has_responded"])
        total = len(suppliers)
        completion = (responded_count / total * 100) if total else 0
        ready = all(supplier["has_responded"] for supplier in suppliers) and total > 0
        return {
            "workflow_id": workflow_id,
            "total_suppliers": total,
            "responded_count": responded_count,
            "pending_count": total - responded_count,
            "completion_percentage": round(completion, 2),
            "suppliers": suppliers,
            "ready_for_negotiation": ready,
        }

    def get_email_thread(self, rfq_id: str, supplier_id: str) -> Dict[str, Any]:
        query = """
        SELECT workflow_id,
               unique_id,
               message_id,
               array_remove(string_to_array(references, ','), '') AS references,
               responded_at,
               dispatched_at
        FROM proc.workflow_email_tracking
        WHERE rfq_id = :rfq_id AND supplier_id = :supplier_id
        ORDER BY dispatched_at
        """
        df = self._fetch_dataframe(query, {"rfq_id": rfq_id, "supplier_id": supplier_id})
        if df.empty:
            return {
                "workflow_id": None,
                "threading_headers": {
                    "In-Reply-To": None,
                    "References": [],
                    "Thread-Index": "0",
                },
            }
        workflow_id = df.iloc[0]["workflow_id"]
        references = []
        for _, row in df.iterrows():
            if row["message_id"]:
                references.append(row["message_id"])
        last_message = references[-1] if references else None
        thread_index = str(len(references))
        response_df = self._fetch_dataframe(
            """
            SELECT message_id
            FROM proc.supplier_response
            WHERE rfq_id = :rfq_id AND supplier_id = :supplier_id
            ORDER BY responded_at DESC
            LIMIT 1
            """,
            {"rfq_id": rfq_id, "supplier_id": supplier_id},
        )
        if not response_df.empty and response_df.iloc[0]["message_id"]:
            references.append(response_df.iloc[0]["message_id"])
            last_message = response_df.iloc[0]["message_id"]
        return {
            "workflow_id": workflow_id,
            "threading_headers": {
                "In-Reply-To": last_message,
                "References": references,
                "Thread-Index": thread_index,
            },
        }

    def _supplier_search(self, question: str) -> List[Dict[str, Any]]:
        vector = self._ollama.embed(question)
        results = self._qdrant.search("suppliers", query_vector=vector, limit=5)
        return [dict(hit.payload or {}, score=hit.score) for hit in results]

    def _spend_analysis(self) -> List[Dict[str, Any]]:
        query = """
        SELECT supplier_name,
               SUM(total_amount) AS spend,
               COUNT(*) AS po_count
        FROM proc.purchase_order_agent
        GROUP BY supplier_name
        ORDER BY spend DESC
        LIMIT 10
        """
        df = self._fetch_dataframe(query)
        return df.to_dict(orient="records")

    def _risk_insight(self) -> List[Dict[str, Any]]:
        query = """
        SELECT supplier_id,
               supplier_name,
               risk_score,
               is_preferred_supplier
        FROM proc.supplier
        ORDER BY risk_score DESC NULLS LAST
        LIMIT 10
        """
        df = self._fetch_dataframe(query)
        return df.to_dict(orient="records")


__all__ = [
    "HybridProcurementQueryEngine",
]
