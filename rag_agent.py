from typing import Dict, List

from workflow_context_manager import WorkflowContextManager


class RAGAgent:
    """Advanced RAG with hybrid search for proc.* tables"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "phi4:latest"
        self.vector_store: List[Dict] = []
        self.proc_tables = [
            "proc_contracts", "proc_sow", "proc_suppliers",
            "proc_historical_quotes", "proc_terms_conditions"
        ]

    async def initialize(self):
        self.vector_store.clear()
        for table in self.proc_tables:
            await self._embed_table_data(table)
        print("✅ RAG initialized")

    async def query(self, user_question: str, context_filters: Optional[Dict] = None) -> Dict:
        enhanced_query = await self._enhance_query(user_question)

        vector_results = await self._vector_search(enhanced_query, top_k=20)
        keyword_results = await self._keyword_search(enhanced_query, top_k=20)

        all_results = self._merge_results(vector_results, keyword_results)
        reranked = await self._rerank(user_question, all_results, top_k=5)

        context = self._format_context(reranked)
        prompt = f"""
You are procurement expert with access to:
- Contracts (proc_contracts)
- SOW documents (proc_sow)
- Supplier data (proc_suppliers)
- Historical quotes (proc_historical_quotes)
- Terms & conditions (proc_terms_conditions)

Question: {user_question}

Context:
{context}

Provide accurate, context-aware answer with specific data references.
"""

        response = await self._call_ollama(prompt)
        confidence = self._calculate_confidence(reranked)

        return {
            "answer": response,
            "sources": reranked,
            "confidence": confidence,
            "requires_human_review": confidence < 0.7
        }

    async def _enhance_query(self, query: str) -> str:
        return query

    async def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        query_terms = set(query.lower().split())
        scored: List[Dict] = []
        for idx, entry in enumerate(self.vector_store):
            doc_type = entry.get("doc_type")
            content = entry.get("content", "")
            base_score = 0.4
            if doc_type in {"contract", "sow"}:
                base_score += 0.3
            if doc_type == "supplier_profile":
                base_score += 0.15
            overlap = query_terms.intersection(content.lower().split())
            if overlap:
                base_score += min(len(overlap) * 0.05, 0.2)
            scored.append({**entry, "score": min(base_score, 1.0), "id": idx})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        query_lower = query.lower()
        matches: List[Dict] = []
        for idx, entry in enumerate(self.vector_store):
            if query_lower in entry.get("content", "").lower():
                score = 0.5 if entry.get("doc_type") in {"contract", "sow"} else 0.3
                matches.append({**entry, "score": score, "id": idx})
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    def _merge_results(self, vector_results: List, keyword_results: List) -> List:
        merged: Dict[int, Dict] = {}
        for entry in vector_results + keyword_results:
            entry_id = entry.get("id")
            if entry_id in merged:
                merged[entry_id]["score"] = max(merged[entry_id]["score"], entry.get("score", 0))
            else:
                merged[entry_id] = dict(entry)
        return list(merged.values())

    async def _rerank(self, query: str, results: List, top_k: int) -> List:
        return sorted(results, key=lambda item: item.get("score", 0), reverse=True)[:top_k]

    def _format_context(self, results: List) -> str:
        return "\n\n".join([r.get('content', '') for r in results])

    def _calculate_confidence(self, results: List) -> float:
        if not results:
            return 0.0
        avg_score = sum(r.get('score', 0) for r in results) / len(results)
        return min(avg_score, 1.0)

    async def _embed_table_data(self, table_name: str):
        doc_type = self._infer_doc_type(table_name)
        placeholder = {
            "table": table_name,
            "doc_type": doc_type,
            "content": f"Indexed {doc_type} data from {table_name} for supplier negotiations.",
            "score": 0.5,
        }
        self.vector_store.append(placeholder)

    def _infer_doc_type(self, table_name: str) -> str:
        if table_name.endswith("contracts"):
            return "contract"
        if table_name.endswith("sow"):
            return "sow"
        if table_name.endswith("suppliers"):
            return "supplier_profile"
        return "reference"

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate
        return "[RAG response]"
