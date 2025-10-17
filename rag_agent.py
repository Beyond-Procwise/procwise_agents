from typing import Dict, List, Optional

from workflow_context_manager import WorkflowContextManager


class RAGAgent:
    """Advanced RAG with hybrid search for proc.* tables"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "phi4:latest"
        self.vector_store = None
        self.proc_tables = [
            "proc_contracts", "proc_sow", "proc_suppliers",
            "proc_historical_quotes", "proc_terms_conditions"
        ]

    async def initialize(self):
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
        # TODO: Integrate with vector DB
        return []

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        # TODO: Implement
        return []

    def _merge_results(self, vector_results: List, keyword_results: List) -> List:
        return vector_results + keyword_results

    async def _rerank(self, query: str, results: List, top_k: int) -> List:
        return results[:top_k]

    def _format_context(self, results: List) -> str:
        return "\n\n".join([r.get('content', '') for r in results])

    def _calculate_confidence(self, results: List) -> float:
        if not results:
            return 0.0
        avg_score = sum(r.get('score', 0) for r in results) / len(results)
        return min(avg_score, 1.0)

    async def _embed_table_data(self, table_name: str):
        # TODO: Integrate with DB
        pass

    async def _call_ollama(self, prompt: str) -> str:
        # TODO: Integrate
        return "[RAG response]"
