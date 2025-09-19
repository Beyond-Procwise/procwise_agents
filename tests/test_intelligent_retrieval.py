"""
Tests for the intelligent retrieval service.
"""

import pytest
from types import SimpleNamespace
from services.intelligent_retrieval import (
    IntelligentRetrievalService, 
    QueryIntent, 
    RetrievalStrategy
)


class DummyRAGService:
    """Mock RAG service for testing."""
    
    def __init__(self):
        self.search_calls = []
        self._reranker = SimpleNamespace(predict=lambda pairs: [0.5 for _ in pairs])
    
    def search(self, query, top_k=5, filters=None):
        self.search_calls.append({'query': query, 'top_k': top_k, 'filters': filters})
        # Return mock hits
        return [
            SimpleNamespace(
                id=f"doc_{i}",
                payload={
                    'content': f"Content for {query} document {i}",
                    'record_id': f"rec_{i}"
                },
                score=0.9 - i * 0.1
            )
            for i in range(min(top_k, 3))
        ]


class DummyAgentNick:
    """Mock agent nick for testing."""
    
    def __init__(self):
        self.settings = SimpleNamespace(extraction_model="test_model")
    
    def call_ollama(self, prompt, model=None, format=None):
        # Return mock LLM response based on prompt content
        if "alternative phrasings" in prompt.lower():
            if format == "json":
                return {"response": '["alternative query 1", "alternative query 2"]'}
            return {"response": "alternative query 1\nalternative query 2"}
        return {"response": "mock response"}


@pytest.fixture
def intelligent_retrieval():
    """Create an intelligent retrieval service for testing."""
    rag_service = DummyRAGService()
    agent_nick = DummyAgentNick()
    return IntelligentRetrievalService(rag_service, agent_nick)


class TestQueryIntentDetection:
    """Test query intent detection functionality."""
    
    def test_detect_factual_intent(self, intelligent_retrieval):
        queries = [
            "What is the cost of procurement software?",
            "Who is the supplier for office supplies?",
            "When is the contract deadline?",
            "Define procurement process"
        ]
        
        for query in queries:
            intent = intelligent_retrieval.detect_query_intent(query)
            assert intent == QueryIntent.FACTUAL
    
    def test_detect_procedural_intent(self, intelligent_retrieval):
        queries = [
            "How to implement a procurement process?",
            "How do I create a purchase order?",
            "Steps to evaluate suppliers",
            "Process for contract approval"
        ]
        
        for query in queries:
            intent = intelligent_retrieval.detect_query_intent(query)
            assert intent == QueryIntent.PROCEDURAL
    
    def test_detect_comparison_intent(self, intelligent_retrieval):
        queries = [
            "Compare supplier A vs supplier B",
            "What are the differences between these contracts?",
            "Which is better: in-house or outsourced procurement?",
            "Advantages and disadvantages of e-procurement"
        ]
        
        for query in queries:
            intent = intelligent_retrieval.detect_query_intent(query)
            assert intent == QueryIntent.COMPARISON
    
    def test_detect_compliance_intent(self, intelligent_retrieval):
        queries = [
            "Compliance requirements for government contracts",
            "What are the mandatory procurement policies?",
            "Legal obligations for supplier audits",
            "Regulatory standards for procurement"
        ]
        
        for query in queries:
            intent = intelligent_retrieval.detect_query_intent(query)
            assert intent == QueryIntent.COMPLIANCE
    
    def test_detect_analytical_intent(self, intelligent_retrieval):
        queries = [
            "Analyze procurement performance metrics",
            "Why did supplier costs increase?",
            "Trends in procurement spending",
            "Impact of digital transformation on procurement"
        ]
        
        for query in queries:
            intent = intelligent_retrieval.detect_query_intent(query)
            assert intent == QueryIntent.ANALYTICAL
    
    def test_default_contextual_intent(self, intelligent_retrieval):
        # Generic query without clear intent markers
        query = "Tell me about xyz"
        intent = intelligent_retrieval.detect_query_intent(query)
        assert intent == QueryIntent.CONTEXTUAL


class TestRetrievalStrategySelection:
    """Test retrieval strategy selection based on query intent."""
    
    def test_factual_uses_semantic_focused(self, intelligent_retrieval):
        strategy = intelligent_retrieval.select_retrieval_strategy(
            "What is procurement?", QueryIntent.FACTUAL
        )
        assert strategy == RetrievalStrategy.SEMANTIC_FOCUSED
    
    def test_procedural_uses_hybrid_balanced(self, intelligent_retrieval):
        strategy = intelligent_retrieval.select_retrieval_strategy(
            "How to procure supplies?", QueryIntent.PROCEDURAL
        )
        assert strategy == RetrievalStrategy.HYBRID_BALANCED
    
    def test_compliance_uses_keyword_focused(self, intelligent_retrieval):
        strategy = intelligent_retrieval.select_retrieval_strategy(
            "Compliance requirements", QueryIntent.COMPLIANCE
        )
        assert strategy == RetrievalStrategy.KEYWORD_FOCUSED
    
    def test_comparison_uses_semantic_focused(self, intelligent_retrieval):
        strategy = intelligent_retrieval.select_retrieval_strategy(
            "Compare suppliers", QueryIntent.COMPARISON
        )
        assert strategy == RetrievalStrategy.SEMANTIC_FOCUSED


class TestQueryExpansion:
    """Test intelligent query expansion functionality."""
    
    def test_domain_specific_expansion(self, intelligent_retrieval):
        query = "Find supplier information"
        expansions = intelligent_retrieval.intelligent_expand_query(query, QueryIntent.FACTUAL)
        
        # Should expand 'supplier' to related terms
        expansion_text = " ".join(expansions)
        assert any(term in expansion_text for term in ['vendor', 'contractor', 'provider'])
    
    def test_procedural_intent_expansion(self, intelligent_retrieval):
        query = "Create purchase order"
        expansions = intelligent_retrieval.intelligent_expand_query(query, QueryIntent.PROCEDURAL)
        
        # Should add procedural variants
        expansion_text = " ".join(expansions)
        assert any("how to" in exp.lower() for exp in expansions) or any("process" in exp.lower() for exp in expansions)
    
    def test_comparison_intent_expansion(self, intelligent_retrieval):
        query = "Supplier options"
        expansions = intelligent_retrieval.intelligent_expand_query(query, QueryIntent.COMPARISON)
        
        # Should add comparison variants
        expansion_text = " ".join(expansions)
        assert any("compare" in exp.lower() for exp in expansions) or any("comparison" in exp.lower() for exp in expansions)
    
    def test_compliance_intent_expansion(self, intelligent_retrieval):
        query = "Procurement rules"
        expansions = intelligent_retrieval.intelligent_expand_query(query, QueryIntent.COMPLIANCE)
        
        # Should add compliance variants
        expansion_text = " ".join(expansions)
        assert any(term in expansion_text.lower() for term in ['requirements', 'compliance', 'policy'])


class TestAdaptiveSearch:
    """Test the main adaptive search functionality."""
    
    def test_adaptive_search_calls_base_service(self, intelligent_retrieval):
        query = "What is procurement?"
        results = intelligent_retrieval.adaptive_search(query, top_k=3)
        
        # Should have called the base RAG service
        assert len(intelligent_retrieval.base_rag_service.search_calls) > 0
        
        # Should return results
        assert len(results) > 0
        assert all(hasattr(hit, 'id') for hit in results)
        assert all(hasattr(hit, 'payload') for hit in results)
    
    def test_adaptive_search_uses_expansions(self, intelligent_retrieval):
        query = "supplier information"
        intelligent_retrieval.adaptive_search(query, top_k=3)
        
        # Should have made multiple search calls (original + expansions)
        search_calls = intelligent_retrieval.base_rag_service.search_calls
        assert len(search_calls) > 1
        
        # First call should be the original query
        assert search_calls[0]['query'] == query
    
    def test_adaptive_search_removes_duplicates(self, intelligent_retrieval):
        # Mock service that returns duplicate IDs
        def mock_search(query, top_k=5, filters=None):
            return [
                SimpleNamespace(id="doc_1", payload={'content': f"Content {query}"}, score=0.9),
                SimpleNamespace(id="doc_1", payload={'content': f"Content {query} duplicate"}, score=0.8),
                SimpleNamespace(id="doc_2", payload={'content': f"Content {query} 2"}, score=0.7),
            ]
        
        intelligent_retrieval.base_rag_service.search = mock_search
        
        results = intelligent_retrieval.adaptive_search("test query", top_k=5)
        
        # Should remove duplicates
        result_ids = [hit.id for hit in results]
        assert len(result_ids) == len(set(result_ids)), "Results should not contain duplicate IDs"


class TestScoreAdjustments:
    """Test score adjustment logic."""
    
    def test_intent_score_adjustments(self, intelligent_retrieval):
        # Create a mock hit with compliance-related content
        hit = SimpleNamespace(
            id="test_doc",
            payload={'content': "This document covers compliance requirements and policies."},
            score=0.5
        )
        
        # Compliance intent should boost the score
        adjusted_score = intelligent_retrieval._apply_intent_score_adjustments(
            hit, 0.5, QueryIntent.COMPLIANCE, "compliance requirements"
        )
        
        assert adjusted_score > 0.5, "Compliance content should get score boost for compliance queries"
    
    def test_procedural_score_adjustments(self, intelligent_retrieval):
        # Create a mock hit with procedural content
        hit = SimpleNamespace(
            id="test_doc",
            payload={'content': "Follow these steps to complete the process: 1. Step one, 2. Step two"},
            score=0.5
        )
        
        # Procedural intent should boost the score
        adjusted_score = intelligent_retrieval._apply_intent_score_adjustments(
            hit, 0.5, QueryIntent.PROCEDURAL, "how to process"
        )
        
        assert adjusted_score > 0.5, "Procedural content should get score boost for procedural queries"


class TestDiversityFiltering:
    """Test diversity filtering to avoid redundant results."""
    
    def test_diversity_filter_keeps_diverse_content(self, intelligent_retrieval):
        hits = [
            SimpleNamespace(id="doc_1", payload={'content': "Supplier evaluation criteria"}, score=0.9),
            SimpleNamespace(id="doc_2", payload={'content': "Contract negotiation strategies"}, score=0.8),
            SimpleNamespace(id="doc_3", payload={'content': "Procurement automation tools"}, score=0.7),
        ]
        
        diverse_hits = intelligent_retrieval._apply_diversity_filter(hits, 3)
        
        # Should keep all diverse content
        assert len(diverse_hits) == 3
        assert [hit.id for hit in diverse_hits] == ["doc_1", "doc_2", "doc_3"]
    
    def test_diversity_filter_removes_similar_content(self, intelligent_retrieval):
        hits = [
            SimpleNamespace(id="doc_1", payload={'content': "Supplier evaluation criteria and assessment"}, score=0.9),
            SimpleNamespace(id="doc_2", payload={'content': "Supplier evaluation criteria and methods"}, score=0.8),
            SimpleNamespace(id="doc_3", payload={'content': "Contract negotiation strategies"}, score=0.7),
        ]
        
        diverse_hits = intelligent_retrieval._apply_diversity_filter(hits, 3)
        
        # Should filter out similar content (doc_2 is similar to doc_1)
        assert len(diverse_hits) <= 3
        result_ids = [hit.id for hit in diverse_hits]
        assert "doc_1" in result_ids  # Top result always kept
        assert "doc_3" in result_ids  # Different content should be kept
    
    def test_content_similarity_calculation(self, intelligent_retrieval):
        content1 = "supplier evaluation criteria and assessment methods"
        content2 = "supplier evaluation criteria and scoring methods"
        content3 = "contract negotiation strategies and techniques"
        
        # Similar content should have high similarity
        sim1_2 = intelligent_retrieval._calculate_content_similarity(content1, content2)
        assert sim1_2 > 0.5, "Similar content should have high similarity score"
        
        # Different content should have low similarity
        sim1_3 = intelligent_retrieval._calculate_content_similarity(content1, content3)
        assert sim1_3 < 0.5, "Different content should have low similarity score"