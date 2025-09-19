"""
Intelligent retrieval service that enhances RAG with context-aware retrieval strategies.

This module provides enhanced retrieval capabilities including:
- Query intent detection for different question types
- Adaptive query expansion with domain knowledge
- Dynamic retrieval strategy selection
- Context-aware document ranking and filtering
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from types import SimpleNamespace
import re

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Different types of procurement-related query intents."""
    FACTUAL = "factual"  # Direct factual questions
    PROCEDURAL = "procedural"  # How-to questions
    COMPARISON = "comparison"  # Compare different options
    COMPLIANCE = "compliance"  # Regulatory/policy questions
    ANALYTICAL = "analytical"  # Analysis and insights
    CONTEXTUAL = "contextual"  # Context-dependent questions


class RetrievalStrategy(Enum):
    """Different retrieval strategies for different query types."""
    SEMANTIC_FOCUSED = "semantic_focused"  # Emphasize semantic similarity
    KEYWORD_FOCUSED = "keyword_focused"  # Emphasize keyword matching
    HYBRID_BALANCED = "hybrid_balanced"  # Balanced semantic + keyword
    TEMPORAL_AWARE = "temporal_aware"  # Consider time-based relevance


class IntelligentRetrievalService:
    """Enhanced retrieval service with intelligent query processing."""
    
    def __init__(self, base_rag_service, agent_nick):
        self.base_rag_service = base_rag_service
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        
        # Query intent patterns for procurement domain
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what|who|when|where|which)\b',
                r'\b(define|definition|meaning)\b',
                r'\b(cost|price|amount|value)\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|how do|how can|process|steps|procedure)\b',
                r'\b(implement|execute|perform|conduct)\b',
                r'\b(guide|tutorial|instruction)\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(compare|comparison|vs|versus|better|best|difference)\b',
                r'\b(alternatives|options|choice)\b',
                r'\b(advantages|disadvantages|pros|cons)\b'
            ],
            QueryIntent.COMPLIANCE: [
                r'\b(compliance|regulation|policy|law|legal|requirement)\b',
                r'\b(must|should|required|mandatory|obligatory)\b',
                r'\b(audit|governance|standard)\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\b(analyze|analysis|insights|trends|patterns)\b',
                r'\b(performance|metrics|kpi|report)\b',
                r'\b(why|reason|cause|impact|effect)\b'
            ]
        }
        
        # Domain-specific expansion terms for procurement
        self.domain_expansions = {
            'supplier': ['vendor', 'contractor', 'provider', 'partner'],
            'procurement': ['purchasing', 'sourcing', 'acquisition', 'buying'],
            'contract': ['agreement', 'deal', 'arrangement', 'terms'],
            'tender': ['bid', 'proposal', 'rfp', 'rfq', 'quotation'],
            'cost': ['price', 'expense', 'fee', 'rate', 'budget'],
            'quality': ['standard', 'specification', 'requirement', 'criteria'],
            'delivery': ['shipment', 'fulfillment', 'timeline', 'schedule'],
            'compliance': ['regulation', 'policy', 'governance', 'standard']
        }
    
    def detect_query_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a query based on linguistic patterns."""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score
        
        # Check for specific patterns that should override generic ones
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference', 'better', 'advantages', 'disadvantages']):
            intent_scores[QueryIntent.COMPARISON] += 2
        
        if any(word in query_lower for word in ['compliance', 'regulation', 'policy', 'mandatory', 'required', 'legal', 'regulatory']):
            intent_scores[QueryIntent.COMPLIANCE] += 2
        
        # Special handling for "what are" questions that might be compliance related
        if 'what are' in query_lower and any(word in query_lower for word in ['mandatory', 'required', 'policy']):
            intent_scores[QueryIntent.COMPLIANCE] += 1
        
        # Default to contextual if no clear intent detected
        if max(intent_scores.values()) == 0:
            return QueryIntent.CONTEXTUAL
        
        return max(intent_scores, key=intent_scores.get)
    
    def select_retrieval_strategy(self, query: str, intent: QueryIntent) -> RetrievalStrategy:
        """Select the optimal retrieval strategy based on query characteristics."""
        query_lower = query.lower()
        
        # Strategy selection based on intent and query characteristics
        if intent == QueryIntent.FACTUAL:
            # Factual questions benefit from semantic similarity
            return RetrievalStrategy.SEMANTIC_FOCUSED
        
        elif intent == QueryIntent.PROCEDURAL:
            # Procedural questions need balanced approach
            return RetrievalStrategy.HYBRID_BALANCED
        
        elif intent == QueryIntent.COMPARISON:
            # Comparison questions need semantic understanding
            return RetrievalStrategy.SEMANTIC_FOCUSED
        
        elif intent == QueryIntent.COMPLIANCE:
            # Compliance questions often need exact keyword matches
            return RetrievalStrategy.KEYWORD_FOCUSED
        
        elif intent == QueryIntent.ANALYTICAL:
            # Analytical questions benefit from semantic similarity
            return RetrievalStrategy.SEMANTIC_FOCUSED
        
        else:
            # Default balanced approach
            return RetrievalStrategy.HYBRID_BALANCED
    
    def intelligent_expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate intelligent query expansions based on intent and domain knowledge."""
        expansions = []
        query_lower = query.lower()
        
        # Domain-specific expansions
        for term, synonyms in self.domain_expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expansion = query_lower.replace(term, synonym)
                    if expansion != query_lower:
                        expansions.append(expansion)
        
        # Intent-specific expansions
        if intent == QueryIntent.PROCEDURAL:
            # Add procedural variants
            if not any(word in query_lower for word in ['how', 'step', 'process']):
                expansions.append(f"how to {query}")
                expansions.append(f"process for {query}")
        
        elif intent == QueryIntent.COMPARISON:
            # Add comparison variants
            if 'compare' not in query_lower:
                expansions.append(f"compare {query}")
                expansions.append(f"{query} comparison")
        
        elif intent == QueryIntent.COMPLIANCE:
            # Add compliance variants
            expansions.append(f"{query} requirements")
            expansions.append(f"{query} compliance")
            expansions.append(f"{query} policy")
        
        # Generate semantic variations using LLM
        llm_expansions = self._generate_llm_expansions(query, intent)
        expansions.extend(llm_expansions)
        
        # Remove duplicates and return top expansions
        unique_expansions = list(dict.fromkeys(expansions))
        return unique_expansions[:5]  # Limit to top 5 expansions
    
    def _generate_llm_expansions(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate query expansions using LLM with context awareness."""
        intent_context = {
            QueryIntent.FACTUAL: "factual information requests",
            QueryIntent.PROCEDURAL: "step-by-step process questions", 
            QueryIntent.COMPARISON: "comparison and evaluation questions",
            QueryIntent.COMPLIANCE: "regulatory and compliance questions",
            QueryIntent.ANALYTICAL: "analytical and insights questions",
            QueryIntent.CONTEXTUAL: "context-dependent questions"
        }
        
        prompt = (
            f"Generate 3 alternative phrasings for this procurement {intent_context[intent]}. "
            f"Focus on maintaining the original meaning while using different terms and structures "
            f"commonly used in procurement and business contexts. "
            f"Return only the alternatives as a JSON list.\n\n"
            f"Original query: {query}\n"
            f"JSON response:"
        )
        
        try:
            # Use the base agent's call_ollama method if available
            if hasattr(self.agent_nick, 'call_ollama'):
                resp = self.agent_nick.call_ollama(prompt, model=self.settings.extraction_model, format="json")
            else:
                # Fallback for testing or when base agent methods aren't available
                return []
            
            data = json.loads(resp.get("response", "[]"))
            if isinstance(data, list):
                return [e for e in data if isinstance(e, str) and e.strip()]
        except Exception as e:
            logger.debug(f"LLM expansion failed: {e}")
        
        return []
    
    def adaptive_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Any] = None,
        **kwargs
    ) -> List[Any]:
        """Perform intelligent search with adaptive strategies."""
        
        # Step 1: Detect query intent
        intent = self.detect_query_intent(query)
        logger.debug(f"Detected query intent: {intent.value}")
        
        # Step 2: Select retrieval strategy
        strategy = self.select_retrieval_strategy(query, intent)
        logger.debug(f"Selected strategy: {strategy.value}")
        
        # Step 3: Generate intelligent query expansions
        expansions = self.intelligent_expand_query(query, intent)
        logger.debug(f"Generated {len(expansions)} query expansions")
        
        # Step 4: Execute retrieval with strategy-specific parameters
        all_hits = []
        seen_ids = set()
        
        # Primary query search
        query_variants = [query] + expansions
        
        for i, q_variant in enumerate(query_variants):
            # Adjust search parameters based on strategy
            search_params = self._get_strategy_parameters(strategy, intent, i == 0)
            
            hits = self.base_rag_service.search(
                q_variant,
                top_k=top_k * 3,  # Get more candidates for better filtering
                filters=filters
            )
            
            # Add hits with strategy-specific scoring adjustments
            for hit in hits:
                if hit.id not in seen_ids:
                    # Adjust scores based on strategy and position in expansion
                    adjusted_hit = self._adjust_hit_score(hit, strategy, i == 0, intent)
                    all_hits.append(adjusted_hit)
                    seen_ids.add(hit.id)
        
        # Step 5: Intelligent re-ranking with intent awareness
        ranked_hits = self._intelligent_rerank(query, all_hits, intent, strategy, top_k)
        
        return ranked_hits
    
    def _get_strategy_parameters(self, strategy: RetrievalStrategy, intent: QueryIntent, is_primary: bool) -> Dict:
        """Get search parameters based on retrieval strategy."""
        params = {}
        
        if strategy == RetrievalStrategy.SEMANTIC_FOCUSED:
            # Favor semantic similarity
            params['semantic_weight'] = 0.7
            params['keyword_weight'] = 0.3
        
        elif strategy == RetrievalStrategy.KEYWORD_FOCUSED:
            # Favor keyword matching
            params['semantic_weight'] = 0.3
            params['keyword_weight'] = 0.7
        
        else:  # HYBRID_BALANCED
            params['semantic_weight'] = 0.5
            params['keyword_weight'] = 0.5
        
        return params
    
    def _adjust_hit_score(self, hit, strategy: RetrievalStrategy, is_primary: bool, intent: QueryIntent):
        """Adjust hit scores based on retrieval strategy and other factors."""
        # Apply penalty for non-primary queries (expansions)
        score_multiplier = 1.0 if is_primary else 0.8
        
        # Apply intent-specific adjustments
        if intent == QueryIntent.COMPLIANCE and 'compliance' in hit.payload.get('content', '').lower():
            score_multiplier *= 1.2
        elif intent == QueryIntent.PROCEDURAL and any(word in hit.payload.get('content', '').lower() 
                                                      for word in ['step', 'process', 'procedure', 'how']):
            score_multiplier *= 1.1
        
        # Create adjusted hit by copying the original and adjusting score
        adjusted_hit = SimpleNamespace(
            id=hit.id,
            payload=hit.payload,
            score=hit.score * score_multiplier
        )
        return adjusted_hit
    
    def _intelligent_rerank(
        self, 
        query: str, 
        hits: List[Any], 
        intent: QueryIntent, 
        strategy: RetrievalStrategy, 
        top_k: int
    ) -> List[Any]:
        """Perform intelligent re-ranking with intent and strategy awareness."""
        
        if not hits:
            return []
        
        # Use base re-ranking
        pairs = [
            (query, hit.payload.get("content", hit.payload.get("summary", "")))
            for hit in hits
        ]
        
        # Get cross-encoder scores
        scores = self.base_rag_service._reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        
        # Apply intent-specific score adjustments
        adjusted_scores = []
        for i, (hit, score) in enumerate(zip(hits, scores)):
            adjusted_score = self._apply_intent_score_adjustments(hit, score, intent, query)
            adjusted_scores.append(adjusted_score)
        
        # Sort and return top results
        ranked = sorted(zip(hits, adjusted_scores), key=lambda x: x[1], reverse=True)
        
        # Apply diversity filter to avoid too similar results
        diverse_hits = self._apply_diversity_filter([h for h, _ in ranked], top_k)
        
        return diverse_hits[:top_k]
    
    def _apply_intent_score_adjustments(self, hit, base_score: float, intent: QueryIntent, query: str) -> float:
        """Apply intent-specific score adjustments."""
        content = hit.payload.get('content', '').lower()
        
        # Intent-specific boosting
        if intent == QueryIntent.COMPLIANCE:
            if any(term in content for term in ['policy', 'regulation', 'compliance', 'requirement']):
                base_score *= 1.15
        
        elif intent == QueryIntent.PROCEDURAL:
            if any(term in content for term in ['step', 'process', 'procedure', 'how to', 'guide']):
                base_score *= 1.1
        
        elif intent == QueryIntent.COMPARISON:
            if any(term in content for term in ['compare', 'versus', 'difference', 'better', 'best']):
                base_score *= 1.1
        
        elif intent == QueryIntent.ANALYTICAL:
            if any(term in content for term in ['analysis', 'insight', 'trend', 'performance', 'metric']):
                base_score *= 1.1
        
        return base_score
    
    def _apply_diversity_filter(self, hits: List[Any], target_count: int) -> List[Any]:
        """Apply diversity filtering to reduce redundant results."""
        if len(hits) <= target_count:
            return hits
        
        diverse_hits = [hits[0]]  # Always include top result
        
        for hit in hits[1:]:
            if len(diverse_hits) >= target_count:
                break
            
            # Check content similarity with already selected hits
            is_diverse = True
            hit_content = hit.payload.get('content', '')
            
            for selected_hit in diverse_hits:
                selected_content = selected_hit.payload.get('content', '')
                
                # Simple diversity check based on content overlap
                similarity = self._calculate_content_similarity(hit_content, selected_content)
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_hits.append(hit)
        
        return diverse_hits
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity based on word overlap."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0