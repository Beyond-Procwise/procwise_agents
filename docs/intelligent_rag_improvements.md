# Intelligent RAG Agent Improvements

This document describes the enhancements made to the RAG (Retrieval-Augmented Generation) agent to provide more intelligent and contextually relevant responses.

## Overview

The RAG agent has been significantly improved with a new `IntelligentRetrievalService` that provides:

1. **Query Intent Detection** - Automatically identifies the type of question being asked
2. **Adaptive Query Expansion** - Generates relevant alternative phrasings based on domain knowledge
3. **Dynamic Retrieval Strategies** - Selects optimal retrieval approaches based on query characteristics
4. **Context-Aware Re-ranking** - Improves document scoring based on query intent
5. **Diversity Filtering** - Reduces redundant results in the response

## Key Components

### Query Intent Detection

The system automatically classifies queries into different intent categories:

- **FACTUAL**: Direct factual questions (What, Who, When, Where)
- **PROCEDURAL**: How-to questions and process inquiries
- **COMPARISON**: Comparative questions and evaluation requests
- **COMPLIANCE**: Regulatory and policy-related questions
- **ANALYTICAL**: Analysis and insights requests
- **CONTEXTUAL**: Context-dependent or general questions

### Retrieval Strategies

Different retrieval strategies are applied based on query intent:

- **SEMANTIC_FOCUSED**: Emphasizes semantic similarity (for factual, comparison, analytical queries)
- **KEYWORD_FOCUSED**: Emphasizes exact keyword matching (for compliance queries)
- **HYBRID_BALANCED**: Balanced approach (for procedural queries)
- **TEMPORAL_AWARE**: Considers time-based relevance (future enhancement)

### Domain-Specific Query Expansion

The system includes procurement-specific knowledge for expanding queries:

- `supplier` → `vendor`, `contractor`, `provider`, `partner`
- `procurement` → `purchasing`, `sourcing`, `acquisition`, `buying`
- `contract` → `agreement`, `deal`, `arrangement`, `terms`
- `tender` → `bid`, `proposal`, `rfp`, `rfq`, `quotation`

## Implementation Details

### Core Files Added/Modified

1. **`services/intelligent_retrieval.py`** - New intelligent retrieval service
2. **`agents/rag_agent.py`** - Updated to use intelligent retrieval
3. **`tests/test_intelligent_retrieval.py`** - Comprehensive test suite
4. **`tests/test_rag_agent.py`** - Updated tests for RAG agent

### Integration with Existing System

The intelligent retrieval service is designed to work seamlessly with the existing RAG infrastructure:

- Falls back to original retrieval method if intelligent retrieval fails
- Maintains compatibility with existing filters and parameters
- Uses the same cross-encoder re-ranking model
- Preserves all existing functionality

### Error Handling and Fallbacks

- If intelligent retrieval fails, the system automatically falls back to the original retrieval method
- LLM-based query expansion is optional and gracefully handles failures
- All components are designed to degrade gracefully

## Usage Examples

### Example 1: Factual Query
```python
query = "What is the cost of procurement software?"
# Intent: FACTUAL
# Strategy: SEMANTIC_FOCUSED
# Expansions: ["What is the price of procurement software?", "Cost of purchasing tools", ...]
```

### Example 2: Procedural Query
```python
query = "How to implement a supplier evaluation process?"
# Intent: PROCEDURAL  
# Strategy: HYBRID_BALANCED
# Expansions: ["steps to implement supplier evaluation", "process for evaluating suppliers", ...]
```

### Example 3: Compliance Query
```python
query = "What are the mandatory procurement policies?"
# Intent: COMPLIANCE
# Strategy: KEYWORD_FOCUSED
# Expansions: ["procurement policies requirements", "mandatory purchasing rules", ...]
```

## Performance Improvements

The intelligent retrieval system provides several performance benefits:

1. **Better Precision**: Intent-aware retrieval reduces irrelevant results
2. **Enhanced Recall**: Domain-specific query expansion finds more relevant documents
3. **Contextual Ranking**: Intent-based scoring prioritizes relevant content
4. **Diversity**: Filtering reduces redundant information in responses
5. **Adaptability**: Different strategies for different question types

## Testing

The implementation includes comprehensive tests covering:

- Query intent detection accuracy
- Retrieval strategy selection logic
- Query expansion functionality
- Adaptive search behavior
- Score adjustment mechanisms
- Diversity filtering
- Integration with existing RAG agent

Run tests with:
```bash
PYTHONPATH=. pytest tests/test_intelligent_retrieval.py tests/test_rag_agent.py -v
```

## Demo

A demonstration script is provided to showcase the improvements:

```bash
PYTHONPATH=. python demo_intelligent_retrieval.py
```

This script demonstrates:
- Query intent detection for various question types
- Intelligent query expansion with domain knowledge
- Adaptive search with different strategies
- Context-aware document ranking

## Future Enhancements

Potential future improvements include:

1. **Learning from User Feedback**: Incorporate user feedback to improve intent detection
2. **Advanced Entity Recognition**: Use NER to better understand query entities
3. **Temporal Awareness**: Implement time-based relevance for recent information
4. **Multi-language Support**: Extend intent detection to multiple languages
5. **Custom Domain Adaptation**: Allow customization of domain-specific expansions
6. **Performance Monitoring**: Add metrics to track retrieval effectiveness

## Configuration

The intelligent retrieval service can be configured through the existing settings:

- `reranker_model`: Cross-encoder model for re-ranking (inherited)
- `extraction_model`: LLM model for query expansion (inherited)

## Backward Compatibility

All changes are designed to be backward compatible:

- Existing RAG agent functionality is preserved
- Original retrieval methods are used as fallbacks
- No breaking changes to existing APIs
- Tests ensure compatibility is maintained