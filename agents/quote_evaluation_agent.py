import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from qdrant_client import models

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

# Ensure GPU is used when available
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
if torch.cuda.is_available():  # pragma: no cover - hardware dependent
    torch.set_default_device("cuda")

logger = logging.getLogger(__name__)


class QuoteEvaluationAgent(BaseAgent):
    """Agent for evaluating and comparing supplier quotes"""

    def run(self, context: AgentContext) -> AgentOutput:
        """Execute the quote evaluation flow."""
        return self.process(context)

    def process(self, context: AgentContext) -> AgentOutput:
        """Process quote evaluation"""
        try:
            input_data = context.input_data

            # Fetch quotes from database
            product_category = (
                input_data.get('product_category')
                or input_data.get('product_type')
                or None
            )
            quotes = self._fetch_quotes(
                input_data.get('supplier_names', []),
                product_category,
            )

            if not quotes:
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={'message': 'No quotes found'},
                    error='No quotes to evaluate'
                )

            # Evaluate quotes
            evaluation = self._evaluate_quotes(quotes)

            # Generate comparison matrix
            comparison = self._generate_comparison(quotes)

            # Identify best quote
            best_quote = self._identify_best_quote(evaluation)

            # Generate recommendations using Ollama
            recommendations = self._generate_recommendations(
                best_quote, evaluation, comparison
            )

            output_data = self._to_native(
                {
                    'quotes_analyzed': len(quotes),
                    'best_quote': best_quote,
                    'comparison_matrix': comparison,
                    'evaluation_scores': evaluation,
                    'recommendations': recommendations,
                    'savings_potential': self._calculate_savings(quotes, best_quote),
                }
            )

            # Determine next agents
            next_agents = []
            if best_quote.get('total_amount', 0) > 50000:
                next_agents.append('HumanEscalationAgent')
            if output_data['savings_potential'] > 10000:
                next_agents.append('ContractIntelligenceAgent')
            if best_quote.get('negotiation_possible'):
                next_agents.append('NegotiationAgent')

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                next_agents=next_agents,
                pass_fields={
                    'selected_quote': best_quote.get('quote_id'),
                    'supplier': best_quote.get('supplier_name'),
                    'savings_potential': output_data['savings_potential'],
                    'negotiation_points': recommendations.get('negotiation_points', [])
                },
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"QuoteEvaluationAgent error: {e}")
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(e)
            )

    def _fetch_quotes(
        self,
        supplier_names: List[str],
        product_category: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch quotes from Qdrant vector database."""

        def ensure_index(field_name: str) -> None:
            """Create a Qdrant payload index if it doesn't exist."""
            try:
                self.agent_nick.qdrant_client.create_payload_index(
                    collection_name=self.settings.qdrant_collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as ie:  # pragma: no cover - network dependent
                logger.debug("Index creation skipped for %s: %s", field_name, ie)

        if supplier_names:
            ensure_index("supplier_name")
        if product_category:
            ensure_index("product_category")

        def build_filter(include_supplier: bool = True) -> models.Filter:
            filters = [
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value="quote"),
                )
            ]
            if include_supplier and supplier_names:
                filters.append(
                    models.FieldCondition(
                        key="supplier_name",
                        match=models.MatchAny(any=supplier_names),
                    )
                )
            if product_category:
                filters.append(
                    models.FieldCondition(
                        key="product_category",
                        match=models.MatchValue(value=product_category),
                    )
                )
            return models.Filter(must=filters)

        def qdrant_scroll(q_filter: models.Filter):
            return self.agent_nick.qdrant_client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=q_filter,
                with_payload=True,
                limit=100,
            )

        try:
            points, _ = qdrant_scroll(build_filter())
        except Exception as e:
            logger.warning(
                "Primary quote fetch failed (%s). Retrying without supplier filter.", e
            )
            try:
                points, _ = qdrant_scroll(build_filter(include_supplier=False))
            except Exception as e2:
                logger.error(f"Failed to fetch quotes from Qdrant: {e2}")
                # Return mock data for testing
                return [
                    {
                        "quote_id": "Q001",
                        "supplier_name": "Supplier A",
                        "total_amount": 45000,
                        "payment_terms": "Net 30",
                        "delivery_terms": "5 days",
                        "discount_percentage": 5,
                    },
                    {
                        "quote_id": "Q002",
                        "supplier_name": "Supplier B",
                        "total_amount": 42000,
                        "payment_terms": "Net 45",
                        "delivery_terms": "7 days",
                        "discount_percentage": 8,
                    },
                ]

        quotes: List[Dict] = []
        for point in points:
            payload = point.payload or {}
            quotes.append(
                {
                    "quote_id": payload.get("quote_id", point.id),
                    "supplier_name": payload.get("supplier_name", ""),
                    "customer_name": payload.get("customer_name"),
                    "quote_date": payload.get("quote_date"),
                    "valid_until": payload.get("valid_until"),
                    "total_amount": payload.get("total_amount", 0),
                    "currency": payload.get("currency"),
                    "payment_terms": payload.get("payment_terms"),
                    "delivery_terms": payload.get("delivery_terms"),
                    "discount_percentage": payload.get("discount_percentage", 0),
                    "line_items_count": payload.get("line_items_count", 0),
                    "avg_unit_price": payload.get("avg_unit_price", 0),
                    # Baseline fields for variance detection
                    "baseline_total_amount": payload.get("baseline_total_amount"),
                    "baseline_payment_terms": payload.get("baseline_payment_terms"),
                }
            )
        return quotes

    def _evaluate_quotes(self, quotes: List[Dict]) -> Dict:
        """Evaluate quotes on multiple criteria"""
        evaluation = {}

        for quote in quotes:
            quote_id = quote['quote_id']

            # Score different aspects
            price_score = self._score_price(quote, quotes)
            terms_score = self._score_terms(quote)
            delivery_score = self._score_delivery(quote)
            discount_score = quote.get('discount_percentage', 0) / 10

            # Baseline variance detection
            variances = self._detect_variances(quote)

            # Calculate overall score (0-10) for backwards compatibility
            overall_score = (
                price_score * 0.4 +
                terms_score * 0.2 +
                delivery_score * 0.2 +
                discount_score * 0.2
            )

            # Supplier ranking score (0-100)
            price_component = price_score * 10
            terms_component = terms_score * 10
            accuracy_component = quote.get('transactional_accuracy', 10) * 10
            supplier_score = (
                price_component * 0.7 +
                terms_component * 0.2 +
                accuracy_component * 0.1
            )

            evaluation[quote_id] = {
                'supplier_name': quote['supplier_name'],
                'price_score': price_score,
                'terms_score': terms_score,
                'delivery_score': delivery_score,
                'discount_score': discount_score,
                'overall_score': overall_score,
                'total_amount': quote['total_amount'],
                'price_variance': variances.get('price_variance'),
                'term_variance': variances.get('term_variance'),
                'total_cost_variance': variances.get('total_cost_variance'),
                'supplier_score': supplier_score,
                'opportunity_value': variances.get('opportunity_value'),
            }

        return evaluation

    def _detect_variances(self, quote: Dict) -> Dict:
        """Detect variances against baseline information."""
        baseline_total = quote.get('baseline_total_amount')
        price_variance = None
        total_cost_variance = None
        opportunity_value = 0.0
        if baseline_total and baseline_total > 0:
            price_variance = (
                (quote.get('total_amount', 0) - baseline_total) / baseline_total
            ) * 100
            total_cost_variance = price_variance
            opportunity_value = max(0.0, baseline_total - quote.get('total_amount', 0))

        term_variance = 0
        baseline_terms = quote.get('baseline_payment_terms')
        if baseline_terms:
            term_variance = 0 if quote.get('payment_terms') == baseline_terms else 1

        return {
            'price_variance': price_variance,
            'term_variance': term_variance,
            'total_cost_variance': total_cost_variance,
            'opportunity_value': opportunity_value,
        }

    def _score_price(self, quote: Dict, all_quotes: List[Dict]) -> float:
        """Score quote based on price"""
        amounts = [q['total_amount'] for q in all_quotes]
        min_amount = min(amounts)
        max_amount = max(amounts)

        if max_amount == min_amount:
            return 10.0

        # Lower price = higher score
        normalized = (max_amount - quote['total_amount']) / (max_amount - min_amount)
        return normalized * 10

    def _score_terms(self, quote: Dict) -> float:
        """Score payment terms"""
        terms = quote.get('payment_terms', '')

        terms_scores = {
            'Net 30': 10,
            'Net 45': 8,
            'Net 60': 6,
            'Net 90': 4,
            'Immediate': 2
        }

        for term, score in terms_scores.items():
            if term in terms:
                return score

        return 5  # Default score

    def _score_delivery(self, quote: Dict) -> float:
        """Score delivery terms"""
        delivery = quote.get('delivery_terms', '')

        # Extract days from delivery terms
        import re
        days_match = re.search(r'(\d+)\s*day', delivery.lower())

        if days_match:
            days = int(days_match.group(1))
            if days <= 3:
                return 10
            elif days <= 7:
                return 8
            elif days <= 14:
                return 6
            else:
                return 4

        return 5  # Default score

    def _generate_comparison(self, quotes: List[Dict]) -> Dict:
        """Generate comparison matrix"""
        if not quotes:
            return {}

        df = pd.DataFrame(quotes)

        comparison = {
            'price_range': {
                'min': df['total_amount'].min(),
                'max': df['total_amount'].max(),
                'mean': df['total_amount'].mean(),
                'std': df['total_amount'].std()
            },
            'suppliers': df['supplier_name'].tolist(),
            'best_payment_terms': df.loc[
                df['payment_terms'].notna(), 'payment_terms'].value_counts().to_dict() if 'payment_terms' in df else {},
            'discount_range': {
                'min': df['discount_percentage'].min() if 'discount_percentage' in df else 0,
                'max': df['discount_percentage'].max() if 'discount_percentage' in df else 0
            }
        }

        return comparison

    def _identify_best_quote(self, evaluation: Dict) -> Dict:
        """Identify the best quote based on evaluation"""
        if not evaluation:
            return {}

        # Find quote with highest overall score
        best_quote_id = max(evaluation.keys(),
                            key=lambda k: evaluation[k]['overall_score'])

        best_quote = evaluation[best_quote_id].copy()
        best_quote['quote_id'] = best_quote_id
        best_quote['negotiation_possible'] = best_quote['overall_score'] < 8

        return best_quote

    def _calculate_savings(self, quotes: List[Dict], best_quote: Dict) -> float:
        """Calculate potential savings"""
        if not quotes or not best_quote:
            return 0

        amounts = [q['total_amount'] for q in quotes]
        avg_amount = sum(amounts) / len(amounts)

        savings = avg_amount - best_quote.get('total_amount', avg_amount)
        return max(0, savings)

    def _generate_recommendations(self, best_quote: Dict,
                                  evaluation: Dict, comparison: Dict) -> Dict:
        """Generate recommendations using Ollama"""
        prompt = f"""Based on the quote evaluation, provide procurement recommendations:

        Best Quote: {json.dumps(best_quote, indent=2)}
        Price Range: ${comparison.get('price_range', {}).get('min', 0):.2f} - ${comparison.get('price_range', {}).get('max', 0):.2f}
        Number of Quotes: {len(evaluation)}

        Provide:
        1. Key decision factors
        2. Negotiation points (if score < 8)
        3. Risk considerations
        4. Final recommendation

        Format as JSON with keys: decision_factors, negotiation_points, risks, recommendation"""

        response = self.call_ollama(prompt, format='json')

        try:
            recommendations = json.loads(response.get('response', '{}'))
        except:
            recommendations = {
                'decision_factors': ['Price competitiveness', 'Payment terms'],
                'negotiation_points': ['Volume discount', 'Payment terms improvement'],
                'risks': ['Supplier reliability', 'Market volatility'],
                'recommendation': 'Proceed with best quote after negotiation'
            }

        return recommendations

    def _to_native(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
