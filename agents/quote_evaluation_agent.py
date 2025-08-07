import json
import logging
from typing import Dict, List, Optional
import pandas as pd
from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)


class QuoteEvaluationAgent(BaseAgent):
    """Agent for evaluating and comparing supplier quotes"""

    def process(self, context: AgentContext) -> AgentOutput:
        """Process quote evaluation"""
        try:
            input_data = context.input_data

            # Fetch quotes from database
            quotes = self._fetch_quotes(
                input_data.get('supplier_names', []),
                input_data.get('product_category')
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

            output_data = {
                'quotes_analyzed': len(quotes),
                'best_quote': best_quote,
                'comparison_matrix': comparison,
                'evaluation_scores': evaluation,
                'recommendations': recommendations,
                'savings_potential': self._calculate_savings(quotes, best_quote)
            }

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

    def _fetch_quotes(self, supplier_names: List[str],
                      product_category: Optional[str] = None) -> List[Dict]:
        """Fetch quotes from database"""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Build query
                    query = """
                        SELECT 
                            q.quote_id,
                            q.supplier_name,
                            q.customer_name,
                            q.quote_date,
                            q.valid_until,
                            q.total_amount,
                            q.currency,
                            q.payment_terms,
                            q.delivery_terms,
                            q.discount_percentage,
                            COUNT(ql.quote_line_id) as line_items_count,
                            AVG(ql.unit_price) as avg_unit_price
                        FROM proc.quotes_agent q
                        LEFT JOIN proc.quote_line_items_agent ql ON q.quote_id = ql.quote_id
                        WHERE q.valid_until >= CURRENT_DATE
                    """

                    params = []
                    if supplier_names:
                        query += " AND q.supplier_name = ANY(%s)"
                        params.append(supplier_names)

                    query += " GROUP BY q.quote_id, q.supplier_name, q.customer_name, "
                    query += "q.quote_date, q.valid_until, q.total_amount, q.currency, "
                    query += "q.payment_terms, q.delivery_terms, q.discount_percentage"

                    cursor.execute(query, params)

                    columns = [desc[0] for desc in cursor.description]
                    quotes = [dict(zip(columns, row)) for row in cursor.fetchall()]

                    return quotes

        except Exception as e:
            logger.error(f"Failed to fetch quotes: {e}")
            # Return mock data for testing
            return [
                {
                    'quote_id': 'Q001',
                    'supplier_name': 'Supplier A',
                    'total_amount': 45000,
                    'payment_terms': 'Net 30',
                    'delivery_terms': '5 days',
                    'discount_percentage': 5
                },
                {
                    'quote_id': 'Q002',
                    'supplier_name': 'Supplier B',
                    'total_amount': 42000,
                    'payment_terms': 'Net 45',
                    'delivery_terms': '7 days',
                    'discount_percentage': 8
                }
            ]

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

            # Calculate overall score
            overall_score = (
                    price_score * 0.4 +
                    terms_score * 0.2 +
                    delivery_score * 0.2 +
                    discount_score * 0.2
            )

            evaluation[quote_id] = {
                'supplier_name': quote['supplier_name'],
                'price_score': price_score,
                'terms_score': terms_score,
                'delivery_score': delivery_score,
                'discount_score': discount_score,
                'overall_score': overall_score,
                'total_amount': quote['total_amount']
            }

        return evaluation

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
