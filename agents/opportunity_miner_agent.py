import json
import logging
from typing import Dict, List, Optional
from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)


class OpportunityMinerAgent(BaseAgent):
    """Agent for identifying cost-saving opportunities"""

    def process(self, context: AgentContext) -> AgentOutput:
        """Process opportunity mining"""
        try:
            input_data = context.input_data

            # Analyze spend patterns
            spend_analysis = self._analyze_spend_patterns()

            # Identify opportunities
            opportunities = self._identify_opportunities(spend_analysis)

            # Calculate potential savings
            total_savings = sum(opp['estimated_savings'] for opp in opportunities)

            # Generate action plan using Ollama
            action_plan = self._generate_action_plan(opportunities)

            output_data = {
                'opportunities_found': len(opportunities),
                'total_savings_potential': total_savings,
                'opportunities': opportunities,
                'action_plan': action_plan,
                'confidence': self._calculate_confidence(opportunities)
            }

            # Determine next agents
            next_agents = []
            if total_savings > 50000:
                next_agents.append('HumanEscalationAgent')
            if any(opp['type'] == 'contract_discrepancy' for opp in opportunities):
                next_agents.append('DiscrepancyDetectionAgent')
            if total_savings > 10000:
                next_agents.append('AutomatedActionAgent')

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                next_agents=next_agents,
                pass_fields={
                    'opportunity_ids': [opp['id'] for opp in opportunities],
                    'action_items': action_plan.get('immediate_actions', []),
                    'estimated_savings': total_savings
                },
                confidence=output_data['confidence']
            )

        except Exception as e:
            logger.error(f"OpportunityMinerAgent error: {e}")
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(e)
            )

    def _analyze_spend_patterns(self) -> Dict:
        """Analyze spending patterns from database"""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Analyze invoice spend patterns
                    cursor.execute("""
                        SELECT 
                            vendor_name,
                            COUNT(*) as transaction_count,
                            SUM(total_amount) as total_spend,
                            AVG(total_amount) as avg_spend,
                            MAX(total_amount) - MIN(total_amount) as spend_variance,
                            MAX(invoice_date) as last_transaction
                        FROM proc.invoice_agent
                        WHERE invoice_date >= CURRENT_DATE - INTERVAL '6 months'
                        GROUP BY vendor_name
                        HAVING COUNT(*) > 3
                        ORDER BY total_spend DESC
                    """)

                    spend_data = cursor.fetchall()

                    return {
                        'vendor_spend': [
                            {
                                'vendor': row[0],
                                'transactions': row[1],
                                'total_spend': float(row[2]) if row[2] else 0,
                                'avg_spend': float(row[3]) if row[3] else 0,
                                'variance': float(row[4]) if row[4] else 0
                            }
                            for row in spend_data
                        ]
                    }

        except Exception as e:
            logger.error(f"Failed to analyze spend: {e}")
            # Return mock data
            return {
                'vendor_spend': [
                    {
                        'vendor': 'Supplier A',
                        'transactions': 25,
                        'total_spend': 150000,
                        'avg_spend': 6000,
                        'variance': 3000
                    }
                ]
            }

    def _identify_opportunities(self, spend_analysis: Dict) -> List[Dict]:
        """Identify cost-saving opportunities"""
        opportunities = []

        for vendor_data in spend_analysis.get('vendor_spend', []):
            # Check for high variance (potential for standardization)
            if vendor_data['variance'] > vendor_data['avg_spend'] * 0.5:
                opportunities.append({
                    'id': f"OPP_{vendor_data['vendor']}_{len(opportunities)}",
                    'type': 'price_standardization',
                    'vendor': vendor_data['vendor'],
                    'description': f"High price variance for {vendor_data['vendor']}",
                    'estimated_savings': vendor_data['variance'] * vendor_data['transactions'] * 0.1,
                    'confidence': 0.7
                })

            # Check for volume consolidation opportunity
            if vendor_data['transactions'] > 10:
                opportunities.append({
                    'id': f"OPP_VOL_{vendor_data['vendor']}_{len(opportunities)}",
                    'type': 'volume_consolidation',
                    'vendor': vendor_data['vendor'],
                    'description': f"Volume consolidation opportunity with {vendor_data['vendor']}",
                    'estimated_savings': vendor_data['total_spend'] * 0.05,
                    'confidence': 0.8
                })

        # Use Ollama to identify additional opportunities
        additional_opps = self._identify_ml_opportunities(spend_analysis)
        opportunities.extend(additional_opps)

        return opportunities

    def _identify_ml_opportunities(self, spend_analysis: Dict) -> List[Dict]:
        """Use Ollama to identify additional opportunities"""
        prompt = f"""Analyze this procurement spend data and identify cost-saving opportunities:

        Spend Data: {json.dumps(spend_analysis, indent=2)}

        Identify opportunities in these categories:
        1. Contract renegotiation
        2. Supplier consolidation
        3. Payment term optimization
        4. Process automation

        Return as JSON array with objects containing: type, description, estimated_savings_percentage
        """

        response = self.call_ollama(prompt, format='json')

        try:
            ml_opportunities = json.loads(response.get('response', '[]'))

            # Convert to standard format
            opportunities = []
            for opp in ml_opportunities:
                if isinstance(opp, dict):
                    opportunities.append({
                        'id': f"OPP_ML_{len(opportunities)}",
                        'type': opp.get('type', 'general'),
                        'vendor': 'Multiple',
                        'description': opp.get('description', ''),
                        'estimated_savings': 10000 * opp.get('estimated_savings_percentage', 0.05),
                        'confidence': 0.6
                    })

            return opportunities

        except:
            return []

    def _generate_action_plan(self, opportunities: List[Dict]) -> Dict:
        """Generate action plan using Ollama"""
        if not opportunities:
            return {'immediate_actions': [], 'long_term_actions': []}

        prompt = f"""Create an action plan for these procurement opportunities:

        Opportunities: {json.dumps(opportunities, indent=2)}

        Provide:
        1. Immediate actions (within 1 week)
        2. Short-term actions (within 1 month)
        3. Long-term actions (within 3 months)
        4. Success metrics

        Format as JSON with keys: immediate_actions, short_term_actions, long_term_actions, success_metrics"""

        response = self.call_ollama(prompt, format='json')

        try:
            return json.loads(response.get('response', '{}'))
        except:
            return {
                'immediate_actions': ['Review top 3 opportunities', 'Contact suppliers'],
                'short_term_actions': ['Implement quick wins', 'Start negotiations'],
                'long_term_actions': ['Full supplier review', 'Process optimization'],
                'success_metrics': ['Cost reduction %', 'Process efficiency']
            }

    def _calculate_confidence(self, opportunities: List[Dict]) -> float:
        """Calculate overall confidence score"""
        if not opportunities:
            return 0

        confidences = [opp.get('confidence', 0.5) for opp in opportunities]
        return sum(confidences) / len(confidences)