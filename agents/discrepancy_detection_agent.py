import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)


class DiscrepancyDetectionAgent(BaseAgent):
    """Agent for detecting discrepancies in procurement data"""

    def process(self, context: AgentContext) -> AgentOutput:
        """Process discrepancy detection"""
        try:
            input_data = context.input_data

            # Detect various types of discrepancies
            invoice_discrepancies = self._detect_invoice_discrepancies()
            contract_discrepancies = self._detect_contract_discrepancies()
            price_discrepancies = self._detect_price_discrepancies()

            # Combine all discrepancies
            all_discrepancies = {
                'invoice': invoice_discrepancies,
                'contract': contract_discrepancies,
                'price': price_discrepancies
            }

            # Calculate impact
            total_impact = self._calculate_financial_impact(all_discrepancies)

            # Generate resolution plan
            resolution_plan = self._generate_resolution_plan(all_discrepancies)

            output_data = {
                'discrepancies_found': sum(len(d) for d in all_discrepancies.values()),
                'financial_impact': total_impact,
                'discrepancies': all_discrepancies,
                'resolution_plan': resolution_plan,
                'priority_level': self._determine_priority(total_impact)
            }

            # Determine next agents
            next_agents = []
            if total_impact > 10000:
                next_agents.append('HumanEscalationAgent')
            if invoice_discrepancies:
                next_agents.append('InvoiceReconciliationAgent')
            if contract_discrepancies:
                next_agents.append('ContractIntelligenceAgent')

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                next_agents=next_agents,
                pass_fields={
                    'critical_discrepancies': self._get_critical_discrepancies(all_discrepancies),
                    'affected_suppliers': self._get_affected_suppliers(all_discrepancies),
                    'resolution_priority': output_data['priority_level']
                },
                confidence=0.9
            )

        except Exception as e:
            logger.error(f"DiscrepancyDetectionAgent error: {e}")
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(e)
            )

    def _detect_invoice_discrepancies(self) -> List[Dict]:
        """Detect invoice vs PO discrepancies"""
        discrepancies = []

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            i.invoice_id,
                            i.vendor_name,
                            i.total_amount as invoice_amount,
                            p.total_amount as po_amount,
                            ABS(i.total_amount - p.total_amount) as difference,
                            i.invoice_date
                        FROM proc.invoice_agent i
                        LEFT JOIN proc.purchase_order_agent p 
                            ON i.vendor_name = p.vendor_name
                            AND ABS(EXTRACT(day FROM i.invoice_date - p.order_date)) < 30
                        WHERE ABS(i.total_amount - p.total_amount) > 100
                        AND i.invoice_date >= CURRENT_DATE - INTERVAL '3 months'
                    """)

                    for row in cursor.fetchall():
                        discrepancies.append({
                            'type': 'invoice_po_mismatch',
                            'invoice_id': row[0],
                            'vendor': row[1],
                            'invoice_amount': float(row[2]) if row[2] else 0,
                            'po_amount': float(row[3]) if row[3] else 0,
                            'difference': float(row[4]) if row[4] else 0,
                            'date': row[5].strftime('%Y-%m-%d') if row[5] else None
                        })

        except Exception as e:
            logger.error(f"Failed to detect invoice discrepancies: {e}")

        return discrepancies

    def _detect_contract_discrepancies(self) -> List[Dict]:
        """Detect contract vs actual pricing discrepancies"""
        discrepancies = []

        # Use Ollama to analyze contract terms
        prompt = """Identify potential contract discrepancies based on these patterns:
        1. Payment terms not matching agreed terms
        2. Pricing above contracted rates
        3. Missing volume discounts
        4. Unauthorized charges

        Return potential issues as JSON array with: type, description, estimated_impact"""

        response = self.call_ollama(prompt, format='json')

        try:
            ml_discrepancies = json.loads(response.get('response', '[]'))
            for disc in ml_discrepancies:
                if isinstance(disc, dict):
                    discrepancies.append({
                        'type': 'contract_violation',
                        'description': disc.get('description', ''),
                        'impact': disc.get('estimated_impact', 0)
                    })
        except:
            pass

        return discrepancies

    def _detect_price_discrepancies(self) -> List[Dict]:
        """Detect price anomalies and variations"""
        discrepancies = []

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Find price variations for same items
                    cursor.execute("""
                        SELECT 
                            il.description,
                            i.vendor_name,
                            AVG(il.unit_price) as avg_price,
                            MIN(il.unit_price) as min_price,
                            MAX(il.unit_price) as max_price,
                            COUNT(*) as purchase_count
                        FROM proc.invoice_line_items_agent il
                        JOIN proc.invoice_agent i ON il.invoice_id = i.invoice_id
                        WHERE il.description IS NOT NULL
                        GROUP BY il.description, i.vendor_name
                        HAVING COUNT(*) > 2 
                        AND (MAX(il.unit_price) - MIN(il.unit_price)) > AVG(il.unit_price) * 0.1
                    """)

                    for row in cursor.fetchall():
                        discrepancies.append({
                            'type': 'price_variation',
                            'item': row[0],
                            'vendor': row[1],
                            'avg_price': float(row[2]) if row[2] else 0,
                            'min_price': float(row[3]) if row[3] else 0,
                            'max_price': float(row[4]) if row[4] else 0,
                            'occurrences': row[5],
                            'potential_savings': (float(row[2]) - float(row[3])) * row[5] if row[2] and row[3] else 0
                        })

        except Exception as e:
            logger.error(f"Failed to detect price discrepancies: {e}")

        return discrepancies

    def _calculate_financial_impact(self, discrepancies: Dict) -> float:
        """Calculate total financial impact"""
        total_impact = 0

        for disc_type, disc_list in discrepancies.items():
            for disc in disc_list:
                if 'difference' in disc:
                    total_impact += abs(disc['difference'])
                elif 'potential_savings' in disc:
                    total_impact += disc['potential_savings']
                elif 'impact' in disc:
                    total_impact += disc['impact']

        return total_impact

    def _generate_resolution_plan(self, discrepancies: Dict) -> Dict:
        """Generate resolution plan using Ollama"""
        prompt = f"""Create a resolution plan for these procurement discrepancies:

        Discrepancies: {json.dumps(discrepancies, indent=2, default=str)}

        Provide:
        1. Immediate actions (within 24 hours)
        2. Supplier engagement strategy
        3. Process improvements
        4. Prevention measures

        Format as JSON with keys: immediate_actions, supplier_strategy, process_improvements, prevention_measures"""

        response = self.call_ollama(prompt, format='json')

        try:
            return json.loads(response.get('response', '{}'))
        except:
            return {
                'immediate_actions': ['Review critical discrepancies', 'Contact affected suppliers'],
                'supplier_strategy': 'Engage suppliers for clarification and correction',
                'process_improvements': ['Enhance validation rules', 'Automate matching'],
                'prevention_measures': ['Regular audits', 'Supplier scorecards']
            }

    def _determine_priority(self, impact: float) -> str:
        """Determine priority level based on impact"""
        if impact > 50000:
            return 'CRITICAL'
        elif impact > 10000:
            return 'HIGH'
        elif impact > 1000:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_critical_discrepancies(self, discrepancies: Dict) -> List[Dict]:
        """Extract critical discrepancies"""
        critical = []

        for disc_type, disc_list in discrepancies.items():
            for disc in disc_list:
                impact = disc.get('difference', disc.get('potential_savings', disc.get('impact', 0)))
                if impact > 5000:
                    critical.append({
                        'type': disc_type,
                        'details': disc,
                        'impact': impact
                    })

        return sorted(critical, key=lambda x: x['impact'], reverse=True)[:5]

    def _get_affected_suppliers(self, discrepancies: Dict) -> List[str]:
        """Get list of affected suppliers"""
        suppliers = set()

        for disc_type, disc_list in discrepancies.items():
            for disc in disc_list:
                if 'vendor' in disc:
                    suppliers.add(disc['vendor'])
                elif 'supplier' in disc:
                    suppliers.add(disc['supplier'])

        return list(suppliers)