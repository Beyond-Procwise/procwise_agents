import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for managing agent workflows"""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.agents = agent_nick.agents
        self.policy_engine = agent_nick.policy_engine
        self.query_engine = agent_nick.query_engine
        self.routing_engine = agent_nick.routing_engine
        self.routing_model = self.routing_engine.routing_model
        self.executor = ThreadPoolExecutor(max_workers=self.settings.max_workers)

    def execute_ranking_flow(self, query: str) -> Dict:
        """Public wrapper for the supplier ranking workflow."""
        return self.execute_workflow("supplier_ranking", {"query": query})

    def execute_ranking_workflow(self, query: str) -> Dict:
        """Backward compatible alias for :meth:`execute_ranking_flow`."""
        return self.execute_ranking_flow(query)

    def execute_extraction_flow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
    ) -> Dict:
        """Public wrapper for the document extraction workflow."""
        return self.execute_workflow(
            "document_extraction",
            {"s3_prefix": s3_prefix, "s3_object_key": s3_object_key},
        )

    def execute_extraction_workflow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
    ) -> Dict:
        """Backward compatible alias for :meth:`execute_extraction_flow`."""
        return self.execute_extraction_flow(s3_prefix, s3_object_key)

    def execute_workflow(self, workflow_name: str, input_data: Dict,
                         user_id: str = None) -> Dict:
        """Execute a complete workflow"""
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting workflow {workflow_name} with ID {workflow_id}")

        try:
            # Create initial context
            context = AgentContext(
                workflow_id=workflow_id,
                agent_id=workflow_name,
                user_id=user_id or self.settings.script_user,
                input_data=input_data
            )

            # Validate against policies
            if not self._validate_workflow(workflow_name, context):
                return {
                    'status': 'blocked',
                    'reason': 'Policy validation failed',
                    'workflow_id': workflow_id
                }

            # Execute workflow based on type
            if workflow_name == 'document_extraction':
                result = self._execute_extraction_workflow(context)
            elif workflow_name == 'supplier_ranking':
                result = self._execute_ranking_workflow(context)
            elif workflow_name == 'quote_evaluation':
                result = self._execute_quote_workflow(context)
            elif workflow_name == 'opportunity_mining':
                result = self._execute_opportunity_workflow(context)
            else:
                result = self._execute_generic_workflow(workflow_name, context)

            return {
                'status': 'completed',
                'workflow_id': workflow_id,
                'result': result,
                'execution_path': context.routing_history
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                'status': 'failed',
                'workflow_id': workflow_id,
                'error': str(e)
            }

    def _validate_workflow(self, workflow_name: str, context: AgentContext) -> bool:
        """Validate workflow against policies"""
        if workflow_name == 'supplier_ranking':
            return True
        validation_result = self.policy_engine.validate_workflow(
                    workflow_name,
                    context.user_id,
                    context.input_data)
        return validation_result.get('allowed', True)

    def _execute_extraction_workflow(self, context: AgentContext) -> Dict:
        """Execute document extraction workflow"""
        # Step 1: Extract documents
        extraction_result = self._execute_agent('data_extraction', context)

        if extraction_result.status != AgentStatus.SUCCESS:
            return extraction_result.data

        # Step 2: Process next agents in parallel if enabled
        if self.settings.parallel_processing and extraction_result.next_agents:
            results = self._execute_parallel_agents(
                extraction_result.next_agents,
                context,
                extraction_result.pass_fields
            )
        else:
            results = self._execute_sequential_agents(
                extraction_result.next_agents,
                context,
                extraction_result.pass_fields
            )

        return {
            'extraction': extraction_result.data,
            'downstream_results': results
        }

    def _execute_ranking_workflow(self, context: AgentContext) -> Dict:
        """Execute supplier ranking workflow"""
        # Get supplier data first
        context.input_data['supplier_data'] = self.query_engine.fetch_supplier_data(
            context.input_data
        )

        # Execute ranking
        ranking_result = self._execute_agent('supplier_ranking', context)

        # Process opportunities if found
        if 'OpportunityMinerAgent' in ranking_result.next_agents:
            opp_context = self._create_child_context(
                context,
                'opportunity_miner',
                ranking_result.pass_fields
            )
            opp_result = self._execute_agent('opportunity_miner', opp_context)

            return {
                'ranking': ranking_result.data,
                'opportunities': opp_result.data if opp_result else None
            }

        return ranking_result.data

    def _execute_quote_workflow(self, context: AgentContext) -> Dict:
        """Execute quote evaluation workflow"""
        # Execute quote evaluation
        quote_result = self._execute_agent('quote_evaluation', context)

        # Check if negotiation needed
        if quote_result.data.get('negotiation_required'):
            # Prepare negotiation
            neg_context = self._create_child_context(
                context,
                'negotiation',
                quote_result.pass_fields
            )
            # Would execute negotiation agent here

        return quote_result.data

    def _execute_opportunity_workflow(self, context: AgentContext) -> Dict:
        """Execute opportunity mining workflow"""
        # Execute opportunity mining
        opp_result = self._execute_agent('opportunity_miner', context)

        # Check for discrepancies
        if opp_result.data.get('total_savings_potential', 0) > 10000:
            disc_context = self._create_child_context(
                context,
                'discrepancy_detection',
                opp_result.pass_fields
            )
            disc_result = self._execute_agent('discrepancy_detection', disc_context)

            return {
                'opportunities': opp_result.data,
                'discrepancies': disc_result.data if disc_result else None
            }

        return opp_result.data

    def _execute_generic_workflow(self, workflow_name: str, context: AgentContext) -> Dict:
        """Execute generic workflow based on routing rules"""
        results = {}
        current_agents = [workflow_name]
        depth = 0
        max_depth = self.routing_model.get('global_settings', {}).get('max_chain_depth', 10)

        while current_agents and depth < max_depth:
            next_agents = []

            for agent_name in current_agents:
                if agent_name in self.agents:
                    result = self._execute_agent(agent_name, context)
                    results[agent_name] = result.data

                    if result.next_agents:
                        next_agents.extend(result.next_agents)

                    # Update context with pass fields
                    if result.pass_fields:
                        context.input_data.update(result.pass_fields)

            current_agents = next_agents
            depth += 1

        return results

    def _execute_agent(self, agent_name: str, context: AgentContext) -> Any:
        """Execute single agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return None

        return agent.execute(context)

    def _execute_parallel_agents(self, agents: List[str], context: AgentContext,
                                 pass_fields: Dict) -> Dict:
        """Execute agents in parallel"""
        results = {}
        futures = {}

        for agent_name in agents:
            if agent_name in self.agents:
                child_context = self._create_child_context(context, agent_name, pass_fields)
                future = self.executor.submit(self._execute_agent, agent_name, child_context)
                futures[future] = agent_name

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result(timeout=30)
                results[agent_name] = result.data if result else None
            except Exception as e:
                logger.error(f"Parallel execution failed for {agent_name}: {e}")
                results[agent_name] = {'error': str(e)}

        return results

    def _execute_sequential_agents(self, agents: List[str], context: AgentContext,
                                   pass_fields: Dict) -> Dict:
        """Execute agents sequentially"""
        results = {}

        for agent_name in agents:
            if agent_name in self.agents:
                child_context = self._create_child_context(context, agent_name, pass_fields)
                result = self._execute_agent(agent_name, child_context)
                results[agent_name] = result.data if result else None

                # Update pass fields for next agent
                if result and result.pass_fields:
                    pass_fields.update(result.pass_fields)

        return results

    def _create_child_context(self, parent_context: AgentContext,
                              agent_name: str, pass_fields: Dict) -> AgentContext:
        """Create child context for agent execution"""
        return AgentContext(
            workflow_id=parent_context.workflow_id,
            agent_id=agent_name,
            user_id=parent_context.user_id,
            input_data={**parent_context.input_data, **pass_fields},
            parent_agent=parent_context.agent_id,
            routing_history=parent_context.routing_history.copy()
        )
