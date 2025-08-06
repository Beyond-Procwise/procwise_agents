# ProcWise/orchestration/orchestrator.py

from orchestration.prompt_engine import PromptEngine  # <-- CORRECTED IMPORT
from engines.policy_engine import PolicyEngine        # <-- CORRECTED IMPORT
from engines.query_engine import QueryEngine


# The orchestrator doesn't need to import the agent classes directly,
# as it will receive the initialized instances from AgentNick.

class Orchestrator:
    def __init__(self, agent_nick):
        """
        Initializes the orchestrator with access to core engines and initialized agents.
        """
        self.agent_nick = agent_nick
        self.prompt_engine = agent_nick.prompt_engine
        self.policy_engine = agent_nick.policy_engine
        self.query_engine = agent_nick.query_engine

        # Access the pre-initialized agents from the agent_nick instance
        self.supplier_ranking_agent = agent_nick.agents.get('supplier_ranking')
        self.data_extraction_agent = agent_nick.agents.get('data_extraction')
        print("Orchestrator initialized and has access to all required agents.")

    def execute_ranking_flow(self, query: str) -> dict:
        """Executes the full supplier ranking workflow."""
        print("Orchestrator: Executing ranking flow...")
        # 1. Deconstruct query into a structured intent
        intent = self.prompt_engine.deconstruct_query(
            query,
            self.supplier_ranking_agent.prompt_library
        )
        if not intent: return {"error": "Orchestrator: Failed to understand the query."}

        # 2. Validate intent against business policies
        is_allowed, reason, intent = self.policy_engine.validate_and_apply(intent)
        if not is_allowed: return {"error": f"Orchestrator: {reason}"}

        # 3. Fetch data from the source
        data = self.query_engine.fetch_supplier_data(intent)
        if data.empty: return {"error": "Orchestrator: No data found."}

        # 4. Execute the agent's core logic with the prepared data
        ranking = self.supplier_ranking_agent.run(intent, data)
        return ranking

    def execute_extraction_flow(self, s3_prefix: str = None, s3_object_key: str = None) -> dict:
        """
        Executes the data extraction workflow by triggering the DataExtractionAgent.
        """
        print("Orchestrator: Executing data extraction flow...")
        if not self.data_extraction_agent:
            return {"error": "Orchestrator: DataExtractionAgent is not available."}

        # Call the run method of the dedicated agent
        result = self.data_extraction_agent.run(
            s3_prefix=s3_prefix,
            s3_object_key=s3_object_key
        )
        return result
