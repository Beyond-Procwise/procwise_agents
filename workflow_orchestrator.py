from datetime import datetime
from typing import List, Dict
import asyncio
import uuid

from email_thread import EmailThread, NegotiationSession
from workflow_context_manager import WorkflowContextManager
from email_drafting_agent import EmailDraftingAgent
from negotiation_agent import NegotiationAgent
from quote_evaluation_agent import QuoteEvaluationAgent


class WorkflowOrchestrator:
    """
    CRITICAL: ONLY this class controls workflow timing and progression
    NO agent has workflow control methods
    """

    def __init__(self, wait_timeout: int = 300):
        self.session_id = str(uuid.uuid4())
        self.wait_timeout = wait_timeout
        self.check_interval = 10
        self.active_threads: Dict[str, EmailThread] = {}
        self.context_manager = WorkflowContextManager()
        self.session = NegotiationSession(
            session_id=self.session_id,
            start_time=datetime.now().isoformat()
        )

    async def wait_for_responses(self, expected_count: int, context: str = "") -> Dict[str, Dict]:
        """EXCLUSIVE orchestrator responsibility - NO agent calls this"""
        responses = {}
        elapsed = 0
        print(f"⏳ Waiting for {expected_count} {context}...")

        while len(responses) < expected_count and elapsed < self.wait_timeout:
            new_responses = await self._check_for_responses()
            responses.update(new_responses)

            if len(responses) >= expected_count:
                print(f"✅ All {expected_count} responses collected")
                break

            await asyncio.sleep(self.check_interval)
            elapsed += self.check_interval

        return responses

    async def _check_for_responses(self) -> Dict[str, Dict]:
        """TODO: Integrate with email API - match by hidden_identifier"""
        return {}

    def create_thread_for_supplier(self, supplier_id: str) -> EmailThread:
        thread = EmailThread(supplier_id=supplier_id)
        self.active_threads[supplier_id] = thread
        self.session.supplier_states[supplier_id] = {
            "thread_id": thread.thread_id,
            "supplier_unique_id": thread.supplier_unique_id,
            "rounds_completed": 0,
            "responses_received": 0
        }
        return thread

    async def execute_workflow(self, suppliers: List[Dict]):
        """Main workflow execution - controls ALL phase transitions"""
        print(f"🚀 Starting Workflow - Session: {self.session_id}\n")

        # PHASE 0: CRITICAL - Initialize context FIRST
        print("🧠 Phase 0: Initializing context...")
        await self.context_manager.initialize_workflow_understanding()

        # PHASE 1: Initial emails
        print("\n📧 Phase 1: Generating initial emails...")
        initial_emails = await self._phase_initial_emails(suppliers)

        # PHASE 2: Wait for ALL initial responses
        print("\n📥 Phase 2: Waiting for initial responses...")
        initial_responses = await self.wait_for_responses(len(suppliers), "initial responses")

        # PHASE 3: 3-round negotiation
        print("\n💼 Phase 3: Negotiation (3 rounds)...")
        negotiation_results = await self._phase_negotiation(suppliers, initial_responses)

        # PHASE 4: Quote evaluation
        print("\n📊 Phase 4: Quote evaluation...")
        final_evaluation = await self._phase_quote_evaluation(negotiation_results)

        print(f"\n✅ Workflow Complete - Session: {self.session_id}")
        return final_evaluation

    async def _phase_initial_emails(self, suppliers: List[Dict]) -> Dict:
        email_agent = EmailDraftingAgent(self.context_manager)
        results = {}

        for supplier in suppliers:
            thread = self.create_thread_for_supplier(supplier['id'])
            email = await email_agent.generate_initial_email(supplier, thread.hidden_identifier)
            thread.add_message("initial_email", email['content'], email['action_id'])
            results[supplier['id']] = email

        return results

    async def _phase_negotiation(self, suppliers: List[Dict], initial_responses: Dict) -> Dict:
        """CRITICAL: 3 rounds with parallel processing and session maintenance"""
        negotiation_agent = NegotiationAgent(self.context_manager)
        results = {}

        for round_num in range(1, 4):
            print(f"\n{'='*70}")
            print(f"ROUND {round_num} - Session: {self.session_id}")
            print(f"{'='*70}\n")

            self.session.current_round = round_num

            # Parallel processing for all suppliers
            round_tasks = []
            for supplier in suppliers:
                thread = self.active_threads[supplier['id']]
                thread.current_round = round_num

                # Get supplier's latest response
                if round_num == 1:
                    latest_response = initial_responses.get(supplier['id'])
                else:
                    latest_response = results.get(f"{supplier['id']}_supplier_response_round_{round_num-1}")

                task = negotiation_agent.generate_negotiation_response(
                    supplier, thread, round_num, latest_response
                )
                round_tasks.append((supplier['id'], task))

            # Execute ALL suppliers in parallel
            print(f"🔄 Processing {len(suppliers)} suppliers in parallel...")
            task_results = await asyncio.gather(*[task for _, task in round_tasks])

            # Store results
            for (supplier_id, _), result in zip(round_tasks, task_results):
                thread = self.active_threads[supplier_id]
                thread.add_message(f"negotiation_round_{round_num}", result['content'], result['action_id'])
                results[f"{supplier_id}_round_{round_num}"] = result
                self.session.supplier_states[supplier_id]['rounds_completed'] = round_num
                print(f"✓ {supplier_id}: {result['action_id']}")

            print(f"\n📤 Round {round_num} drafts ready for human review")

            # Wait for responses before next round
            if round_num < 3:
                print(f"\n⏳ Waiting for Round {round_num} responses...")
                round_responses = await self.wait_for_responses(len(suppliers), f"Round {round_num} responses")

                for supplier_id, response in round_responses.items():
                    thread = self.active_threads[supplier_id]
                    thread.add_message(
                        f"supplier_response_round_{round_num}",
                        response['content'],
                        f"SUPP-R{round_num}-{thread.supplier_unique_id}"
                    )
                    results[f"{supplier_id}_supplier_response_round_{round_num}"] = response
                    self.session.supplier_states[supplier_id]['responses_received'] = round_num

                print(f"✅ All Round {round_num} responses collected\n")

        print(f"✅ All 3 negotiation rounds complete")
        return results

    async def _phase_quote_evaluation(self, negotiation_results: Dict) -> Dict:
        eval_agent = QuoteEvaluationAgent(self.context_manager)
        return await eval_agent.evaluate_quotes(self.active_threads)
