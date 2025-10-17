"""Workflow orchestrator enforcing hidden-identifier based supplier gating."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Set

from email_thread import EmailThread, NegotiationSession
from workflow_context_manager import WorkflowContextManager
from email_drafting_agent import EmailDraftingAgent
from negotiation_agent import NegotiationAgent
from quote_evaluation_agent import QuoteEvaluationAgent
from supplier_interaction_agent import SupplierInteractionAgent


class WorkflowOrchestrator:
    """Coordinates procurement workflow phases while owning all timing controls."""

    def __init__(self, wait_timeout: int = 1800):
        self.session_id = str(uuid.uuid4())
        timeout_env = int(os.getenv("RESPONSE_GATE_TIMEOUT_SEC", str(wait_timeout)))
        self.response_gate_timeout = max(timeout_env, 1)
        poll_env = int(os.getenv("RESPONSE_GATE_POLL_INTERVAL_SEC", "5"))
        self.poll_interval = max(poll_env, 1)
        self.strict_gate = os.getenv("STRICT_GATE", "true").lower() != "false"
        self.parallelism_override = os.getenv("NEGOTIATION_PARALLELISM")

        self.context_manager = WorkflowContextManager()
        self.interaction_agent = SupplierInteractionAgent()
        self.active_threads: Dict[str, EmailThread] = {}
        self.session = NegotiationSession(
            session_id=self.session_id,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

    def create_thread_for_supplier(self, supplier_id: str) -> EmailThread:
        thread = EmailThread(supplier_id=supplier_id)
        self.active_threads[supplier_id] = thread
        self.session.supplier_states[supplier_id] = {
            "thread_id": thread.thread_id,
            "supplier_unique_id": thread.supplier_unique_id,
            "hidden_identifier": thread.hidden_identifier,
            "rounds_completed": 0,
            "responses_received": 0,
        }
        return thread

    async def execute_workflow(self, suppliers: List[Dict]) -> Dict:
        """Execute the procurement workflow end-to-end."""

        print(f"🚀 Starting Workflow - Session: {self.session_id}\n")

        print("🧠 Phase 0: Initializing context...")
        await self.context_manager.initialize_workflow_understanding()

        print("\n📧 Phase 1: Generating initial emails...")
        initial_emails = await self._phase_initial_emails(suppliers)

        expected_hidden_ids = [
            self.active_threads[supplier["id"]].hidden_identifier for supplier in suppliers
        ]

        print("\n📥 Phase 2: Waiting for initial responses...")
        await self._await_all_supplier_replies(expected_hidden_ids)
        initial_responses = await self._collect_supplier_responses(
            suppliers, round_num=0
        )
        for supplier_id, response in initial_responses.items():
            key = f"{supplier_id}_supplier_response_round_0"
            initial_emails[key] = response

        print("\n💼 Phase 3: Negotiation (3 rounds)...")
        negotiation_results = await self._phase_negotiation(suppliers, initial_responses)

        print("\n📊 Phase 4: Quote evaluation...")
        final_evaluation = await self._phase_quote_evaluation(negotiation_results)

        print(f"\n✅ Workflow Complete - Session: {self.session_id}")
        return final_evaluation

    async def _phase_initial_emails(self, suppliers: List[Dict]) -> Dict[str, Dict]:
        email_agent = EmailDraftingAgent(self.context_manager)
        drafts: Dict[str, Dict] = {}

        for supplier in suppliers:
            thread = self.create_thread_for_supplier(supplier["id"])
            email = await email_agent.generate_initial_email(
                supplier,
                thread.hidden_identifier,
                supplier_unique_id=thread.supplier_unique_id,
                thread_id=thread.thread_id,
            )
            thread.add_message(
                "initial_email",
                email["content"],
                email["action_id"],
                round_num=0,
                headers={"X-Proc-Hidden-Identifier": thread.hidden_identifier},
            )
            drafts[supplier["id"]] = email
        return drafts

    async def _await_all_supplier_replies(
        self, expected_hidden_ids: Iterable[str]
    ) -> Set[str]:
        expected: Set[str] = {hid for hid in expected_hidden_ids if hid}
        if not expected:
            return set()

        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.response_gate_timeout)
        while True:
            status = await self.interaction_agent.refresh_received_from_db(expected)
            received = status["received"]
            missing = status["missing"]
            print(
                "[GATE] session=%s expected=%d received=%d missing=%s deadline=%s"
                % (
                    self.session_id,
                    len(expected),
                    len(received),
                    sorted(missing),
                    deadline.isoformat(),
                )
            )
            if not missing:
                return received

            if datetime.now(timezone.utc) >= deadline:
                message = (
                    f"Supplier responses missing for session {self.session_id}: "
                    f"{sorted(missing)} after {self.response_gate_timeout}s"
                )
                print(f"❌ {message}")
                if self.strict_gate:
                    raise TimeoutError(message)
                return received

            await asyncio.sleep(self.poll_interval)

    async def _collect_supplier_responses(
        self, suppliers: List[Dict], *, round_num: int
    ) -> Dict[str, Dict]:
        collected: Dict[str, Dict] = {}
        for supplier in suppliers:
            supplier_id = supplier["id"]
            thread = self.active_threads[supplier_id]
            response = await self.interaction_agent.collect_response(
                supplier_id, thread.hidden_identifier
            )
            if not response:
                continue

            supplier_action_id = f"SUPP-R{round_num}-{thread.supplier_unique_id}"
            thread.add_message(
                f"supplier_response_round_{round_num}",
                response.get("content") or "",
                supplier_action_id,
                round_num=round_num,
            )
            self.session.supplier_states[supplier_id]["responses_received"] = round_num
            collected[supplier_id] = response
        return collected

    async def _phase_negotiation(
        self,
        suppliers: List[Dict],
        initial_responses: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        negotiation_agent = NegotiationAgent(self.context_manager)
        results: Dict[str, Dict] = {}
        latest_responses: Dict[str, Dict] = dict(initial_responses)

        for round_num in range(1, 4):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num} - Session: {self.session_id}")
            print(f"{'=' * 70}\n")

            self.session.current_round = round_num
            tasks: List[asyncio.Future] = []
            for supplier in suppliers:
                supplier_id = supplier["id"]
                thread = self.active_threads[supplier_id]
                thread.current_round = round_num
                latest_response = latest_responses.get(supplier_id)
                tasks.append(
                    negotiation_agent.generate_negotiation_response(
                        supplier,
                        thread,
                        round_num,
                        latest_response,
                    )
                )

            parallelism = len(tasks)
            if self.parallelism_override:
                parallelism = min(parallelism, int(self.parallelism_override))
            print(f"🔄 Processing {parallelism} suppliers in parallel...")
            drafts = await asyncio.gather(*tasks)

            for supplier, draft in zip(suppliers, drafts):
                supplier_id = supplier["id"]
                results[f"{supplier_id}_round_{round_num}"] = draft
                self.session.supplier_states[supplier_id]["rounds_completed"] = round_num
                print(f"✓ {supplier_id}: {draft['action_id']}")

            print(f"\n📤 Round {round_num} drafts ready for human review")

            if round_num < 3:
                expected_hidden_ids = [
                    self.active_threads[supplier["id"]].hidden_identifier
                    for supplier in suppliers
                ]
                await self._await_all_supplier_replies(expected_hidden_ids)
                latest_responses = await self._collect_supplier_responses(
                    suppliers, round_num=round_num
                )
                for supplier_id, response in latest_responses.items():
                    results[f"{supplier_id}_supplier_response_round_{round_num}"] = response
                print(f"✅ All Round {round_num} responses collected\n")

        print("✅ All 3 negotiation rounds complete")
        return results

    async def _phase_quote_evaluation(
        self, negotiation_results: Dict[str, Dict]
    ) -> Dict:
        eval_agent = QuoteEvaluationAgent(self.context_manager)
        return await eval_agent.evaluate_quotes(self.active_threads)
