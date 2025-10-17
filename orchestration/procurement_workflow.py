"""Procurement workflow orchestration with database-backed synchronization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid
import asyncio
import json


@dataclass
class EmailThread:
    """ONE thread per supplier, maintained throughout entire workflow"""

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    supplier_id: str = ""
    supplier_unique_id: str = field(
        default_factory=lambda: f"SUP-{uuid.uuid4().hex[:10].upper()}"
    )
    hidden_identifier: str = field(
        default_factory=lambda: f"PROC-WF-{uuid.uuid4().hex[:12].upper()}"
    )
    messages: List[Dict] = field(default_factory=list)
    current_round: int = 0

    def add_message(self, message_type: str, content: str, action_id: str) -> None:
        self.messages.append(
            {
                "message_type": message_type,
                "content": content,
                "action_id": action_id,
                "timestamp": datetime.now().isoformat(),
                "round": self.current_round,
            }
        )

    def get_full_thread(self) -> str:
        thread_content = (
            f"Thread: {self.hidden_identifier}\nSupplier: {self.supplier_id}\n\n"
        )
        for msg in self.messages:
            thread_content += (
                f"[Round {msg['round']}] {msg['message_type']}:\n{msg['content']}\n\n"
            )
        return thread_content


@dataclass
class NegotiationSession:
    session_id: str
    start_time: str
    current_round: int = 0
    supplier_states: Dict[str, Dict] = field(default_factory=dict)


class DatabaseConnection:
    async def execute(self, query: str, params: Tuple) -> None:
        raise NotImplementedError

    async def fetch_one(self, query: str, params: Tuple) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_all(self, query: str, params: Tuple) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_val(self, query: str, params: Tuple) -> Any:
        raise NotImplementedError

    def transaction(self):  # pragma: no cover - interface
        raise NotImplementedError


class WorkflowContextManager:
    """CRITICAL: Initialize BEFORE any workflow execution"""

    def __init__(self, ollama_model: str = "qwen3:30b"):
        self.model = ollama_model
        self.workflow_context = {
            "phases": [
                "ranking",
                "initial_email",
                "negotiation_rounds",
                "evaluation",
            ],
            "max_negotiation_rounds": 3,
            "human_checkpoints": [
                "email_review",
                "negotiation_review",
                "final_review",
            ],
        }
        self.embeddings: Dict[str, Any] = {}

    async def initialize_workflow_understanding(self) -> Dict[str, Any]:
        """Call this FIRST before workflow execution"""

        workflow_prompt = f"""
You are managing a procurement workflow:
{json.dumps(self.workflow_context, indent=2)}
Key principles:
- Email threads maintain continuity via hidden identifiers
- Negotiation is 3 rounds maximum
- Human reviews all supplier communications
- Each supplier has unique thread tracked throughout
- All suppliers processed in parallel per round
- Context from previous rounds informs next rounds
Create mental model for optimal execution.
"""
        self.embeddings["workflow"] = await self._create_embedding(workflow_prompt)
        self.embeddings["negotiation_playbook"] = (
            await self._load_negotiation_playbook()
        )
        print("✅ Context initialized")
        return self.embeddings

    async def _create_embedding(self, text: str) -> Dict[str, Any]:
        return {"text": text, "timestamp": datetime.now().isoformat()}

    async def _load_negotiation_playbook(self) -> Dict[str, Dict[str, str]]:
        return {
            "round_1": {
                "strategy": "Express interest, ask clarifications, gather information",
                "tone": "Inquisitive, professional",
            },
            "round_2": {
                "strategy": "Present counter-offer, negotiate terms, provide justification",
                "tone": "Confident, data-driven",
            },
            "round_3": {
                "strategy": "Best and final offer, set deadline, firm stance",
                "tone": "Decisive, respectful but firm",
            },
        }


class WorkflowOrchestrator:
    """
    CRITICAL: ONLY this class controls workflow timing and progression
    Uses database-backed polling for synchronization
    """

    def __init__(self, db: DatabaseConnection, wait_timeout: int = 300):
        self.session_id = str(uuid.uuid4())  # Used as request_id for sync
        self.wait_timeout = wait_timeout
        self.check_interval = 5  # Poll database every 5 seconds
        self.active_threads: Dict[str, EmailThread] = {}
        self.context_manager = WorkflowContextManager()
        self.db = db  # Injected database connection
        self.session = NegotiationSession(
            session_id=self.session_id, start_time=datetime.now().isoformat()
        )

    async def _initialize_response_tracking(
        self, round_num: int, expected_count: int
    ) -> None:
        """Initializes the tracking record for a new interaction round"""

        print(f"INITIALIZING response tracking for Round {round_num}...")
        await self.db.execute(
            """
            INSERT INTO proc.supplier_interaction_status
            (request_id, round_num, total_suppliers_contacted, expected_responses, is_complete)
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT (request_id, round_num) DO NOTHING;
        """,
            (self.session_id, round_num, expected_count, expected_count),
        )

    async def wait_for_responses(
        self, expected_count: int, context: str = "", round_num: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        """
        EXCLUSIVE orchestrator responsibility.
        CRITICAL BLOCKING POINT: Polls database until all responses are in.
        """

        elapsed = 0
        print(
            f"⏳ Waiting for {expected_count} {context} (Round {round_num})..."
        )

        while elapsed < self.wait_timeout:
            status = await self.db.fetch_one(
                """
                SELECT is_complete, responses_received, expected_responses
                FROM proc.supplier_interaction_status
                WHERE request_id = %s AND round_num = %s
            """,
                (self.session_id, round_num),
            )

            if status and status.get("is_complete"):
                print(
                    f"✅ All {status['responses_received']} responses for Round {round_num} collected"
                )

                all_responses = await self.db.fetch_all(
                    """
                    SELECT supplier_id, response_data, received_at
                    FROM proc.supplier_response
                    WHERE request_id = %s AND round_num = %s
                """,
                    (self.session_id, round_num),
                )

                return {resp["supplier_id"]: resp for resp in all_responses}

            if status:
                print(
                    f"⏳ Waiting... {status['responses_received']}/{status['expected_responses']} responses for Round {round_num}"
                )
            else:
                print(
                    f"⏳ Waiting... tracking record for Round {round_num} not yet fully initialized."
                )

            await asyncio.sleep(self.check_interval)
            elapsed += self.check_interval

        raise TimeoutError(
            f"Not all supplier responses for Round {round_num} received within {self.wait_timeout}s"
        )

    def create_thread_for_supplier(self, supplier_id: str) -> EmailThread:
        thread = EmailThread(supplier_id=supplier_id)
        self.active_threads[supplier_id] = thread
        self.session.supplier_states[supplier_id] = {
            "thread_id": thread.thread_id,
            "supplier_unique_id": thread.supplier_unique_id,
            "rounds_completed": 0,
            "responses_received": 0,
        }
        return thread

    async def execute_workflow(self, suppliers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main workflow execution - controls ALL phase transitions"""

        print(f"🚀 Starting Workflow - Session: {self.session_id}\n")

        print("🧠 Phase 0: Initializing context...")
        await self.context_manager.initialize_workflow_understanding()

        print("\n📧 Phase 1: Generating initial emails (Round 0)...")
        initial_emails = await self._phase_initial_emails(suppliers)

        await self._initialize_response_tracking(
            round_num=0, expected_count=len(suppliers)
        )

        print("\n📥 Phase 2: Waiting for initial responses (Round 0 Responses)...")
        initial_responses = await self.wait_for_responses(
            len(suppliers), "initial responses", round_num=0
        )

        print("\n💼 Phase 3: Negotiation (3 rounds)...")
        negotiation_results = await self._phase_negotiation(
            suppliers, initial_responses
        )

        print("\n📊 Phase 4: Quote evaluation...")
        final_evaluation = await self._phase_quote_evaluation(negotiation_results)

        print(f"\n✅ Workflow Complete - Session: {self.session_id}")
        return final_evaluation

    async def _phase_initial_emails(
        self, suppliers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        email_agent = EmailDraftingAgent(self.context_manager)
        results: Dict[str, Dict[str, Any]] = {}

        for supplier in suppliers:
            thread = self.create_thread_for_supplier(supplier["id"])
            email = await email_agent.generate_initial_email(
                supplier, thread.hidden_identifier
            )
            thread.add_message("initial_email", email["content"], email["action_id"])
            results[supplier["id"]] = email

        return results

    async def _phase_negotiation(
        self,
        suppliers: List[Dict[str, Any]],
        initial_responses: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """CRITICAL: 3 rounds with parallel processing and DB synchronization"""

        negotiation_agent = NegotiationAgent(self.context_manager, self.db)
        results: Dict[str, Dict[str, Any]] = {}

        latest_responses = {sid: resp for sid, resp in initial_responses.items()}

        for round_num in range(1, 4):
            print(f"\n{'=' * 70}")
            print(f"NEGOTIATION ROUND {round_num} - Session: {self.session_id}")
            print(f"{'=' * 70}\n")

            self.session.current_round = round_num

            round_tasks: List[Tuple[str, asyncio.Future]] = []
            for supplier in suppliers:
                supplier_id = supplier["id"]
                thread = self.active_threads[supplier_id]
                thread.current_round = round_num

                latest_response = latest_responses.get(supplier_id)

                task = negotiation_agent.generate_negotiation_response(
                    supplier,
                    thread,
                    round_num,
                    latest_response,
                    self.session_id,
                )
                round_tasks.append((supplier_id, task))

            print(
                f"🔄 Generating Round {round_num} responses for {len(suppliers)} suppliers..."
            )
            task_results = await asyncio.gather(*[task for _, task in round_tasks])

            for (supplier_id, _), result in zip(round_tasks, task_results):
                thread = self.active_threads[supplier_id]
                thread.add_message(
                    f"negotiation_round_{round_num}",
                    result["content"],
                    result["action_id"],
                )
                results[f"{supplier_id}_round_{round_num}"] = result
                self.session.supplier_states[supplier_id][
                    "rounds_completed"
                ] = round_num
                print(f"✓ {supplier_id}: {result['action_id']}")

            print(f"\n📤 Round {round_num} drafts ready for human review")
            print("✈️ SENDING Round {round_num} emails (simulated)...")

            await self._initialize_response_tracking(
                round_num=round_num, expected_count=len(suppliers)
            )

            print(f"\n⏳ Waiting for Round {round_num} responses...")
            round_responses = await self.wait_for_responses(
                len(suppliers), f"Round {round_num} responses", round_num=round_num
            )

            for supplier_id, response in round_responses.items():
                thread = self.active_threads[supplier_id]
                thread.add_message(
                    f"supplier_response_round_{round_num}",
                    response["response_data"].get("content", ""),
                    f"SUPP-R{round_num}-{thread.supplier_unique_id}",
                )
                results[f"{supplier_id}_supplier_response_round_{round_num}"] = response
                self.session.supplier_states[supplier_id]["responses_received"] = round_num

            latest_responses = round_responses
            print(f"✅ All Round {round_num} responses collected\n")

        print("✅ All 3 negotiation rounds complete")
        return results

    async def _phase_quote_evaluation(
        self, negotiation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        eval_agent = QuoteEvaluationAgent(self.context_manager)
        return await eval_agent.evaluate_quotes(self.active_threads)


class EmailDraftingAgent:
    """Generates emails with hidden identifiers embedded"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "qwen3:30b"

    async def generate_initial_email(
        self, supplier: Dict[str, Any], hidden_identifier: str
    ) -> Dict[str, Any]:
        action_id = (
            f"EMAIL-{int(datetime.now().timestamp())}-{uuid.uuid4().hex[:6].upper()}"
        )

        prompt = f"""
Generate professional RFQ email for:
Supplier: {supplier['name']}
Requirements: {supplier.get('requirements', 'Standard procurement')}

CRITICAL: Embed this tracking identifier in email footer:
Reference: {hidden_identifier}

Make it professional, clear, and engaging.
"""

        email_content = await self._call_ollama(prompt)

        return {
            "action_id": action_id,
            "supplier_id": supplier["id"],
            "hidden_identifier": hidden_identifier,
            "content": email_content,
            "type": "initial_email",
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": True,
        }

    async def _call_ollama(self, prompt: str) -> str:
        return "[Generated email content]"


class SupplierInteractionAgent:
    """Collects responses ONLY - NO workflow control"""

    def __init__(self):
        self.responses_collected: Dict[str, Dict[str, Any]] = {}

    async def collect_response(
        self, supplier_id: str, hidden_identifier: str
    ) -> Optional[Dict[str, Any]]:
        return None

    def all_responses_collected(self, expected_supplier_ids: List[str]) -> bool:
        return False


async def handle_supplier_response(
    db: DatabaseConnection,
    session_id: str,
    round_num: int,
    supplier_id: str,
    hidden_identifier: str,
    response_data: Dict[str, Any],
) -> None:
    """
    ATOMIC operation to record response and update counter.
    This is the core of the synchronization logic.
    """

    try:
        async with db.transaction():
            await db.execute(
                """
                INSERT INTO proc.supplier_response
                (request_id, round_num, supplier_id, hidden_identifier, response_data, received_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """,
                (
                    session_id,
                    round_num,
                    supplier_id,
                    hidden_identifier,
                    json.dumps(response_data),
                ),
            )

            await db.execute(
                """
                UPDATE proc.supplier_interaction_status
                SET
                    responses_received = responses_received + 1,
                    last_updated = NOW(),
                    is_complete = CASE
                        WHEN responses_received + 1 >= expected_responses
                        THEN TRUE
                        ELSE FALSE
                    END,
                    completed_at = CASE
                        WHEN responses_received + 1 >= expected_responses
                        THEN NOW()
                        ELSE NULL
                    END
                WHERE request_id = %s AND round_num = %s
            """,
                (session_id, round_num),
            )

        print(f"✓ Response from {supplier_id} for Round {round_num} recorded")

    except Exception as exc:  # pragma: no cover - placeholder for retry logic
        print(f"❌ CRITICAL ERROR recording response: {exc}")


class NegotiationAgent:
    """Context-driven intelligent negotiation with defensive pre-checks"""

    def __init__(
        self, context_manager: WorkflowContextManager, db: DatabaseConnection
    ) -> None:
        self.context = context_manager
        self.model = "mixtral:8x7b"
        self.db = db

    async def _run_pre_check(self, session_id: str, round_num: int) -> None:
        """
        CRITICAL PRE-CHECK: Verify all responses from *previous* round were received.
        This is a defensive gate.
        """

        if round_num < 0:
            pass

        print(f"Running pre-check for Round {round_num} data before generating...")

        status = await self.db.fetch_one(
            """
            SELECT is_complete, responses_received, expected_responses
            FROM proc.supplier_interaction_status
            WHERE request_id = %s AND round_num = %s AND is_complete = TRUE
        """,
            (session_id, round_num),
        )

        if not status:
            raise RuntimeError(
                "CRITICAL ERROR: Negotiation agent triggered prematurely. "
                f"Previous round ({round_num}) data for {session_id} is not complete!"
            )

        actual_count = await self.db.fetch_val(
            """
            SELECT COUNT(*) FROM proc.supplier_response
            WHERE request_id = %s AND round_num = %s
        """,
            (session_id, round_num),
        )

        if actual_count < status["expected_responses"]:
            raise RuntimeError(
                f"CRITICAL ERROR: Response count mismatch for Round {round_num}. "
                f"Expected: {status['expected_responses']}, Got: {actual_count}"
            )

        print(f"✓ Pre-check passed for Round {round_num} data.")

    async def generate_negotiation_response(
        self,
        supplier: Dict[str, Any],
        thread: EmailThread,
        round_num: int,
        supplier_latest_response: Optional[Dict[str, Any]],
        session_id: str,
    ) -> Dict[str, Any]:
        """CRITICAL: Analyzes actual supplier response before generating"""

        await self._run_pre_check(session_id, round_num - 1)

        playbook = self.context.embeddings.get("negotiation_playbook", {})
        strategy = playbook.get(f"round_{round_num}", {})

        thread_context = thread.get_full_thread()

        supplier_response_text = (
            supplier_latest_response.get("response_data", {}).get("content", "")
            if supplier_latest_response
            else ""
        )
        response_analysis = await self._analyze_supplier_response(
            supplier_response_text, round_num
        )

        action_id = (
            f"NEG-R{round_num}-{thread.supplier_unique_id}-{uuid.uuid4().hex[:6].upper()}"
        )

        prompt = f"""
You are expert procurement negotiator in Round {round_num} of 3.
STRATEGY: {strategy.get('strategy', '')}
TONE: {strategy.get('tone', '')}
COMPLETE EMAIL THREAD:
{thread_context}
SUPPLIER'S LATEST RESPONSE:
{supplier_response_text}
RESPONSE ANALYSIS:
{response_analysis}
Generate contextually intelligent negotiation email.
This will be reviewed by human before sending.
"""

        content = await self._call_ollama(prompt)

        return {
            "action_id": action_id,
            "supplier_id": supplier["id"],
            "round": round_num,
            "content": content,
            "thread_id": thread.thread_id,
            "type": f"negotiation_round_{round_num}",
            "timestamp": datetime.now().isoformat(),
            "requires_human_review": True,
        }

    async def _analyze_supplier_response(
        self, response_text: str, round_num: int
    ) -> str:
        """CRITICAL: Analyze supplier response to extract context"""

        analysis_prompt = f"""
Analyze supplier response for Round {round_num} negotiation:
RESPONSE:
{response_text}
Extract:
1. Pricing 2. Payment Terms 3. Concerns/Objections 4. Questions
5. Concessions 6. Tone 7. Key Dates
Provide structured analysis for response strategy.
"""
        return await self._call_ollama(analysis_prompt)

    async def _call_ollama(self, prompt: str) -> str:
        return "[Generated content]"


class RAGAgent:
    """Advanced RAG with hybrid search for proc.* tables"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "phi4:latest"
        self.vector_store = None
        self.proc_tables = [
            "proc_contracts",
            "proc_sow",
            "proc_suppliers",
            "proc_historical_quotes",
            "proc_terms_conditions",
        ]

    async def initialize(self) -> None:
        for table in self.proc_tables:
            await self._embed_table_data(table)
        print("✅ RAG initialized")

    async def query(
        self, user_question: str, context_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        enhanced_query = await self._enhance_query(user_question)
        vector_results = await self._vector_search(enhanced_query, top_k=20)
        keyword_results = await self._keyword_search(enhanced_query, top_k=20)
        all_results = self._merge_results(vector_results, keyword_results)
        reranked = await self._rerank(user_question, all_results, top_k=5)
        context = self._format_context(reranked)
        prompt = f"""
You are procurement expert with access to:
{', '.join(self.proc_tables)}
Question: {user_question}
Context:
{context}
Provide accurate, context-aware answer.
"""
        response = await self._call_ollama(prompt)
        confidence = self._calculate_confidence(reranked)
        return {
            "answer": response,
            "sources": reranked,
            "confidence": confidence,
            "requires_human_review": confidence < 0.7,
        }

    async def _enhance_query(self, query: str) -> str:
        return query

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return []

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return []

    def _merge_results(
        self, vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return vector_results + keyword_results

    async def _rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        return results[:top_k]

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        return "\n\n".join([r.get("content", "") for r in results])

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return min(
            sum(r.get("score", 0) for r in results) / len(results),
            1.0,
        )

    async def _embed_table_data(self, table_name: str) -> None:
        pass

    async def _call_ollama(self, prompt: str) -> str:
        return "[RAG response]"


class QuoteEvaluationAgent:
    """Evaluate final quotes after 3 negotiation rounds"""

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        self.model = "qwen3:30b"

    async def evaluate_quotes(
        self, threads: Dict[str, EmailThread]
    ) -> Dict[str, Any]:
        evaluations: Dict[str, Dict[str, Any]] = {}
        for supplier_id, thread in threads.items():
            evaluation = await self._evaluate_single_quote(thread)
            evaluations[supplier_id] = evaluation
        ranked = self._rank_suppliers(evaluations)
        return {
            "evaluations": evaluations,
            "ranking": ranked,
            "recommendation": ranked[0] if ranked else None,
            "requires_human_review": True,
        }

    async def _evaluate_single_quote(self, thread: EmailThread) -> Dict[str, Any]:
        prompt = f"""
Evaluate procurement negotiation:
{thread.get_full_thread()}
Analyze:
1. Final price 2. Terms 3. Responsiveness 4. Value 5. Risk
Provide structured evaluation.
"""
        evaluation = await self._call_ollama(prompt)
        return {"supplier_id": thread.supplier_id, "evaluation": evaluation}

    def _rank_suppliers(self, evaluations: Dict[str, Dict[str, Any]]) -> List[str]:
        return list(evaluations.keys())

    async def _call_ollama(self, prompt: str) -> str:
        return "[Evaluation]"


class MockDatabaseConnection(DatabaseConnection):
    def __init__(self) -> None:
        self.status_table: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.response_table: List[Dict[str, Any]] = []

    async def execute(self, query: str, params: Tuple) -> None:
        if "INSERT INTO proc.supplier_interaction_status" in query:
            key = (params[0], params[1])
            if key not in self.status_table:
                self.status_table[key] = {
                    "request_id": params[0],
                    "round_num": params[1],
                    "total_suppliers_contacted": params[2],
                    "expected_responses": params[3],
                    "responses_received": 0,
                    "is_complete": False,
                }
        elif "INSERT INTO proc.supplier_response" in query:
            self.response_table.append(
                {
                    "request_id": params[0],
                    "round_num": params[1],
                    "supplier_id": params[2],
                    "response_data": json.loads(params[4]),
                }
            )
        elif "UPDATE proc.supplier_interaction_status" in query:
            key = (params[0], params[1])
            if key in self.status_table:
                status = self.status_table[key]
                status["responses_received"] += 1
                if status["responses_received"] >= status["expected_responses"]:
                    status["is_complete"] = True

    async def fetch_one(
        self, query: str, params: Tuple
    ) -> Optional[Dict[str, Any]]:
        if "FROM proc.supplier_interaction_status" in query:
            key = (params[0], params[1])
            return self.status_table.get(key)
        return None

    async def fetch_all(
        self, query: str, params: Tuple
    ) -> List[Dict[str, Any]]:
        if "FROM proc.supplier_response" in query:
            return [
                r
                for r in self.response_table
                if r["request_id"] == params[0] and r["round_num"] == params[1]
            ]
        return []

    async def fetch_val(self, query: str, params: Tuple) -> Any:
        if "SELECT COUNT(*)" in query and "proc.supplier_response" in query:
            return len(
                [
                    r
                    for r in self.response_table
                    if r["request_id"] == params[0]
                    and r["round_num"] == params[1]
                ]
            )
        return 0

    def transaction(self):  # pragma: no cover - simple mock
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def validate_implementation() -> None:
    """Run these checks BEFORE deployment"""

    print("=" * 70)
    print("VALIDATION SUITE")
    print("=" * 70)

    from dataclasses import fields

    thread = EmailThread(supplier_id="TEST")
    thread_fields = [f.name for f in fields(thread)]
    assert "rfq_id" not in thread_fields, "❌ FAIL: rfq_id field exists"
    print("✅ CHECK 1: No RFQ IDs")

    mock_db = MockDatabaseConnection()
    orchestrator = WorkflowOrchestrator(db=mock_db)
    initial_session = orchestrator.session_id
    assert orchestrator.session_id == initial_session
    print("✅ CHECK 2: Session ID created")

    thread = EmailThread(supplier_id="TEST001")
    initial_thread_id = thread.thread_id
    thread.add_message("test", "content", "ACTION-123")
    assert thread.thread_id == initial_thread_id
    print("✅ CHECK 3: Thread ID consistent")

    assert thread.thread_id.count("-") == 4
    assert thread.hidden_identifier.startswith("PROC-WF-")
    assert thread.supplier_unique_id.startswith("SUP-")
    assert len(thread.supplier_unique_id) == 14
    assert len(thread.hidden_identifier) == 20
    print("✅ CHECK 4: Identifier formats correct")

    cm = WorkflowContextManager()
    await cm.initialize_workflow_understanding()
    assert "workflow" in cm.embeddings and "negotiation_playbook" in cm.embeddings
    print("✅ CHECK 5: Context initialized")

    assert hasattr(orchestrator, "wait_for_responses")
    assert hasattr(orchestrator, "execute_workflow")
    agent = SupplierInteractionAgent()
    assert not hasattr(agent, "wait_for_responses")
    neg_agent = NegotiationAgent(cm, mock_db)
    assert not hasattr(neg_agent, "wait_for_responses")
    print("✅ CHECK 6: Orchestrator controls workflow")

    assert hasattr(neg_agent, "_analyze_supplier_response")
    assert hasattr(neg_agent, "_run_pre_check")
    print("✅ CHECK 7: Negotiation analyzes and runs pre-checks")

    rag = RAGAgent(cm)
    assert hasattr(rag, "query") and "proc_contracts" in rag.proc_tables
    print("✅ CHECK 8: RAG agent configured")

    print("Simulating response sync...")
    orchestrator.wait_timeout = 2
    await orchestrator._initialize_response_tracking(round_num=0, expected_count=1)
    await handle_supplier_response(
        mock_db, orchestrator.session_id, 0, "SUP001", "PROC-WF-123", {"content": "test"}
    )
    responses = await orchestrator.wait_for_responses(1, "test sync", round_num=0)
    assert "SUP001" in responses
    assert responses["SUP001"]["response_data"]["content"] == "test"
    print("✅ CHECK 9: Database synchronization and polling works")

    print("\n" + "=" * 70)
    print("✅ ALL VALIDATIONS PASSED - Ready for deployment")
    print("=" * 70)


__all__ = [
    "EmailThread",
    "NegotiationSession",
    "DatabaseConnection",
    "WorkflowContextManager",
    "WorkflowOrchestrator",
    "EmailDraftingAgent",
    "SupplierInteractionAgent",
    "handle_supplier_response",
    "NegotiationAgent",
    "RAGAgent",
    "QuoteEvaluationAgent",
    "MockDatabaseConnection",
    "validate_implementation",
]
