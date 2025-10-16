import json
import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supplier_interaction_agent import SupplierInteractionAgent
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from repositories import (
    draft_rfq_emails_repo,
    supplier_response_repo,
    workflow_email_tracking_repo,
)
from repositories.supplier_response_repo import SupplierResponseRow


class DummyCursor:
    description: tuple = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, *args, **kwargs):
        self._row = ("gold",)

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []


class DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor()


class DummyQdrant:
    def search(self, **kwargs):
        return [SimpleNamespace(payload={"document_type": "email"})]


class DummyEmbedding:
    def encode(self, _):
        return [0.0]


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            qdrant_collection_name="test",
            script_user="tester",
            email_response_poll_seconds=1,
            email_response_timeout_seconds=5,
            email_response_batch_limit=3,
        )
        self.qdrant_client = DummyQdrant()
        self.embedding_model = DummyEmbedding()
        self.agents: Dict[str, object] = {}
        self.dispatch_service_started = True

    def get_db_connection(self):
        return DummyConn()


def test_build_thread_headers_payload_readds_angle_brackets():
    agent = SupplierInteractionAgent(DummyNick())

    payload = agent._build_thread_headers_payload(
        "message-123@example.com",
        [
            "reference-1@example.com",
            "<reference-2@example.com>",
            " reference-3@example.com> ",
            "<reference-4@example.com",
        ],
    )

    assert payload == {
        "message_id": "<message-123@example.com>",
        "references": [
            "<reference-1@example.com>",
            "<reference-2@example.com>",
            "<reference-3@example.com>",
            "<reference-4@example.com>",
        ],
    }


@pytest.fixture(autouse=True)
def stub_response_coordinator(monkeypatch):
    class StubCoordinator:
        def await_dispatch_completion(
            self,
            *,
            workflow_id,
            unique_ids,
            timeout=None,
            poll_interval=None,
            wait_for_all=True,
        ):
            filtered = [uid for uid in unique_ids if uid]
            expected = len(filtered)
            return {
                "workflow_id": workflow_id,
                "expected_dispatches": expected,
                "completed_dispatches": expected,
                "complete": True,
                "unique_ids": list(filtered),
            }

        def await_response_population(
            self,
            *,
            workflow_id,
            unique_ids,
            timeout=None,
            poll_interval=None,
            wait_for_all=True,
        ):
            filtered = [uid for uid in unique_ids if uid]
            expected = len(filtered)
            return {
                "workflow_id": workflow_id,
                "expected_responses": expected,
                "completed_responses": expected,
                "complete": True,
                "unique_ids": list(filtered),
            }

        def await_first_dispatch(
            self,
            *,
            workflow_id,
            unique_ids,
            timeout=None,
            poll_interval=None,
        ):
            filtered = [uid for uid in unique_ids if uid]
            return {
                "workflow_id": workflow_id,
                "activated": bool(filtered),
                "first_unique_id": filtered[0] if filtered else None,
                "wait_seconds": 0.0,
                "unique_ids": list(filtered),
            }

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.SupplierResponseWorkflow",
        lambda: StubCoordinator(),
    )


@pytest.fixture(autouse=True)
def reduce_response_gate_interval(monkeypatch):
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.SupplierInteractionAgent.RESPONSE_GATE_POLL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.SupplierInteractionAgent.WORKFLOW_POLL_INTERVAL_SECONDS",
        0.01,
    )


def test_supplier_interaction_agent():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    context = AgentContext(
        workflow_id='wf1',
        agent_id='supplier_interaction',
        user_id='u1',
        input_data={
            'subject': 'Re: RFQ-20240101-abcd1234',
            'message': 'Our price is 1500 with lead time 10 days',
            'supplier_id': 's1',
            'target_price': '1000',
        }
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.next_agents == ['NegotiationAgent']
    assert output.data['price'] == 1500.0
    assert output.data['lead_time'] == '10'


def test_supplier_interaction_wait_for_response(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-001",
                unique_id="uid-001",
                supplier_id="S1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m-001",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-001"],
    }

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return [
            {
                "workflow_id": "wf-001",
                "unique_id": "uid-001",
                "supplier_id": "S1",
                "response_text": "Thanks for the opportunity",
                "subject": "Re: Quote-20240101-abcd1234",
                "message_id": "m-001",
                "from_addr": "supplier@example.com",
                "received_time": datetime.fromtimestamp(0),
            }
        ]

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 2,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 2,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-001",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: None,
    )

    result = agent.wait_for_response(
        timeout=1,
        poll_interval=0,
        rfq_id="RFQ-20240101-abcd1234",
        supplier_id="S1",
        dispatch_run_id="run-001",
        workflow_id="wf-001",
        draft_action_id="draft-001",
        unique_id="uid-001",
    )

    assert result is not None
    assert fetch_calls == ["wf-001"]


def test_waits_for_dispatch_metadata_before_polling(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-await",
                unique_id="uid-await",
                supplier_id="S1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="msg-await",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-await"],
    }

    metadata_sequence = [None, metadata]
    load_calls: List[str] = []

    def fake_load(workflow_id: str):
        load_calls.append(workflow_id)
        if metadata_sequence:
            return metadata_sequence.pop(0)
        return metadata

    fetch_calls: List[str] = []

    def fake_fetch(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return [
            {
                "workflow_id": workflow_id,
                "unique_id": "uid-await",
                "supplier_id": "S1",
                "response_text": "Appreciate the opportunity",
                "subject": "Re: Quote",
                "message_id": "msg-await",
                "from_addr": "supplier@example.com",
                "received_time": datetime.fromtimestamp(0),
            }
        ]

    monkeypatch.setattr(agent, "_load_dispatch_metadata", fake_load)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-await",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: None,
    )
    monkeypatch.setattr(
        agent,
        "_process_responses_concurrently",
        lambda rows: list(rows),
    )
    monkeypatch.setattr(
        agent,
        "_collect_workflow_unique_ids",
        lambda *_, **__: (["uid-await"], 0.0),
    )

    sleep_calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", fake_sleep)

    responses = agent._await_supplier_response_rows(
        "wf-await",
        timeout=2,
        poll_interval=0,
    )

    assert responses
    assert load_calls == ["wf-await", "wf-await"]
    assert fetch_calls == ["wf-await"]
    assert sleep_calls  # ensure loop yielded while waiting for metadata


def test_wait_for_response_realigns_workflow_from_unique(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-canonical",
                unique_id="uid-redirect",
                supplier_id="S2",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="msg-redirect",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-redirect"],
    }

    awaited: Dict[str, str] = {}

    def fake_metadata(workflow_id, **_kwargs):
        awaited["workflow_id"] = workflow_id
        return metadata

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return [
            {
                "workflow_id": workflow_id,
                "unique_id": "uid-redirect",
                "supplier_id": "S2",
                "response_text": "Redirected",
                "subject": "Re: Quote-20240101-redirect",
                "message_id": "msg-redirect",
                "from_addr": "supplier2@example.com",
                "received_time": datetime.fromtimestamp(0),
            }
        ]

    agent._load_dispatch_metadata = fake_metadata  # type: ignore[assignment]
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-canonical",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: None,
    )
    result = agent.wait_for_response(
        timeout=1,
        poll_interval=0,
        rfq_id="RFQ-20240101-redirect",
        supplier_id="S2",
        dispatch_run_id="run-redirect",
        workflow_id="wf-mismatch",
        draft_action_id="draft-redirect",
        unique_id="uid-redirect",
    )

    assert result is not None
    assert fetch_calls == ["wf-canonical"]
    assert awaited.get("workflow_id") == "wf-canonical"


def test_collect_workflow_unique_ids_merges_repo(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    sequence = [[], ["uid-seed", "uid-extra"]]

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.init_schema",
        lambda: None,
    )

    def fake_load_unique_ids(*, workflow_id):
        assert workflow_id == "wf-extra"
        return sequence.pop(0) if sequence else ["uid-seed", "uid-extra"]

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.load_workflow_unique_ids",
        fake_load_unique_ids,
    )

    sleep_calls: List[float] = []

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    collected, elapsed = agent._collect_workflow_unique_ids(
        workflow_id="wf-extra",
        seed_ids=["uid-seed"],
        timeout=5,
        poll_interval=0,
        expected_total=2,
    )

    assert set(collected) == {"uid-seed", "uid-extra"}
    assert sleep_calls  # ensured at least one wait cycle occurred
    assert elapsed >= 0


def test_wait_for_response_requires_ready_dispatch_metadata(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata_ready = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-slow",
                unique_id="uid-slow",
                supplier_id="SUP-SLOW",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="msg-slow",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-slow"],
    }

    metadata_calls: List[str] = []
    metadata_sequence = deque([None, None, metadata_ready])

    def fake_metadata(workflow_id, **_kwargs):
        metadata_calls.append(workflow_id)
        if metadata_sequence:
            return metadata_sequence.popleft()
        return metadata_ready

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return [
            {
                "workflow_id": workflow_id,
                "unique_id": "uid-slow",
                "supplier_id": "SUP-SLOW",
                "response_text": "Quote ready",
                "subject": "Re: RFQ-20240101-SLOW",
                "message_id": "msg-slow",
                "from_addr": "slow@example.com",
                "received_time": datetime.fromtimestamp(1),
            }
        ]

    agent._load_dispatch_metadata = fake_metadata  # type: ignore[assignment]
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 2,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-slow",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: None,
    )

    sleep_calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", fake_sleep)
    monkeypatch.setattr(
        agent,
        "_collect_workflow_unique_ids",
        lambda *_, **__: (["uid-002"], 0.0),
    )

    result = agent.wait_for_response(
        timeout=5,
        poll_interval=0,
        rfq_id="RFQ-20240101-SLOW",
        supplier_id="SUP-SLOW",
        workflow_id="wf-slow",
        unique_id="uid-slow",
    )

    assert result is not None
    assert result["supplier_id"] == "SUP-SLOW"
    assert fetch_calls == ["wf-slow"]
    assert metadata_calls == ["wf-slow", "wf-slow", "wf-slow"]
    assert sleep_calls  # ensures we waited for metadata


def test_wait_for_response_requires_available_payload(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-002",
                unique_id="uid-002",
                supplier_id="S1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m-002",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-002"],
    }

    pending_batches = deque(
        [
            [],
            [
                {
                    "workflow_id": "wf-002",
                    "unique_id": "uid-002",
                    "supplier_id": "S1",
                    "response_text": "Pricing ready",
                    "subject": "Re: Quote-20240101-abcd1234",
                    "message_id": "m-002",
                    "from_addr": "supplier@example.com",
                    "received_time": datetime.fromtimestamp(5),
                }
            ],
        ]
    )

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        batch = pending_batches[0]
        if len(pending_batches) > 1:
            pending_batches.popleft()
        return batch

    count_sequence = deque([0, 1])

    def fake_count_pending(**_kwargs):
        if count_sequence:
            return count_sequence.popleft()
        return 1

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        fake_count_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-002",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: None,
    )

    sleep_calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", fake_sleep)

    result = agent.wait_for_response(
        timeout=90,
        poll_interval=0,
        rfq_id="RFQ-20240101-abcd1234",
        supplier_id="S1",
        dispatch_run_id="run-002",
        workflow_id="wf-002",
        draft_action_id="draft-002",
        unique_id="uid-002",
    )

    assert result is not None
    assert result["supplier_status"] == "success"
    assert result["supplier_output"]["response_text"] == "Pricing ready"
    assert len(fetch_calls) >= 2
    assert sleep_calls and sleep_calls[0] == pytest.approx(1.0)


def test_supplier_interaction_waits_using_drafts():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    drafts = [
        {
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "S1",
            "receiver": "supplier@example.com",
        }
    ]

    def fake_wait_for_response(**kwargs):
        assert kwargs["rfq_id"] == "RFQ-20240101-abcd1234"
        assert kwargs["supplier_id"] == "S1"
        return {
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "S1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "supplier_status": "success",
            "target_price": 1000.0,
            "supplier_output": {
                "price": 900.0,
                "lead_time": "5",
                "response_text": "Quoted price 900 in 5 days",
                "related_documents": [{"doc": 1}],
            },
        }

    agent.wait_for_response = fake_wait_for_response  # type: ignore[assignment]

    context = AgentContext(
        workflow_id="wf2",
        agent_id="supplier_interaction",
        user_id="u1",
        input_data={
            "subject": "RFQ-20240101-abcd1234 – Request for Quotation",
            "drafts": drafts,
            "supplier_id": "S1",
            "target_price": "1000",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["rfq_id"] == "RFQ-20240101-abcd1234"
    assert output.data["price"] == 900.0
    assert output.data["target_price"] == 1000.0
    assert output.next_agents == ["QuoteEvaluationAgent"]


def test_wait_for_multiple_responses_aggregates_by_workflow(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-shared",
                unique_id="uid-1",
                supplier_id="SUP-1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m1",
            ),
            SimpleNamespace(
                workflow_id="wf-shared",
                unique_id="uid-2",
                supplier_id="SUP-2",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m2",
            ),
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return [
            {
                "workflow_id": "wf-shared",
                "unique_id": "uid-1",
                "supplier_id": "SUP-1",
                "response_text": "Quote 1",
                "subject": "Re: Quote-20240101-AAA11111",
                "message_id": "m1",
                "from_addr": "one@example.com",
                "received_time": datetime.fromtimestamp(10),
            },
            {
                "workflow_id": "wf-shared",
                "unique_id": "uid-2",
                "supplier_id": "SUP-2",
                "response_text": "Quote 2",
                "subject": "Re: Quote-20240101-BBB22222",
                "message_id": "m2",
                "from_addr": "two@example.com",
                "received_time": datetime.fromtimestamp(15),
            },
        ]

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 2,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-shared",
    )

    drafts = [
        {
            "rfq_id": "RFQ-20240101-AAA11111",
            "supplier_id": "SUP-1",
            "recipients": ["one@example.com"],
            "subject": "Re: RFQ-20240101-AAA11111",
            "dispatch_run_id": "run-sup-1",
            "workflow_id": "wf-shared",
            "draft_action_id": "draft-1",
            "unique_id": "uid-1",
        },
        {
            "rfq_id": "RFQ-20240101-BBB22222",
            "supplier_id": "SUP-2",
            "receiver": "two@example.com",
            "subject": "Re: RFQ-20240101-BBB22222",
            "dispatch_run_id": "run-sup-2",
            "workflow_id": "wf-shared",
            "draft_action_id": "draft-2",
            "unique_id": "uid-2",
        },
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=5,
        poll_interval=1,
        limit=1,
    )

    assert len(results) == len(drafts)
    assert fetch_calls == ["wf-shared"]
    for result, draft in zip(results, drafts):
        assert result is not None
        assert result["supplier_id"] == draft["supplier_id"]
        assert result["unique_id"] == draft["unique_id"]


def test_wait_for_multiple_responses_realigns_mixed_workflows(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-shared",
                unique_id="uid-1",
                supplier_id="SUP-1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m1",
            ),
            SimpleNamespace(
                workflow_id="wf-shared",
                unique_id="uid-2",
                supplier_id="SUP-2",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m2",
            ),
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    responses = [
        {
            "workflow_id": "wf-shared",
            "unique_id": "uid-1",
            "supplier_id": "SUP-1",
            "response_text": "Quote 1",
            "subject": "Re: RFQ-1",
            "message_id": "m1",
            "from_addr": "one@example.com",
            "received_time": datetime.fromtimestamp(1),
        },
        {
            "workflow_id": "wf-shared",
            "unique_id": "uid-2",
            "supplier_id": "SUP-2",
            "response_text": "Quote 2",
            "subject": "Re: RFQ-2",
            "message_id": "m2",
            "from_addr": "two@example.com",
            "received_time": datetime.fromtimestamp(2),
        },
    ]

    class DummyCoordinator:
        def __init__(self):
            self.dispatch_calls: List[Dict[str, Any]] = []
            self.response_calls: List[Dict[str, Any]] = []

        def await_dispatch_completion(self, **kwargs):
            self.dispatch_calls.append(kwargs)
            return {
                "workflow_id": kwargs["workflow_id"],
                "expected_dispatches": len(kwargs["unique_ids"]),
                "completed_dispatches": len(kwargs["unique_ids"]),
                "complete": True,
                "unique_ids": list(kwargs["unique_ids"]),
            }

        def await_response_population(self, **kwargs):
            self.response_calls.append(kwargs)
            return {
                "workflow_id": kwargs["workflow_id"],
                "expected_responses": len(kwargs["unique_ids"]),
                "completed_responses": len(kwargs["unique_ids"]),
                "complete": True,
                "unique_ids": list(kwargs["unique_ids"]),
            }

    coordinator = DummyCoordinator()
    agent._response_coordinator = coordinator  # type: ignore[assignment]

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.load_workflow_unique_ids",
        lambda **_: ["uid-1", "uid-2"],
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-shared",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        lambda **_: list(responses),
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        agent,
        "_process_responses_concurrently",
        lambda rows: list(rows),
    )

    drafts = [
        {
            "rfq_id": "RFQ-1",
            "supplier_id": "SUP-1",
            "workflow_id": "wf-alpha",
            "unique_id": "uid-1",
        },
        {
            "rfq_id": "RFQ-2",
            "supplier_id": "SUP-2",
            "workflow_id": "wf-beta",
            "unique_id": "uid-2",
        },
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=5,
        poll_interval=1,
        limit=1,
    )

    assert coordinator.dispatch_calls
    assert coordinator.dispatch_calls[0]["workflow_id"] == "wf-shared"
    assert coordinator.dispatch_calls[0]["unique_ids"] == ["uid-1", "uid-2"]
    for result, draft in zip(results, drafts):
        assert result is not None
        assert result["unique_id"] == draft["unique_id"]
        assert result["supplier_id"] == draft["supplier_id"]


def test_wait_for_multiple_responses_requires_dispatch_completion(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(
        agent,
        "_await_dispatch_ready",
        lambda **_: {
            "workflow_id": "wf-gap",
            "unique_ids": ["uid-gap"],
            "expected_dispatches": 1,
            "completed_dispatches": 0,
            "complete": False,
        },
    )

    drafts = [
        {
            "rfq_id": "RFQ-gap",
            "supplier_id": "SUP-gap",
            "unique_id": "uid-gap",
            "workflow_id": "wf-gap",
        }
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=5,
        poll_interval=1,
        limit=1,
    )

    assert results == [None]


def test_wait_for_multiple_responses_requires_response_completion(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-incomplete",
                unique_id="uid-x",
                supplier_id="SUP-X",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="mx",
            )
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-x"],
    }

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        agent,
        "_await_dispatch_ready",
        lambda **_: {
            "workflow_id": "wf-incomplete",
            "unique_ids": ["uid-x"],
            "expected_dispatches": 1,
            "completed_dispatches": 1,
            "complete": True,
        },
    )
    monkeypatch.setattr(
        agent._response_coordinator,
        "await_response_population",
        lambda **_: {
            "workflow_id": "wf-incomplete",
            "expected_responses": 1,
            "completed_responses": 0,
            "complete": False,
        },
    )

    drafts = [
        {
            "rfq_id": "RFQ-incomplete",
            "supplier_id": "SUP-X",
            "unique_id": "uid-x",
            "workflow_id": "wf-incomplete",
        }
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=10,
        poll_interval=0,
        limit=1,
    )

    assert results == [None]


def test_wait_for_multiple_responses_waits_until_complete(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-wait",
                unique_id="uid-1",
                supplier_id="SUP-1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m1",
            ),
            SimpleNamespace(
                workflow_id="wf-wait",
                unique_id="uid-2",
                supplier_id="SUP-2",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m2",
            ),
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    agent._load_dispatch_metadata = lambda *_args, **_kwargs: metadata  # type: ignore[assignment]

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        lambda **_: [
            {
                "workflow_id": "wf-wait",
                "unique_id": "uid-1",
                "supplier_id": "SUP-1",
                "response_text": "Quote 1",
                "subject": "Re: RFQ-1",
                "message_id": "m1",
                "from_addr": "one@example.com",
                "received_time": datetime.fromtimestamp(1),
            }
        ],
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        agent._response_coordinator,
        "await_response_population",
        lambda **_: {
            "workflow_id": "wf-wait",
            "expected_responses": 2,
            "completed_responses": 2,
            "complete": True,
        },
    )

    drafts = [
        {
            "rfq_id": "RFQ-1",
            "supplier_id": "SUP-1",
            "unique_id": "uid-1",
            "workflow_id": "wf-wait",
        },
        {
            "rfq_id": "RFQ-2",
            "supplier_id": "SUP-2",
            "unique_id": "uid-2",
            "workflow_id": "wf-wait",
        },
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=10,
        poll_interval=0,
        limit=1,
    )

    assert results == [None, None]


def test_wait_for_multiple_responses_uses_canonical_workflow(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-canonical",
                unique_id="uid-a",
                supplier_id="SUP-A",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="ma",
            ),
            SimpleNamespace(
                workflow_id="wf-canonical",
                unique_id="uid-b",
                supplier_id="SUP-B",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="mb",
            ),
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-a", "uid-b"],
    }

    metadata_calls: List[str] = []

    def fake_metadata(workflow_id, **_kwargs):
        metadata_calls.append(workflow_id)
        return metadata

    agent._load_dispatch_metadata = fake_metadata  # type: ignore[assignment]

    pending_rows = [
        {
            "workflow_id": "wf-canonical",
            "unique_id": "uid-a",
            "supplier_id": "SUP-A",
            "response_text": "Quote A",
            "subject": "Re: Quote-20240101-AAA",
            "message_id": "ma",
            "from_addr": "a@example.com",
            "received_time": datetime.fromtimestamp(10),
        },
        {
            "workflow_id": "wf-canonical",
            "unique_id": "uid-b",
            "supplier_id": "SUP-B",
            "response_text": "Quote B",
            "subject": "Re: Quote-20240101-BBB",
            "message_id": "mb",
            "from_addr": "b@example.com",
            "received_time": datetime.fromtimestamp(12),
        },
    ]

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        return pending_rows

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        lambda **_: 2,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-canonical",
    )

    drafts = [
        {
            "rfq_id": "RFQ-20240101-AAA",
            "supplier_id": "SUP-A",
            "unique_id": "uid-a",
            "subject": "RFQ-20240101-AAA",
        },
        {
            "rfq_id": "RFQ-20240101-BBB",
            "supplier_id": "SUP-B",
            "unique_id": "uid-b",
            "subject": "RFQ-20240101-BBB",
        },
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=5,
        poll_interval=1,
        limit=1,
    )

    assert fetch_calls == ["wf-canonical"]
    assert metadata_calls == ["wf-canonical"]
    assert all(result is not None for result in results)
    assert {r["unique_id"] for r in results if r} == {"uid-a", "uid-b"}


def test_poll_collects_only_when_all_responses_present(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    base_time = datetime.fromtimestamp(0)
    dispatch_rows = [
        SimpleNamespace(unique_id="uid-1", supplier_id="SUP-1", dispatched_at=base_time, message_id="m1"),
        SimpleNamespace(unique_id="uid-2", supplier_id="SUP-2", dispatched_at=base_time, message_id="m2"),
    ]
    metadata_ready = {
        "rows": dispatch_rows,
        "last_dispatched_at": base_time,
        "unique_ids": ["uid-1", "uid-2"],
    }

    metadata_sequence = deque([None, metadata_ready])
    metadata_calls: List[str] = []

    def fake_metadata(workflow_id):
        assert workflow_id == "wf-bulk"
        metadata_calls.append(workflow_id)
        return metadata_sequence.popleft() if metadata_sequence else metadata_ready

    agent._load_dispatch_metadata = fake_metadata  # type: ignore[assignment]

    pending_batches = deque(
        [
            [
                {
                    "workflow_id": "wf-bulk",
                    "unique_id": "uid-1",
                    "supplier_id": "SUP-1",
                    "response_text": "Quote 1",
                    "subject": "Re: Quote-20240101-AAA11111",
                    "message_id": "m1",
                    "from_addr": "one@example.com",
                    "received_time": datetime.fromtimestamp(20),
                }
            ],
            [
                {
                    "workflow_id": "wf-bulk",
                    "unique_id": "uid-1",
                    "supplier_id": "SUP-1",
                    "response_text": "Quote 1",
                    "subject": "Re: Quote-20240101-AAA11111",
                    "message_id": "m1",
                    "from_addr": "one@example.com",
                    "received_time": datetime.fromtimestamp(20),
                },
                {
                    "workflow_id": "wf-bulk",
                    "unique_id": "uid-2",
                    "supplier_id": "SUP-2",
                    "response_text": "Quote 2",
                    "subject": "Re: Quote-20240101-BBB22222",
                    "message_id": "m2",
                    "from_addr": "two@example.com",
                    "received_time": datetime.fromtimestamp(25),
                },
            ],
        ]
    )

    fetch_calls: List[str] = []

    def fake_fetch_pending(*, workflow_id, **_kwargs):
        fetch_calls.append(workflow_id)
        assert workflow_id == "wf-bulk"
        batch = pending_batches[0]
        if len(pending_batches) > 1:
            pending_batches.popleft()
        return batch

    def fake_count_pending(*, workflow_id, unique_ids=None, **_kwargs):
        assert workflow_id == "wf-bulk"
        batch = pending_batches[0]
        if unique_ids:
            targets = set(unique_ids)
            return len([row for row in batch if row.get("unique_id") in targets])
        return len(batch)

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.count_pending",
        fake_count_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )
    initial = agent._poll_supplier_response_rows("wf-bulk")
    assert initial == []
    assert fetch_calls == []

    incomplete = agent._poll_supplier_response_rows("wf-bulk")
    assert incomplete == []
    assert fetch_calls == ["wf-bulk"]

    complete = agent._poll_supplier_response_rows("wf-bulk")

    assert fetch_calls == ["wf-bulk", "wf-bulk"]
    assert metadata_calls == ["wf-bulk", "wf-bulk", "wf-bulk"]
    assert {row["unique_id"] for row in complete} == {"uid-1", "uid-2"}
    assert [row["unique_id"] for row in complete] == ["uid-1", "uid-2"]


def test_await_all_responses_processes_parallel_results(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    drafts = [
        {"rfq_id": "RFQ-20240101-AAA11111", "supplier_id": "SUP-1"},
        {"rfq_id": "RFQ-20240101-BBB22222", "supplier_id": "SUP-2"},
    ]

    parallel_payloads = [
        {
            "rfq_id": "RFQ-20240101-AAA11111",
            "workflow_id": "wf-parallel",
            "unique_id": "draft-1",
            "supplier_id": "SUP-1",
            "subject": "Re: RFQ-20240101-AAA11111",
            "supplier_output": {
                "price": 1100.0,
                "lead_time": "5",
                "response_text": "Quoted 1100 in 5 days",
            },
        },
        {
            "rfq_id": "RFQ-20240101-BBB22222",
            "workflow_id": "wf-parallel",
            "unique_id": "draft-2",
            "supplier_id": "SUP-2",
            "subject": "Re: RFQ-20240101-BBB22222",
            "supplier_output": {
                "price": 900.0,
                "lead_time": "4",
                "response_text": "Quoted 900 in 4 days",
            },
        },
    ]

    wait_calls: List[Dict[str, Any]] = []

    def fake_wait_for_multiple(
        self,
        drafts,
        *,
        timeout: int,
        poll_interval: Optional[int],
        limit: int,
    ):
        wait_calls.append(
            {
                "drafts": drafts,
                "timeout": timeout,
                "poll_interval": poll_interval,
                "limit": limit,
            }
        )
        return parallel_payloads

    def fail_wait_for_response(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("wait_for_response should not be called when awaiting all responses")

    monkeypatch.setattr(
        SupplierInteractionAgent,
        "wait_for_multiple_responses",
        fake_wait_for_multiple,
        raising=False,
    )
    monkeypatch.setattr(
        SupplierInteractionAgent,
        "wait_for_response",
        fail_wait_for_response,
        raising=False,
    )

    stored: List[Dict[str, Any]] = []

    def fake_store(workflow_id, supplier_id, message, parsed, *, unique_id=None, rfq_id=None, **kwargs):
        stored.append(
            {
                "workflow_id": workflow_id,
                "rfq_id": rfq_id,
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "message": message,
                "parsed": parsed,
                "meta": kwargs,
            }
        )

    monkeypatch.setattr(agent, "_store_response", fake_store)
    monkeypatch.setattr(agent, "vector_search", lambda *_args, **_kwargs: [])

    context = AgentContext(
        workflow_id="wf-parallel",
        agent_id="supplier_interaction",
        user_id="tester",
        input_data={
            "subject": "RFQ multi supplier",
            "drafts": drafts,
            "await_response": True,
            "await_all_responses": True,
            "rfq_id": "RFQ-20240101-AAA11111",
            "supplier_id": "SUP-1",
            "target_price": "1000",
        },
    )

    output = agent.run(context)

    assert wait_calls and wait_calls[0]["drafts"] == drafts
    assert output.status == AgentStatus.SUCCESS
    assert output.data["rfq_id"] == "RFQ-20240101-AAA11111"
    assert output.data["price"] == 1100.0
    assert output.next_agents == ["NegotiationAgent"]
    assert any(entry["supplier_id"] == "SUP-2" for entry in stored)
    assert any(entry["supplier_id"] == "SUP-1" for entry in stored)


def test_parse_response_uses_llm_when_available(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    llm_payload = {
        "response": json.dumps(
            {
                "price": 775.5,
                "lead_time_days": 7,
                "summary": "Supplier will expedite the delivery schedule.",
            }
        )
    }

    monkeypatch.setattr(agent, "call_ollama", lambda **kwargs: llm_payload)

    parsed = agent._parse_response(
        "Offering revised pricing",
        subject="Re: RFQ-20240101-ABCD1234",
        rfq_id="RFQ-20240101-ABCD1234",
        supplier_id="SUP-1",
    )

    assert parsed["price"] == pytest.approx(775.5)
    assert parsed["lead_time"] == "7"
    assert parsed["context_summary"].startswith("Supplier will expedite")


def test_store_response_persists_workflow_records(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    captured: Dict[str, SupplierResponseRow] = {}

    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)
    monkeypatch.setattr(
        workflow_email_tracking_repo, "lookup_workflow_for_unique", lambda **_: None
    )
    dispatch_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_dispatch_row",
        lambda **_: SimpleNamespace(
            dispatched_at=dispatch_ts,
            message_id="orig-msg",
            subject="Original subject",
        ),
    )
    monkeypatch.setattr(
        supplier_response_repo, "lookup_workflow_for_unique", lambda **_: None
    )

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    received_ts = dispatch_ts + timedelta(minutes=15)
    agent._store_response(
        "wf-123",
        "SUP-123",
        "Appreciate the opportunity",
        {"price": "1200.00", "lead_time": "5"},
        unique_id="line-001",
        rfq_id="RFQ-20240101-ABCD1234",
        message_id="msg-1",
        from_address="supplier@example.com",
        received_at=received_ts,
    )

    assert "row" in captured
    stored = captured["row"]
    assert stored.workflow_id == "wf-123"
    assert stored.unique_id == "line-001"
    assert stored.supplier_id == "SUP-123"
    assert stored.price == Decimal("1200.00")
    assert stored.lead_time == 5
    assert stored.response_message_id == "msg-1"
    assert stored.original_message_id == "orig-msg"
    assert stored.response_time == Decimal("900")


def test_store_response_skips_without_unique_id(monkeypatch, caplog):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)
    monkeypatch.setattr(
        workflow_email_tracking_repo, "lookup_workflow_for_unique", lambda **_: None
    )
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_dispatch_row",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supplier_response_repo, "lookup_workflow_for_unique", lambda **_: None
    )

    called = []

    def fake_insert(row):
        called.append(row)

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    with caplog.at_level(logging.ERROR):
        agent._store_response(
            "wf-123",
            "SUP-123",
            "Missing ids",
            {"price": None, "lead_time": None},
            unique_id=None,
            rfq_id="RFQ-20240101-ABCD1234",
        )

    assert not called
    assert "unique_id" in caplog.text


def test_store_response_aligns_workflow_with_dispatch(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)

    canonical_workflow = "wf-dispatch"

    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_workflow_for_unique",
        lambda **_: canonical_workflow,
    )
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_dispatch_row",
        lambda **_: None,
    )

    monkeypatch.setattr(
        supplier_response_repo,
        "lookup_workflow_for_unique",
        lambda **_: None,
    )

    captured: Dict[str, SupplierResponseRow] = {}

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    agent._store_response(
        "wf-orig",
        "SUP-999",
        "Response body",
        {"price": "4100.00", "lead_time": 12},
        unique_id="uniq-1",
        message_id="msg-123",
    )

    assert "row" in captured
    assert captured["row"].workflow_id == canonical_workflow


def test_store_response_aligns_workflow_with_existing_record(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)

    stored_workflow = "wf-existing"

    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_workflow_for_unique",
        lambda **_: None,
    )
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_dispatch_row",
        lambda **_: None,
    )

    monkeypatch.setattr(
        supplier_response_repo,
        "lookup_workflow_for_unique",
        lambda **_: stored_workflow,
    )

    captured: Dict[str, SupplierResponseRow] = {}

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    agent._store_response(
        "wf-mismatch",
        "SUP-888",
        "Response body",
        {"price": "5250.00", "lead_time": 20},
        unique_id="uniq-2",
        message_id="msg-789",
    )

    assert "row" in captured
    assert captured["row"].workflow_id == stored_workflow


def test_store_response_realigns_existing_unique_ids(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_workflow_for_unique",
        lambda **_: None,
    )
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "lookup_dispatch_row",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supplier_response_repo,
        "lookup_workflow_for_unique",
        lambda **_: None,
    )

    dispatched_ids = ["uniq-alpha", "uniq-beta", None, "  "]
    monkeypatch.setattr(
        workflow_email_tracking_repo,
        "load_workflow_unique_ids",
        lambda **_: dispatched_ids,
    )

    align_calls: Dict[str, Any] = {}

    def fake_align(*, workflow_id: str, unique_ids):
        align_calls["workflow_id"] = workflow_id
        align_calls["unique_ids"] = list(unique_ids)

    monkeypatch.setattr(
        supplier_response_repo,
        "align_workflow_assignments",
        fake_align,
    )

    captured: Dict[str, SupplierResponseRow] = {}

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    agent._store_response(
        "wf-canonical",
        "SUP-007",
        "Body",
        {"price": None, "lead_time": None},
        unique_id="uniq-gamma",
        message_id="msg-999",
    )

    assert align_calls["workflow_id"] == "wf-canonical"
    assert set(align_calls["unique_ids"]) == {"uniq-alpha", "uniq-beta", "uniq-gamma"}
    assert captured["row"].workflow_id == "wf-canonical"


def test_run_await_workflow_batch(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    def fake_expected_unique_ids_and_last_dispatch(*, workflow_id, run_id=None):
        assert workflow_id == "wf-batch"
        return {"uid-1", "uid-2"}, {"uid-1": "SUP-1", "uid-2": "SUP-2"}, datetime.fromtimestamp(0)

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch",
        fake_expected_unique_ids_and_last_dispatch,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.init_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.load_workflow_rows",
        lambda *, workflow_id: [
            SimpleNamespace(
                workflow_id=workflow_id,
                unique_id="uid-1",
                supplier_id="SUP-1",
                message_id="m1",
                subject="RFQ-1",
                dispatched_at=datetime.fromtimestamp(0),
            ),
            SimpleNamespace(
                workflow_id=workflow_id,
                unique_id="uid-2",
                supplier_id="SUP-2",
                message_id="m2",
                subject="RFQ-2",
                dispatched_at=datetime.fromtimestamp(0),
            ),
        ],
    )

    metadata = {
        "rows": [
            SimpleNamespace(
                workflow_id="wf-batch",
                unique_id="uid-1",
                supplier_id="SUP-1",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m1",
            ),
            SimpleNamespace(
                workflow_id="wf-batch",
                unique_id="uid-2",
                supplier_id="SUP-2",
                dispatched_at=datetime.fromtimestamp(0),
                message_id="m2",
            ),
        ],
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema",
        lambda: None,
    )

    def fake_fetch_pending(*, workflow_id, include_processed=False):
        assert include_processed is False
        return [
            {
                "workflow_id": workflow_id,
                "unique_id": "uid-1",
                "supplier_id": "SUP-1",
                "response_text": "Quote A",
                "subject": "Re: RFQ-1",
                "message_id": "m1",
                "from_addr": "one@example.com",
                "received_time": datetime.fromtimestamp(0),
            },
            {
                "workflow_id": workflow_id,
                "unique_id": "uid-2",
                "supplier_id": "SUP-2",
                "response_text": "Quote B",
                "subject": "Re: RFQ-2",
                "message_id": "m2",
                "from_addr": "two@example.com",
                "received_time": datetime.fromtimestamp(1),
            },
        ]

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        fake_fetch_pending,
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.lookup_workflow_for_unique",
        lambda **_: "wf-batch",
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.lookup_workflow_for_unique",
        lambda **_: "wf-batch",
    )
    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", lambda *_args, **_kwargs: None)

    context = AgentContext(
        workflow_id="wf-batch",
        agent_id="supplier_interaction",
        user_id="tester",
        input_data={
            "action": "await_workflow_batch",
            "workflow_id": "wf-batch",
            "response_poll_interval": 1,
            "response_timeout": 5,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["batch_ready"] is True
    assert output.data["expected_responses"] == 2
    assert output.data["collected_responses"] == 2
    assert output.data["supplier_responses_count"] == 2
    assert output.data["negotiation_batch"] is True
    assert output.next_agents == ["NegotiationAgent"]
    assert {resp.get("unique_id") for resp in output.data["supplier_responses"]} == {
        "uid-1",
        "uid-2",
    }


def test_dispatch_gate_requires_all_expected(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch",
        lambda **_: ({"uid-1", "uid-2"}, {"uid-1": "SUP-1"}, datetime.fromtimestamp(0)),
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.init_schema",
        lambda: None,
    )

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.workflow_email_tracking_repo.load_workflow_rows",
        lambda *, workflow_id: [
            SimpleNamespace(
                workflow_id=workflow_id,
                unique_id="uid-1",
                supplier_id="SUP-1",
                message_id="m1",
                dispatched_at=datetime.fromtimestamp(0),
            )
        ],
    )
    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", lambda *_args, **_kwargs: None)

    summary = agent._await_dispatch_gate("wf-batch", timeout=0, poll_interval=0)

    assert summary["complete"] is False
    assert summary["expected_count"] == 2
    assert summary["unique_ids"] == ["uid-1", "uid-2"]


def test_response_gate_requires_all_responses(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.init_schema", lambda: None
    )
    monkeypatch.setattr(
        "agents.supplier_interaction_agent.supplier_response_repo.fetch_pending",
        lambda **_: [
            {
                "workflow_id": "wf-batch",
                "unique_id": "uid-1",
                "supplier_id": "SUP-1",
                "response_text": "Quote",
                "subject": "Re: RFQ",
                "message_id": "m1",
                "from_addr": "one@example.com",
                "received_time": datetime.fromtimestamp(0),
            }
        ],
    )

    summary = agent._await_response_gate(
        "wf-batch",
        expected_unique_ids=["uid-1", "uid-2"],
        expected_count=2,
        timeout=0,
    )

    assert summary["complete"] is False
    assert summary["collected_count"] == 1


def test_await_full_response_batch_requires_complete_population(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    dispatch_rows = [
        SimpleNamespace(
            unique_id="uid-1",
            supplier_id="SUP-1",
            dispatched_at=datetime.fromtimestamp(0),
            message_id="m1",
        ),
        SimpleNamespace(
            unique_id="uid-2",
            supplier_id="SUP-2",
            dispatched_at=datetime.fromtimestamp(0),
            message_id="m2",
        ),
    ]

    metadata = {
        "rows": dispatch_rows,
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    dispatch_calls: List[Dict[str, Any]] = []

    def fake_dispatch_ready(**kwargs):
        dispatch_calls.append(kwargs)
        return {
            "workflow_id": kwargs["workflow_id"],
            "unique_ids": ["uid-1", "uid-2"],
            "expected_dispatches": 2,
            "completed_dispatches": 2,
            "complete": True,
        }

    monkeypatch.setattr(agent, "_await_dispatch_ready", fake_dispatch_ready)
    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)

    poll_calls: List[Dict[str, Any]] = []

    def fake_poll(*args, **kwargs):
        poll_calls.append(dict(kwargs))
        return []

    monkeypatch.setattr(agent, "_poll_supplier_response_rows", fake_poll)

    class DummyCoordinator:
        def __init__(self):
            self.response_calls: List[Dict[str, Any]] = []

        def await_response_population(self, **kwargs):
            self.response_calls.append(kwargs)
            return {
                "workflow_id": kwargs["workflow_id"],
                "expected_responses": 2,
                "completed_responses": 1,
                "complete": False,
                "unique_ids": list(kwargs["unique_ids"]),
            }

    coordinator = DummyCoordinator()
    agent._response_coordinator = coordinator  # type: ignore[assignment]

    result = agent._await_full_response_batch(
        "wf-bulk",
        timeout=5,
        poll_interval=1,
    )

    assert result["ready"] is False
    assert result["expected_count"] == 2
    assert result["collected_count"] == 1
    assert dispatch_calls and dispatch_calls[0]["workflow_id"] == "wf-bulk"
    assert coordinator.response_calls and coordinator.response_calls[0]["wait_for_all"] is True
    assert poll_calls == []


def test_await_full_response_batch_returns_responses_when_ready(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    dispatch_rows = [
        SimpleNamespace(
            unique_id="uid-1",
            supplier_id="SUP-1",
            dispatched_at=datetime.fromtimestamp(0),
            message_id="m1",
        ),
        SimpleNamespace(
            unique_id="uid-2",
            supplier_id="SUP-2",
            dispatched_at=datetime.fromtimestamp(0),
            message_id="m2",
        ),
    ]

    metadata = {
        "rows": dispatch_rows,
        "last_dispatched_at": datetime.fromtimestamp(0),
        "unique_ids": ["uid-1", "uid-2"],
    }

    monkeypatch.setattr(agent, "_await_dispatch_ready", lambda **_: {
        "workflow_id": "wf-bulk",
        "unique_ids": ["uid-1", "uid-2"],
        "expected_dispatches": 2,
        "completed_dispatches": 2,
        "complete": True,
    })
    monkeypatch.setattr(agent, "_load_dispatch_metadata", lambda *_args, **_kwargs: metadata)

    responses = [
        {
            "workflow_id": "wf-bulk",
            "unique_id": "uid-1",
            "supplier_id": "SUP-1",
            "response_text": "Quote 1",
            "subject": "Re: RFQ-1",
            "message_id": "m1",
            "from_addr": "one@example.com",
            "received_time": datetime.fromtimestamp(1),
        },
        {
            "workflow_id": "wf-bulk",
            "unique_id": "uid-2",
            "supplier_id": "SUP-2",
            "response_text": "Quote 2",
            "subject": "Re: RFQ-2",
            "message_id": "m2",
            "from_addr": "two@example.com",
            "received_time": datetime.fromtimestamp(2),
        },
    ]

    poll_calls: List[Dict[str, Any]] = []

    def fake_poll(*args, **kwargs):
        poll_calls.append(dict(kwargs))
        return responses

    monkeypatch.setattr(agent, "_poll_supplier_response_rows", fake_poll)
    monkeypatch.setattr(agent, "_process_responses_concurrently", lambda rows: list(rows))

    class DummyCoordinator:
        def __init__(self):
            self.response_calls: List[Dict[str, Any]] = []

        def await_response_population(self, **kwargs):
            self.response_calls.append(kwargs)
            return {
                "workflow_id": kwargs["workflow_id"],
                "expected_responses": 2,
                "completed_responses": 2,
                "complete": True,
                "unique_ids": list(kwargs["unique_ids"]),
            }

    coordinator = DummyCoordinator()
    agent._response_coordinator = coordinator  # type: ignore[assignment]

    result = agent._await_full_response_batch(
        "wf-bulk",
        timeout=5,
        poll_interval=1,
    )

    assert result["ready"] is True
    assert result["expected_count"] == 2
    assert result["collected_count"] == 2
    assert {resp.get("unique_id") for resp in result["responses"]} == {"uid-1", "uid-2"}
    assert coordinator.response_calls and coordinator.response_calls[0]["wait_for_all"] is True
    assert poll_calls and poll_calls[0]["include_processed"] is False
