import json
import os
import sys
import threading
from typing import Dict, List, Optional
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supplier_interaction_agent import SupplierInteractionAgent
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


class DummyCursor:
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


def test_supplier_interaction_wait_for_response():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    calls = {"count": 0}

    def poll_once(limit=None, match_filters=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {
                    "rfq_id": "RFQ-20240101-abcd1234",
                    "supplier_id": "S1",
                    "negotiation_output": {"message": "counter"},
                }
            ]
        return []

    watcher = SimpleNamespace(poll_once=poll_once)

    result = agent.wait_for_response(watcher=watcher, timeout=1, poll_interval=0)

    assert result is not None
    assert result["rfq_id"] == "RFQ-20240101-abcd1234"
    assert calls["count"] == 1


def test_wait_for_response_waits_until_payload_ready(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    batches = [
        [
            {
                "rfq_id": "RFQ-20240101-abcd1234",
                "supplier_id": "S1",
                "supplier_status": "processing",
                "supplier_output": None,
            }
        ],
        [
            {
                "rfq_id": "RFQ-20240101-abcd1234",
                "supplier_id": "S1",
                "supplier_status": "success",
                "supplier_output": {"price": 900},
            }
        ],
    ]

    calls = {"count": 0}

    def poll_once(limit=None, match_filters=None):
        calls["count"] += 1
        return batches.pop(0) if batches else []

    watcher = SimpleNamespace(poll_once=poll_once)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", lambda *_args, **_kwargs: None)

    result = agent.wait_for_response(
        watcher=watcher,
        timeout=5,
        poll_interval=0,
        rfq_id="RFQ-20240101-abcd1234",
        supplier_id="S1",
    )

    assert result is not None
    assert result["supplier_status"] == "success"
    assert result["supplier_output"]["price"] == 900
    assert calls["count"] == 2

def test_supplier_interaction_wait_for_response_respects_attempt_limit(monkeypatch):
    nick = DummyNick()
    nick.settings.email_response_max_attempts = 3
    agent = SupplierInteractionAgent(nick)

    calls = {"count": 0}

    def poll_once(limit=None, match_filters=None):
        calls["count"] += 1
        return []

    watcher = SimpleNamespace(poll_once=poll_once)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", lambda *_args, **_kwargs: None)

    result = agent.wait_for_response(
        watcher=watcher,
        timeout=30,
        poll_interval=None,
        rfq_id="RFQ-20240101-missing",
        max_attempts=3,
    )

    assert result is None
    assert calls["count"] == 3


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


def test_wait_for_multiple_responses_starts_parallel_watchers(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    created_watchers = []

    class DummyWatcher:
        def __init__(self, *args, **kwargs):
            created_watchers.append(kwargs)

    monkeypatch.setattr(
        "services.email_watcher.SESEmailWatcher",
        DummyWatcher,
    )

    calls = []

    def fake_wait_for_response(
        self,
        *,
        watcher=None,
        timeout: int,
        poll_interval: Optional[int] = None,
        limit: int = 1,
        rfq_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        **kwargs,
    ):
        assert watcher is not None
        calls.append((threading.current_thread().name, rfq_id, supplier_id))
        return {"rfq_id": rfq_id, "supplier_id": supplier_id}

    monkeypatch.setattr(
        SupplierInteractionAgent,
        "wait_for_response",
        fake_wait_for_response,
        raising=False,
    )

    drafts = [
        {
            "rfq_id": "RFQ-20240101-AAA11111",
            "supplier_id": "SUP-1",
            "recipients": ["one@example.com"],
            "subject": "Re: RFQ-20240101-AAA11111",
        },
        {
            "rfq_id": "RFQ-20240101-BBB22222",
            "supplier_id": "SUP-2",
            "receiver": "two@example.com",
            "subject": "Re: RFQ-20240101-BBB22222",
        },
    ]

    results = agent.wait_for_multiple_responses(
        drafts,
        timeout=5,
        poll_interval=1,
        limit=1,
    )

    assert len(results) == len(drafts)
    assert created_watchers and len(created_watchers) == len(drafts)
    thread_names = {name for name, _rfq, _sup in calls}
    assert thread_names == {f"supplier-watch-{idx}" for idx in range(len(drafts))}
    assert all(
        result and result["rfq_id"] == draft["rfq_id"]
        for result, draft in zip(results, drafts)
    )


def test_wait_for_response_initialises_watcher_with_negotiation_toggle(monkeypatch):
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    captured_kwargs: List[Dict[str, object]] = []

    class DummyWatcher:
        def __init__(self, *args, **kwargs):
            self.enable_negotiation = kwargs.get("enable_negotiation")
            self.negotiation_agent = kwargs.get("negotiation_agent")
            captured_kwargs.append(kwargs)

        def poll_once(self, limit=None, match_filters=None):
            return []

    monkeypatch.setattr(
        "services.email_watcher.SESEmailWatcher",
        DummyWatcher,
    )

    agent._negotiation_agent = SimpleNamespace(
        execute=lambda ctx: AgentOutput(status=AgentStatus.SUCCESS, data={})
    )

    agent.wait_for_response(
        timeout=0,
        poll_interval=0,
        rfq_id="RFQ-20240101-AAA11111",
        supplier_id="SUP-1",
    )

    assert captured_kwargs
    assert captured_kwargs[-1]["enable_negotiation"] is True
    assert (
        captured_kwargs[-1]["negotiation_agent"] is agent._negotiation_agent
    )

    captured_kwargs.clear()

    agent.wait_for_response(
        timeout=0,
        poll_interval=0,
        rfq_id="RFQ-20240101-AAA11111",
        supplier_id="SUP-1",
        enable_negotiation=False,
    )

    assert captured_kwargs
    assert captured_kwargs[-1]["enable_negotiation"] is False
    assert captured_kwargs[-1]["negotiation_agent"] is None


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

    def fake_store(rfq_id, supplier_id, message, parsed):
        stored.append(
            {
                "rfq_id": rfq_id,
                "supplier_id": supplier_id,
                "message": message,
                "parsed": parsed,
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
