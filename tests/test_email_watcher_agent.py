from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentContext, AgentStatus
from agents.email_watcher_agent import EmailWatcherAgent
from services.imap_supplier_response_watcher import DatabaseBackend
from services.email_watcher_v2 import EmailResponse
from utils.email_tracking import embed_unique_id_in_email_body, generate_unique_email_id


def _prepare_dispatch_table(db_path: str, entries):
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS "proc.email_dispatch_chains" ('
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "rfq_id TEXT,"
        "message_id TEXT,"
        "thread_index INTEGER,"
        "supplier_id TEXT,"
        "workflow_ref TEXT,"
        "recipients TEXT,"
        "subject TEXT,"
        "body TEXT,"
        "dispatch_metadata TEXT,"
        "awaiting_response INTEGER,"
        "responded_at TEXT,"
        "response_message_id TEXT,"
        "response_metadata TEXT,"
        "created_at TEXT,"
        "updated_at TEXT"
        ")"
    )
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    for entry in entries:
        metadata = json.dumps(entry["metadata"])
        conn.execute(
            'INSERT INTO "proc.email_dispatch_chains" '
            "(rfq_id, message_id, workflow_ref, dispatch_metadata, awaiting_response, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 1, ?, ?)",
            (
                entry["rfq_id"],
                entry.get("message_id"),
                entry.get("action_id"),
                metadata,
                now.isoformat(),
                now.isoformat(),
            ),
        )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _sqlite_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for env in ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]:
        monkeypatch.delenv(env, raising=False)
    yield
    if os.path.exists("procwise_dev.sqlite"):
        os.remove("procwise_dev.sqlite")


class DummySettings:
    email_response_poll_seconds = 1
    email_inbound_initial_wait_seconds = 0
    imap_mailbox = "INBOX"
    imap_host = "imap.example.com"
    imap_username = "user@example.com"
    imap_password = "secret"
    imap_port = 993
    imap_login = None


class DummyAgentNick:
    def __init__(self) -> None:
        self.settings = DummySettings()


class StubFetcher:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, *, since):
        self.calls += 1
        if self.calls == 1:
            return list(self._responses)
        return []

    def wait(self):
        return None


def test_email_watcher_agent_persists_matches(tmp_path):
    workflow_id = "wf-email-watcher"
    round_number = 1
    dispatch_run_id = "round-1"
    DatabaseBackend().ensure_schema()

    unique_ids = [generate_unique_email_id(workflow_id, f"supplier-{idx}") for idx in (1, 2)]
    dispatch_entries = []
    for idx, unique_id in enumerate(unique_ids, start=1):
        dispatch_entries.append(
            {
                "rfq_id": f"RFQ-2024-00{idx}",
                "message_id": f"<dispatch-{idx}>",
                "action_id": str(round_number),
                "metadata": {
                    "workflow_id": workflow_id,
                    "run_id": dispatch_run_id,
                    "supplier_id": f"supplier-{idx}",
                    "unique_id": unique_id,
                },
            }
        )
    _prepare_dispatch_table("procwise_dev.sqlite", dispatch_entries)

    base_time = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    responses = []
    for idx, unique_id in enumerate(unique_ids, start=1):
        body = embed_unique_id_in_email_body("Thank you for the opportunity", unique_id)
        responses.append(
            EmailResponse(
                unique_id=unique_id,
                supplier_id=f"supplier-{idx}",
                supplier_email=f"supplier{idx}@example.com",
                from_address=f"supplier{idx}@example.com",
                message_id=f"<reply-{idx}>",
                subject="Re: Pricing request",
                body=body,
                body_text=body,
                body_html=None,
                received_at=base_time + timedelta(minutes=idx),
                in_reply_to=(f"<dispatch-{idx}>",),
                references=(f"<dispatch-{idx}>",),
                workflow_id=workflow_id,
                rfq_id=f"RFQ-2024-00{idx}",
                headers={"In-Reply-To": (f"<dispatch-{idx}>",)},
            )
        )

    # Add an unmatched response that should be flagged for review.
    unmatched = EmailResponse(
        unique_id="PROC-WF-UNKNOWN",
        supplier_id="supplier-x",
        supplier_email="other@example.com",
        from_address="other@example.com",
        message_id="<reply-x>",
        subject="Re: Something else",
        body="Hello",
        body_text="Hello",
        body_html=None,
        received_at=base_time + timedelta(minutes=10),
        in_reply_to=("<unknown>",),
        references=(),
        workflow_id=workflow_id,
        rfq_id="RFQ-UNKNOWN",
        headers={}
    )
    fetcher = StubFetcher(responses + [unmatched])

    agent_nick = DummyAgentNick()
    current_time = base_time

    def _fake_now() -> datetime:
        nonlocal current_time
        moment = current_time
        current_time = current_time + timedelta(seconds=5)
        return moment

    agent = EmailWatcherAgent(agent_nick, now=_fake_now, sleep=lambda _: None)

    context = AgentContext(
        workflow_id=workflow_id,
        agent_id="EmailWatcherAgent",
        user_id="tester",
        input_data={
            "round": round_number,
            "dispatch_run_id": dispatch_run_id,
            "expected_responses": 2,
            "poll_seconds": 1,
            "initial_wait_seconds": 0,
            "email_fetcher": fetcher,
            "match_threshold": 0.6,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    data = output.data
    assert data["responses_received"] == 2
    assert data["expected_responses"] == 2
    assert not data["timeout"]
    assert fetcher.calls >= 1

    conn = sqlite3.connect("procwise_dev.sqlite")
    rows = conn.execute(
        'SELECT rfq_id, supplier_id, match_method, match_score FROM "proc.supplier_responses"'
    ).fetchall()
    conn.close()

    assert len(rows) == 3
    accepted = [row for row in rows if row[2] == "accepted"]
    review = [row for row in rows if row[2] == "needs_review"]
    assert len(accepted) == 2
    assert len(review) == 1
    assert all(score is not None for *_, score in accepted)
