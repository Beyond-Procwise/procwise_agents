import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import services.imap_supplier_response_watcher as watcher  # noqa: E402


def _prepare_sqlite_dispatch_table(db_path, entries):
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


def _make_record(rfq_id, *, token, received=None, supplier="SI-1"):
    received_at = received or datetime.now(timezone.utc)
    return watcher.SupplierResponseRecord(
        workflow_id="wf-1",
        action_id="act-1",
        run_id=token,
        rfq_id=rfq_id,
        supplier_id=supplier,
        message_id=f"<{rfq_id}-msg>",
        subject=f"RFQ {rfq_id}",
        body=f"<!-- PROCWISE:UID={rfq_id};SUPPLIER={supplier};TOKEN={token};RUN_ID={token} -->",
        from_address="supplier@example.com",
        received_at=received_at,
        headers={"In-Reply-To": "<draft-msg>"},
        mailbox="INBOX",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setenv("IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("IMAP_USERNAME", "user")
    monkeypatch.setenv("IMAP_PASSWORD", "pass")
    yield
    if os.path.exists("procwise_dev.sqlite"):
        os.remove("procwise_dev.sqlite")


def test_run_watcher_processes_latest_per_rfq(monkeypatch):
    _prepare_sqlite_dispatch_table(
        "procwise_dev.sqlite",
        [
            {
                "rfq_id": "RFQ-1",
                "action_id": "act-1",
                "metadata": {"run_id": "token-1", "workflow_id": "wf-1", "supplier_id": "SI-1"},
            },
            {
                "rfq_id": "RFQ-1",
                "action_id": "act-1",
                "metadata": {"run_id": "token-2", "workflow_id": "wf-1", "supplier_id": "SI-2"},
            },
        ],
    )

    latest = datetime.now(timezone.utc)
    earlier = latest - timedelta(minutes=1)
    records = [
        _make_record("RFQ-1", token="token-1", received=earlier, supplier="SI-1"),
        _make_record("RFQ-1", token="token-2", received=latest, supplier="SI-2"),
    ]
    processed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        watcher,
        "_collect_imap_messages",
        lambda **_: records,
    )

    result = watcher.run_imap_supplier_response_watcher(
        agent_nick=SimpleNamespace(),
        workflow_id="wf-1",
        action_id="act-1",
        run_id=None,
        process_callback=lambda payload: processed.append((payload["rfq_id"], payload["run_id"])),
        max_workers=2,
    )

    assert result["persisted"] == 2
    assert result["processed"] == 2
    assert sorted(result["expected_tokens"]) == ["token-1", "token-2"]
    assert sorted(result["inbound_tokens"]) == ["token-1", "token-2"]
    assert sorted(processed) == [("RFQ-1", "token-1"), ("RFQ-1", "token-2")]

    conn = sqlite3.connect("procwise_dev.sqlite")
    rows = conn.execute(
        'SELECT rfq_id, supplier_id, processed_at FROM "proc.supplier_responses"'
    ).fetchall()
    conn.close()
    assert any(row[0] == "RFQ-1" and row[1] == "SI-1" and row[2] is not None for row in rows)
    assert any(row[0] == "RFQ-1" and row[1] == "SI-2" and row[2] is not None for row in rows)


def test_run_watcher_halts_when_gate_mismatched(monkeypatch):
    _prepare_sqlite_dispatch_table(
        "procwise_dev.sqlite",
        [
            {
                "rfq_id": "RFQ-1",
                "action_id": "act-1",
                "metadata": {"run_id": "token-1", "workflow_id": "wf-1", "supplier_id": "SI-1"},
            },
            {
                "rfq_id": "RFQ-2",
                "action_id": "act-1",
                "metadata": {"run_id": "token-2", "workflow_id": "wf-1", "supplier_id": "SI-2"},
            },
        ],
    )

    records = [_make_record("RFQ-1", token="token-1")]
    monkeypatch.setattr(
        watcher,
        "_collect_imap_messages",
        lambda **_: records,
    )

    processed = []
    result = watcher.run_imap_supplier_response_watcher(
        agent_nick=SimpleNamespace(),
        workflow_id="wf-1",
        action_id="act-1",
        run_id=None,
        process_callback=lambda payload: processed.append(payload["rfq_id"]),
    )

    assert result["processed"] == 0
    assert sorted(result["expected_tokens"]) == ["token-1", "token-2"]
    assert sorted(result["inbound_tokens"]) == ["token-1"]
    assert processed == []

    conn = sqlite3.connect("procwise_dev.sqlite")
    row = conn.execute(
        'SELECT processed_at FROM "proc.supplier_responses" WHERE rfq_id = ?',
        ("RFQ-1",),
    ).fetchone()
    conn.close()
    assert row[0] is None


def test_run_watcher_recovers_rfq_via_dispatch_headers(monkeypatch):
    _prepare_sqlite_dispatch_table(
        "procwise_dev.sqlite",
        [
            {
                "rfq_id": "RFQ-2",
                "action_id": "act-headers",
                "message_id": "<dispatch-headers>",
                "metadata": {
                    "run_id": "token-headers",
                    "workflow_id": "wf-headers",
                    "supplier_id": "SI-777",
                },
            }
        ],
    )

    def _fake_collect(**kwargs):
        dispatch_context = kwargs.get("dispatch_context")
        msg = EmailMessage()
        msg["Subject"] = "Re: Quote"
        msg["From"] = "supplier@example.com"
        msg["In-Reply-To"] = "<dispatch-headers>"
        msg.set_content("Response body without marker")
        record = watcher._extract_rfq_payload(
            msg,
            workflow_id=kwargs.get("workflow_id"),
            action_id=kwargs.get("action_id"),
            run_id=kwargs.get("run_id"),
            mailbox="INBOX",
            dispatch_context=dispatch_context,
        )
        return [record] if record else []

    monkeypatch.setattr(watcher, "_collect_imap_messages", _fake_collect)

    processed = []
    result = watcher.run_imap_supplier_response_watcher(
        agent_nick=SimpleNamespace(),
        workflow_id="wf-headers",
        action_id="act-headers",
        run_id=None,
        process_callback=lambda payload: processed.append((payload["rfq_id"], payload["supplier_id"], payload["run_id"])),
    )

    assert result["processed"] == 1
    assert result["expected_tokens"] == ["token-headers"]
    assert result["inbound_tokens"] == ["token-headers"]
    assert processed == [("RFQ-2", "SI-777", "token-headers")]

    conn = sqlite3.connect("procwise_dev.sqlite")
    row = conn.execute(
        'SELECT supplier_id, run_id FROM "proc.supplier_responses" WHERE rfq_id = ?',
        ("RFQ-2",),
    ).fetchone()
    conn.close()
    assert row == ("SI-777", "token-headers")
