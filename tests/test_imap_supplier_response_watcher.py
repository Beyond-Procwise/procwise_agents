import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
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
            "(rfq_id, workflow_ref, dispatch_metadata, awaiting_response, created_at, updated_at)"
            " VALUES (?, ?, ?, 1, ?, ?)",
            (entry["rfq_id"], entry.get("action_id"), metadata, now.isoformat(), now.isoformat()),
        )
    conn.commit()
    conn.close()


def _make_record(rfq_id, *, run_id, received=None, supplier="SI-1"):
    received_at = received or datetime.now(timezone.utc)
    return watcher.SupplierResponseRecord(
        workflow_id="wf-1",
        action_id="act-1",
        run_id=run_id,
        rfq_id=rfq_id,
        supplier_id=supplier,
        message_id=f"<{rfq_id}-msg>",
        subject=f"RFQ {rfq_id}",
        body=f"<!-- PROCWISE:RFQ_ID={rfq_id};SUPPLIER={supplier};TOKEN=t;RUN_ID={run_id} -->",
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
            {"rfq_id": "RFQ-1", "action_id": "act-1", "metadata": {"run_id": "run-1", "workflow_id": "wf-1"}},
        ],
    )

    latest = datetime.now(timezone.utc)
    earlier = latest - timedelta(minutes=1)
    records = [
        _make_record("RFQ-1", run_id="run-1", received=earlier),
        _make_record("RFQ-1", run_id="run-1", received=latest, supplier="SI-2"),
    ]
    processed = []

    monkeypatch.setattr(
        watcher,
        "_collect_imap_messages",
        lambda **_: records,
    )

    result = watcher.run_imap_supplier_response_watcher(
        agent_nick=SimpleNamespace(),
        workflow_id="wf-1",
        action_id="act-1",
        run_id="run-1",
        process_callback=lambda payload: processed.append(payload["rfq_id"]),
        max_workers=2,
    )

    assert result["persisted"] == 2
    assert result["processed"] == 1
    assert processed == ["RFQ-1"]

    conn = sqlite3.connect("procwise_dev.sqlite")
    rows = conn.execute(
        'SELECT rfq_id, supplier_id, processed_at FROM "proc.supplier_responses"'
    ).fetchall()
    conn.close()
    assert any(row[0] == "RFQ-1" and row[1] == "SI-2" and row[2] is not None for row in rows)


def test_run_watcher_halts_when_gate_mismatched(monkeypatch):
    _prepare_sqlite_dispatch_table(
        "procwise_dev.sqlite",
        [
            {"rfq_id": "RFQ-1", "action_id": "act-1", "metadata": {"run_id": "run-1", "workflow_id": "wf-1"}},
            {"rfq_id": "RFQ-2", "action_id": "act-1", "metadata": {"run_id": "run-1", "workflow_id": "wf-1"}},
        ],
    )

    records = [_make_record("RFQ-1", run_id="run-1")]
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
        run_id="run-1",
        process_callback=lambda payload: processed.append(payload["rfq_id"]),
    )

    assert result["processed"] == 0
    assert result["dispatched_rfqs"] == ["RFQ-1", "RFQ-2"]
    assert result["inbound_rfqs"] == ["RFQ-1"]
    assert processed == []

    conn = sqlite3.connect("procwise_dev.sqlite")
    row = conn.execute(
        'SELECT processed_at FROM "proc.supplier_responses" WHERE rfq_id = ?',
        ("RFQ-1",),
    ).fetchone()
    conn.close()
    assert row[0] is None
