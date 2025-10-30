import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import repositories.workflow_email_tracking_repo as tracking_repo

from repositories.workflow_email_tracking_repo import (
    WorkflowDispatchRow,
    _ensure_primary_key,
    init_schema,
    load_active_workflow_ids,
    load_workflow_rows,
    mark_response,
    record_dispatches,
    reset_workflow,
)


def _install_expected_map(monkeypatch) -> Dict[str, Set[str]]:
    expected_map: Dict[str, Set[str]] = {}

    def _fake_expected_unique_ids_and_last_dispatch(*, workflow_id: str, run_id=None):
        ids = expected_map.get(workflow_id, set())
        return (set(ids), {}, None)

    monkeypatch.setattr(
        tracking_repo.draft_rfq_emails_repo,
        "expected_unique_ids_and_last_dispatch",
        _fake_expected_unique_ids_and_last_dispatch,
    )
    return expected_map


def test_ensure_primary_key_rebuilds_legacy_constraint():
    executed = []

    class FakeCursor:
        def __init__(self) -> None:
            self._result_one = None
            self._result_all = None

        def execute(self, query, params=None):
            executed.append(" ".join(query.split()))
            if "to_regclass" in query:
                self._result_one = ("proc.workflow_email_tracking",)
                self._result_all = None
            elif "constraint_name" in query and "PRIMARY KEY" in query:
                self._result_one = None
                self._result_all = [
                    ("workflow_email_tracking_pkey", "workflow_id", 1),
                    ("workflow_email_tracking_pkey", "unique_id", 2),
                    ("workflow_email_tracking_pkey", "dispatch_key", 3),
                ]
            else:
                self._result_one = None
                self._result_all = None

        def fetchone(self):
            result = self._result_one
            self._result_one = None
            return result

        def fetchall(self):
            result = list(self._result_all or [])
            self._result_all = None
            return result

    cursor = FakeCursor()
    _ensure_primary_key(cursor)

    drop_statements = [stmt for stmt in executed if "DROP CONSTRAINT" in stmt]
    add_statements = [stmt for stmt in executed if "ADD PRIMARY KEY" in stmt]

    assert drop_statements, executed
    assert add_statements, executed


def _make_dispatch(
    *,
    workflow_id: str,
    unique_id: str,
    message_id: Optional[str],
    dispatched_at: datetime,
    responded_at: Optional[datetime] = None,
    matched: bool = False,
    dispatch_key: Optional[str] = None,
) -> WorkflowDispatchRow:
    return WorkflowDispatchRow(
        workflow_id=workflow_id,
        unique_id=unique_id,
        dispatched_at=dispatched_at,
        supplier_id="supplier-a",
        supplier_email="buyer@example.com",
        message_id=message_id,
        subject="Subject",
        dispatch_key=dispatch_key,
        responded_at=responded_at,
        response_message_id=None,
        matched=matched,
        thread_headers=None,
    )


def test_active_workflows_skip_dispatches_without_message_id(monkeypatch):
    workflow_id = "wf-init"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    expected_map = _install_expected_map(monkeypatch)
    expected_map[workflow_id] = {"uid-init"}

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-init",
                message_id=None,
                dispatched_at=now,
            )
        ],
    )

    rows = load_workflow_rows(workflow_id=workflow_id)
    assert rows and rows[0].message_id is None
    assert workflow_id not in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)


def test_dispatch_key_optional_and_persisted(monkeypatch):
    workflow_id = "wf-dispatch-key"
    unique_id = "uid-dk-1"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    expected_map = _install_expected_map(monkeypatch)
    expected_map[workflow_id] = {unique_id}

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id=unique_id,
                message_id="mid-dk",
                dispatched_at=now,
                dispatch_key="run-123",
            )
        ],
    )

    rows = load_workflow_rows(workflow_id=workflow_id)
    assert rows
    assert rows[0].dispatch_key == "run-123"

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-1",
                message_id=None,
                dispatched_at=now,
            )
        ],
    )
    expected_map[workflow_id] = {"uid-1"}
    assert workflow_id not in load_active_workflow_ids()

    # Updating the dispatch metadata keeps the workflow active.
    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-1",
                message_id="mid-1",
                dispatched_at=now,
            )
        ],
    )
    assert workflow_id in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)


def test_active_workflows_require_all_dispatches_to_complete(monkeypatch):
    workflow_id = "wf-partial"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    expected_map = _install_expected_map(monkeypatch)
    expected_map[workflow_id] = {"uid-1", "uid-2"}

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-1",
                message_id="mid-1",
                dispatched_at=now,
            ),
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-2",
                message_id="mid-2",
                dispatched_at=now,
            ),
        ],
    )
    assert workflow_id in load_active_workflow_ids()

    mark_response(
        workflow_id=workflow_id,
        unique_id="uid-2",
        responded_at=now,
        response_message_id="resp-2",
    )
    assert workflow_id in load_active_workflow_ids()

    mark_response(
        workflow_id=workflow_id,
        unique_id="uid-1",
        responded_at=now,
        response_message_id="resp-1",
    )
    assert workflow_id not in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)


def test_active_workflows_skip_when_all_responses_matched(monkeypatch):
    workflow_id = "wf-complete"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    expected_map = _install_expected_map(monkeypatch)
    expected_map[workflow_id] = {"uid-1"}

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-1",
                message_id="mid-1",
                dispatched_at=now,
            )
        ],
    )
    assert workflow_id in load_active_workflow_ids()

    mark_response(
        workflow_id=workflow_id,
        unique_id="uid-1",
        responded_at=now,
        response_message_id="resp-1",
    )
    assert workflow_id not in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)


def test_active_workflows_skip_without_expected_unique_ids(monkeypatch):
    workflow_id = "wf-missing-expected"
    unique_id = "uid-missing"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    expected_map = _install_expected_map(monkeypatch)

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id=unique_id,
                message_id="mid-1",
                dispatched_at=now,
            )
        ],
    )

    assert workflow_id not in load_active_workflow_ids()

    expected_map[workflow_id] = {unique_id}
    assert workflow_id in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)
    expected_map.pop(workflow_id, None)
