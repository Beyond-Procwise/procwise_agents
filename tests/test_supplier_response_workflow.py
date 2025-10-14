import os
import sys
from types import SimpleNamespace
from collections import deque

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.supplier_response_workflow import SupplierResponseWorkflow


class DummyWorkflowRepo:
    def __init__(self, batches):
        self._batches = deque(batches)
        self.init_called = 0
        self.load_calls = []
        self._last = []

    def init_schema(self):
        self.init_called += 1

    def load_workflow_rows(self, workflow_id):
        self.load_calls.append(workflow_id)
        if self._batches:
            self._last = self._batches.popleft()
            return self._last
        return list(self._last)


class DummyResponseRepo:
    def __init__(self, batches):
        self._batches = deque(batches)
        self.init_called = 0
        self.fetch_calls = []

    def init_schema(self):
        self.init_called += 1

    def fetch_all(self, workflow_id):
        self.fetch_calls.append(workflow_id)
        if self._batches:
            return self._batches.popleft()
        return []


class SleepRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def test_workflow_waits_for_dispatch_and_responses():
    workflow_id = "wf-123"
    unique_ids = ["uid-1", "uid-2"]

    dispatch_batches = [
        [SimpleNamespace(unique_id="uid-1", dispatched_at=None, message_id=None)],
        [
            SimpleNamespace(unique_id="uid-1", dispatched_at=1, message_id="m1"),
            SimpleNamespace(unique_id="uid-2", dispatched_at=1, message_id="m2"),
        ],
    ]
    response_batches = [
        [{"unique_id": "uid-1"}],
        [{"unique_id": "uid-1"}, {"unique_id": "uid-2"}],
    ]

    workflow_repo = DummyWorkflowRepo(dispatch_batches)
    response_repo = DummyResponseRepo(response_batches)
    sleep = SleepRecorder()

    helper = SupplierResponseWorkflow(
        workflow_repo=workflow_repo,
        response_repo=response_repo,
        sleep_fn=sleep,
    )

    summary = helper.ensure_ready(
        workflow_id=workflow_id,
        unique_ids=unique_ids,
        dispatch_timeout=10,
        dispatch_poll_interval=1,
        response_timeout=10,
        response_poll_interval=2,
    )

    activation = summary["activation"]
    assert activation["activated"] is True
    assert activation["first_unique_id"] in {"uid-1", "uid-2"}

    dispatch = summary["dispatch"]
    assert dispatch["completed_dispatches"] == 2
    assert dispatch["complete"] is True

    responses = summary["responses"]
    assert responses["completed_responses"] == 1
    assert responses["complete"] is False
    assert workflow_repo.init_called == 1
    assert response_repo.init_called == 1
    # First loop sleeps twice (once for dispatch, once for responses)
    assert sleep.calls and all(call in (0, 1, 2) for call in sleep.calls)


def test_workflow_raises_when_dispatch_incomplete():
    workflow_repo = DummyWorkflowRepo(
        [[SimpleNamespace(unique_id="uid-1", dispatched_at=None, message_id=None)]]
    )
    response_repo = DummyResponseRepo([[{"unique_id": "uid-1"}]])

    helper = SupplierResponseWorkflow(
        workflow_repo=workflow_repo,
        response_repo=response_repo,
        sleep_fn=lambda *_: None,
    )

    with pytest.raises(TimeoutError):
        helper.ensure_ready(
            workflow_id="wf-timeout",
            unique_ids=["uid-1", "uid-2"],
            dispatch_timeout=0,
            dispatch_poll_interval=0,
            response_timeout=0,
            response_poll_interval=0,
        )


def test_workflow_snapshot_returns_partial_response_progress():
    dispatch_batches = [
        [
            SimpleNamespace(unique_id="uid-1", dispatched_at=1, message_id="m1"),
            SimpleNamespace(unique_id="uid-2", dispatched_at=1, message_id="m2"),
        ]
    ]
    response_batches = [[{"unique_id": "uid-1"}]]

    helper = SupplierResponseWorkflow(
        workflow_repo=DummyWorkflowRepo(dispatch_batches),
        response_repo=DummyResponseRepo(response_batches),
        sleep_fn=lambda *_: None,
    )

    summary = helper.ensure_ready(
        workflow_id="wf-no-responses",
        unique_ids=["uid-1", "uid-2"],
        dispatch_timeout=0,
        dispatch_poll_interval=0,
        response_timeout=0,
        response_poll_interval=0,
    )

    assert summary["activation"]["activated"] is True
    assert summary["dispatch"]["complete"] is True
    assert summary["responses"]["complete"] is False
