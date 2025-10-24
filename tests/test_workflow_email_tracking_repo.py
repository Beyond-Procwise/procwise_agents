from datetime import datetime, timezone
from typing import Optional

from repositories.workflow_email_tracking_repo import (
    WorkflowDispatchRow,
    init_schema,
    load_active_workflow_ids,
    mark_response,
    record_dispatches,
    reset_workflow,
)


def _make_dispatch(
    *,
    workflow_id: str,
    unique_id: str,
    message_id: Optional[str],
    dispatched_at: datetime,
    responded_at: Optional[datetime] = None,
    matched: bool = False,
) -> WorkflowDispatchRow:
    return WorkflowDispatchRow(
        workflow_id=workflow_id,
        unique_id=unique_id,
        supplier_id="supplier-a",
        supplier_email="buyer@example.com",
        message_id=message_id,
        subject="Subject",
        dispatched_at=dispatched_at,
        responded_at=responded_at,
        response_message_id=None,
        matched=matched,
        thread_headers=None,
    )


def test_active_workflows_wait_until_dispatch_metadata_complete():
    workflow_id = "wf-init"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

    # Initial row without outbound metadata should not trigger the watcher.
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
    assert workflow_id not in load_active_workflow_ids()

    # Once dispatch metadata is populated the workflow becomes active.
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


def test_active_workflows_require_all_dispatches_to_complete():
    workflow_id = "wf-partial"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

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
                message_id=None,
                dispatched_at=now,
            ),
        ],
    )
    assert workflow_id not in load_active_workflow_ids()

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            _make_dispatch(
                workflow_id=workflow_id,
                unique_id="uid-2",
                message_id="mid-2",
                dispatched_at=now,
            )
        ],
    )
    assert workflow_id in load_active_workflow_ids()

    reset_workflow(workflow_id=workflow_id)


def test_active_workflows_skip_when_all_responses_matched():
    workflow_id = "wf-complete"
    now = datetime.now(timezone.utc)

    init_schema()
    reset_workflow(workflow_id=workflow_id)

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
