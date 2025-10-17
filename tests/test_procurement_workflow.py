import os
import sys
from pathlib import Path

import asyncio
import pytest

sys.path.append(os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..")))

from orchestration.procurement_workflow import (
    MockDatabaseConnection,
    WorkflowOrchestrator,
    handle_supplier_response,
    validate_implementation,
)


def test_validate_implementation_smoke():
    asyncio.run(validate_implementation())


def test_wait_for_responses_times_out_when_missing_supplier_reply():
    async def runner():
        db = MockDatabaseConnection()
        orchestrator = WorkflowOrchestrator(db=db, wait_timeout=0.05)
        orchestrator.check_interval = 0.01
        await orchestrator._initialize_response_tracking(round_num=0, expected_count=1)

        with pytest.raises(TimeoutError):
            await orchestrator.wait_for_responses(1, "missing response", round_num=0)

    asyncio.run(runner())


def test_handle_supplier_response_marks_round_complete():
    async def runner():
        db = MockDatabaseConnection()
        orchestrator = WorkflowOrchestrator(db=db, wait_timeout=0.1)
        orchestrator.check_interval = 0.01
        await orchestrator._initialize_response_tracking(round_num=0, expected_count=1)

        await handle_supplier_response(
            db,
            orchestrator.session_id,
            0,
            "SUP001",
            "PROC-WF-TRACK",
            {"content": "hello"},
        )

        responses = await orchestrator.wait_for_responses(
            1, "response sync", round_num=0
        )
        assert responses["SUP001"]["response_data"]["content"] == "hello"

    asyncio.run(runner())
