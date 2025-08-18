import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import workflows


class DummyOrchestrator:
    def __init__(self):
        self.last_input = None

    def execute_workflow(self, workflow_name, input_data, user_id=None):
        self.last_input = input_data
        return {
            "status": "completed",
            "workflow_id": "wf1",
            "result": input_data,
            "execution_path": [],
        }


def create_app():
    app = FastAPI()
    app.include_router(workflows.router)
    app.state.orchestrator = DummyOrchestrator()
    return app


def test_email_endpoint_without_attachments():
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/workflows/email",
        data={"subject": "Hi", "recipient": "user@example.com"},
    )
    assert response.status_code == 200
    assert "attachments" not in app.state.orchestrator.last_input
