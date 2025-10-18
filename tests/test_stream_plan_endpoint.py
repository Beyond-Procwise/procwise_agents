import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers.stream import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.orchestrator = None
    return app


def _collect_events(response):
    events = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        events.append(line)
    return events


def test_stream_plan_emits_expected_events():
    app = create_app()
    client = TestClient(app)

    with client.stream("POST", "/stream/plan", json={"task": "Evaluate supplier quotes"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = _collect_events(response)

    event_names = [line.split(": ")[1] for line in lines if line.startswith("event:")]
    assert "connected" in event_names
    assert "planning" in event_names
    assert "plan_created" in event_names
    assert "execution_complete" == event_names[-1]

    data_lines = [line for line in lines if line.startswith("data:")]
    assert any("\"plan\"" in line for line in data_lines)
    assert any("\"status\": \"success\"" in line for line in data_lines)
