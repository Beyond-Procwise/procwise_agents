import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.negotiation_agent import NegotiationAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            qdrant_collection_name="dummy",
            extraction_model="llama3",
            script_user="tester",
            ses_default_sender="noreply@example.com",
        )
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_action=lambda **_: None,
        )
        self.ollama_options = lambda: {}
        self.qdrant_client = SimpleNamespace()
        self.embedding_model = SimpleNamespace(encode=lambda x: [0.0])
        def get_db_connection():
            class DummyConn:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def cursor(self):
                    class DummyCursor:
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                        def execute(self, *args, **kwargs): pass
                        def fetchone(self): return None
                    return DummyCursor()
            return DummyConn()
        self.get_db_connection = get_db_connection



def test_negotiation_agent_handles_missing_fields():
    nick = DummyNick()
    agent = NegotiationAgent(nick)
    context = AgentContext(
        workflow_id="wf1",
        agent_id="negotiation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["counter_proposals"] == []
