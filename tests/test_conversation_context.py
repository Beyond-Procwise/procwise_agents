import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from models.context_trainer import ConversationDatasetWriter
from services.conversation_memory import ConversationMemoryService


class DummyRAGService:
    def __init__(self):
        self.upserts = []

    def upsert_texts(self, texts, metadata=None):
        self.upserts.append({"texts": list(texts), "metadata": dict(metadata or {})})

    def search(self, query, top_k=5, filters=None):
        class Hit:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        return [
            Hit({"content": "Previous scanned concession", "document_origin": "scanned"}, 0.92),
            Hit({"content": "Digital quote revision", "document_origin": "digital"}, 0.81),
        ]


def _conversation_samples():
    return [
        {
            "message_id": "m-1",
            "from_address": "buyer@acme.com",
            "message_body": "Need a better offer on the widgets.",
            "document_origin": "digital",
            "supplier_id": "SUP-1",
        },
        {
            "message_id": "m-2",
            "from_address": "supplier@widgets.com",
            "message_body": "Revising price to 9.95.",
            "document_origin": "digital",
            "supplier_id": "SUP-1",
        },
        {
            "message_id": "m-3",
            "from_address": "buyer@acme.com",
            "message_body": "Scanned counter signed and uploaded.",
            "document_origin": "scanned",
            "supplier_id": "SUP-1",
        },
        {
            "message_id": "m-4",
            "from_address": "supplier@widgets.com",
            "message_body": "Scanned acknowledgement received.",
            "document_origin": "scanned",
            "supplier_id": "SUP-1",
        },
    ]


def test_conversation_memory_ingests_scanned_and_digital():
    rag = DummyRAGService()
    service = ConversationMemoryService(SimpleNamespace(), rag_service=rag)

    stored = service.ingest("wf-123", _conversation_samples())

    assert len(stored) == 4
    origins = {meta.get("document_origin") for meta in stored}
    assert origins == {"digital", "scanned"}

    hits = service.retrieve("wf-123", "price revision", limit=2)
    assert len(hits) == 2
    assert {hit.metadata.get("document_origin") for hit in hits} == {"digital", "scanned"}


class RecordingProcessRoutingService:
    def log_process(self, **kwargs):
        return 1

    def log_run_detail(self, **kwargs):
        return "run-1"

    def log_action(self, **kwargs):
        return "action-1"

    def validate_workflow_id(self, *_args, **_kwargs):
        return True


class ContextAwareAgent(base_agent.BaseAgent):
    def __init__(self, nick, memory, writer):
        self._test_memory = memory
        super().__init__(nick)
        self._context_dataset_writer = writer

    def _conversation_memory_service(self):
        return self._test_memory

    def run(self, context):
        return base_agent.AgentOutput(
            status=base_agent.AgentStatus.SUCCESS,
            data={"message": "Here is the consolidated negotiation summary."},
        )


class DummyConversationMemory:
    def __init__(self):
        self.ingested = []

    def ingest(self, workflow_id, entries):
        self.ingested.append((workflow_id, list(entries)))
        return []

    def retrieve(self, workflow_id, query, limit=5):
        return [
            SimpleNamespace(
                content="Supplier offered 9.95 GBP on last round",
                score=0.88,
                metadata={"document_origin": "digital"},
            )
        ]


def test_base_agent_writes_context_dataset(tmp_path):
    dataset_dir = tmp_path / "context"
    writer = ConversationDatasetWriter(str(dataset_dir))
    memory = DummyConversationMemory()

    settings = SimpleNamespace(
        script_user="tester",
        extraction_model="gpt-oss",
        context_training_data_dir=str(dataset_dir),
        context_model_output_dir=str(tmp_path / "model"),
    )

    nick = SimpleNamespace(
        settings=settings,
        prompt_engine=SimpleNamespace(),
        process_routing_service=RecordingProcessRoutingService(),
        learning_repository=None,
        lmstudio_options=lambda: {},
    )

    agent = ContextAwareAgent(nick, memory, writer)

    context = base_agent.AgentContext(
        workflow_id="wf-ctx",
        agent_id="agent-ctx",
        user_id="user-1",
        input_data={
            "conversation_history": _conversation_samples(),
            "prompt": "Summarise the negotiation so far",
        },
    )

    result = agent.execute(context)

    assert result.context_snapshot
    assert len(result.context_snapshot["conversation_history"]) == 4
    assert memory.ingested and len(memory.ingested[0][1]) == 4

    saved_files = list(dataset_dir.glob("*.json"))
    assert len(saved_files) == 1
    payload = json.loads(saved_files[0].read_text())
    assert "context" in payload and "response" in payload
    assert "digital" in payload["context"]
    assert "scanned" in payload["context"]
