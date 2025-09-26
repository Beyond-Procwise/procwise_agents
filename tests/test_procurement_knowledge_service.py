import json
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.procurement_knowledge_service import ProcurementKnowledgeService, ProcurementBrief


def test_load_briefs_filters_invalid_entries(tmp_path):
    payload = [
        {"id": "brief-1", "title": "Valid", "summary": "Insights"},
        {"id": "", "title": "Missing", "summary": ""},
    ]
    knowledge_file = tmp_path / "briefs.json"
    knowledge_file.write_text(json.dumps(payload), encoding="utf-8")

    agent = types.SimpleNamespace(settings=types.SimpleNamespace(procurement_knowledge_path=str(knowledge_file)))
    service = ProcurementKnowledgeService(agent)

    briefs = service.load_briefs()

    assert len(briefs) == 1
    assert isinstance(briefs[0], ProcurementBrief)
    assert briefs[0].identifier == "brief-1"


def test_embed_briefs_strips_customer_fields(monkeypatch):
    captured = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def upsert_texts(self, texts, metadata=None):
            captured["texts"] = texts
            captured["metadata"] = metadata

    dummy_module = types.ModuleType("services.rag_service")
    dummy_module.RAGService = DummyRAG
    monkeypatch.setitem(sys.modules, "services.rag_service", dummy_module)

    brief = types.SimpleNamespace(
        identifier="brief-99",
        title="Market",
        summary="Overview",
        source="UnitTest",
        to_payload=lambda: {
            "record_id": "brief-99",
            "document_type": "external_procurement_brief",
            "customer": "PrivateCo",
        },
    )

    agent = types.SimpleNamespace()
    service = ProcurementKnowledgeService(agent)
    service.embed_briefs([brief])

    assert captured["metadata"]["document_type"] == "external_procurement_brief"
    assert "PrivateCo" not in " ".join(captured["texts"])
