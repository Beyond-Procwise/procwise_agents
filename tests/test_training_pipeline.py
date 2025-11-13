import json
from pathlib import Path

import pytest

from training.pipeline import (
    ModelfileConfig,
    RAGEvaluationConfig,
    evaluate_rag_model,
    load_evaluation_queries,
    write_modelfile,
)


def test_load_evaluation_queries_handles_jsonl(tmp_path):
    queries_path = tmp_path / "eval.jsonl"
    queries_path.write_text(
        '{"query": "How do I file expenses?", "ensure_min_docs": 4, "topic": "expenses"}\n'
        '"Summarise onboarding rules"\n',
        encoding="utf-8",
    )
    queries = load_evaluation_queries(queries_path)
    assert len(queries) == 2
    assert queries[0].ensure_min_docs == 4
    assert queries[0].metadata["topic"] == "expenses"
    assert queries[1].query == "Summarise onboarding rules"


def test_write_modelfile_renders_template(tmp_path):
    template_path = tmp_path / "template"
    template_path.write_text(
        "FROM {MODEL_PATH}\nPARAMETER temperature {TEMPERATURE}\n{EXTRA_PARAMETERS}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "Modelfile"
    cfg = ModelfileConfig(
        template_path=template_path,
        output_path=output_path,
        model_name="dummy",
        extra_parameters={"stop": "<|im_end|>"},
    )
    weights = tmp_path / "model.gguf"
    weights.write_text("mock", encoding="utf-8")
    result_path = write_modelfile(cfg, weights)
    assert result_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "FROM" in content and "stop" in content
    assert str(weights) in content


def test_evaluate_rag_model_outputs_metrics(tmp_path):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(["Query A", "Query B"]), encoding="utf-8")
    output_path = tmp_path / "report.json"

    class DummyRAG:
        def __init__(self):
            self.calls = 0

        def answer(self, query, ensure_min_docs=3, collections=None):
            self.calls += 1
            return {
                "answer": "All good\n\nSources: doc1, doc2, doc3",
                "sources": ["doc1", "doc2", "doc3"],
                "diagnostics": {
                    "dense": 10,
                    "after_rerank": 8,
                    "after_dedupe": 6,
                    "after_cap": 4,
                    "packed_chars": 500,
                    "elapsed_seconds": 0.3,
                },
            }

    cfg = RAGEvaluationConfig(
        queries_path=queries_path,
        output_path=output_path,
        collections=["primary", "uploaded"],
    )
    rag_instances = []

    def factory():
        rag = DummyRAG()
        rag_instances.append(rag)
        return rag

    result = evaluate_rag_model(factory, cfg)
    assert result.aggregate["queries_evaluated"] == 2
    assert output_path.exists()
    assert rag_instances[0].calls == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "aggregate" in payload and payload["aggregate"]["multi_doc_rate"] == pytest.approx(1.0)
