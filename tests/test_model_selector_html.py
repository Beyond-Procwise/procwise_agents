import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_selector import RAGPipeline


def _make_pipeline() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


def test_normalise_answer_html_preserves_angle_brackets():
    pipeline = _make_pipeline()

    result = pipeline._normalise_answer_html("Cost A < Cost B")

    assert result == "<section><h2>Response</h2><p>Cost A &lt; Cost B</p></section>"


def test_normalise_answer_html_strips_basic_tags():
    pipeline = _make_pipeline()

    result = pipeline._normalise_answer_html("<strong>Winner:</strong> Supplier < 2")

    assert result == "<section><h2>Response</h2><p>Winner: Supplier &lt; 2</p></section>"
    assert not re.search(r"<strong>.*</strong>", result)
