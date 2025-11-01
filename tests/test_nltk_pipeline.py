import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.nltk_pipeline import NLTKProcessor


@pytest.fixture(scope="module")
def nltk_processor():
    processor = NLTKProcessor()
    if not processor.available:
        pytest.skip("NLTK resources unavailable in test environment")
    return processor


def test_preprocess_extracts_keywords_and_phrases(nltk_processor):
    text = "We need guidance on handling late payment charges for supplier invoices."
    features = nltk_processor.preprocess(text)
    assert features.keywords, "Expected keywords to be extracted"
    assert any("payment" in keyword for keyword in features.keywords)
    assert any("supplier" in phrase.lower() for phrase in features.key_phrases)


def test_postprocess_sentiment_alignment(nltk_processor):
    draft = "Thanks for flagging this. Here's what I can confirm. late fees apply for overdue invoices"
    cleaned = nltk_processor.postprocess(draft, sentiment={"compound": -0.6})
    assert "thanks for flagging" not in cleaned.lower()
    assert cleaned.split(".")[0].lower().startswith("i understand this situation")
