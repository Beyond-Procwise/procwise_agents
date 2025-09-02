from __future__ import annotations

"""Lightweight NLP utilities used across the ProcWise agentic framework.

This module exposes helper functions for performing named-entity
recognition (NER) using Hugging Face ``transformers``.  The pipeline is
lazily initialised and configured to utilise the GPU when available via
:func:`utils.gpu.configure_gpu`.

If the ``transformers`` library or required model is unavailable, the
functions in this module gracefully fall back to returning empty
results.  This ensures that unit tests and environments without the
optional dependency continue to operate without failure while still
providing enhanced accuracy when the dependency is present.
"""

from typing import Any, Dict, List, Optional

from utils.gpu import configure_gpu

try:  # ``transformers`` is optional at runtime
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

_NER_PIPELINE: Optional[Any] = None


def get_ner_pipeline() -> Optional[Any]:
    """Return a cached NER pipeline configured for GPU acceleration.

    The model ``dslim/bert-base-NER`` is used for entity extraction.  GPU
    usage is enabled when available by passing the appropriate ``device``
    index to the Hugging Face pipeline.  If the ``transformers`` library is
    not installed the function returns ``None``.
    """

    global _NER_PIPELINE
    if _NER_PIPELINE is not None:
        return _NER_PIPELINE

    if pipeline is None:  # pragma: no cover - optional dependency
        return None

    device = 0 if configure_gpu() == "cuda" else -1
    try:  # pragma: no cover - defensive
        _NER_PIPELINE = pipeline(
            "ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device
        )
    except Exception:
        _NER_PIPELINE = None
    return _NER_PIPELINE


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from ``text`` using the NER pipeline.

    Parameters
    ----------
    text:
        The unstructured text from which to extract entities.

    Returns
    -------
    List[Dict[str, Any]]
        A list of entities as returned by the Hugging Face pipeline.  If the
        pipeline is unavailable an empty list is returned.
    """

    ner = get_ner_pipeline()
    if ner is None:
        return []
    try:  # pragma: no cover - defensive
        return ner(text)  # type: ignore[no-any-return]
    except Exception:
        return []
