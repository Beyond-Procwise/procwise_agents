"""Utilities for loading shared reference datasets.

The procurement agents rely on structured reference data – such as
scoring rules and model hyper-parameters – that should not be hard-coded
inside the agent implementations.  This module centralises loading and
lightweight caching of those datasets which are stored under
``resources/reference_data`` as JSON documents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REFERENCE_CACHE: Dict[str, Dict[str, Any]] = {}


def _reference_base_path() -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / "reference_data"


def load_reference_dataset(name: str) -> Dict[str, Any]:
    """Return the JSON payload for ``name`` from the reference store.

    Parameters
    ----------
    name:
        The logical dataset name. The loader will look for
        ``resources/reference_data/{name}.json``.
    """

    key = str(name).strip()
    if not key:
        raise ValueError("reference dataset name must be a non-empty string")

    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    base = _reference_base_path()
    path = base / f"{key}.json"

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        logger.warning("Reference dataset '%s' not found at %s", key, path)
        payload = {}
    except json.JSONDecodeError:
        logger.exception("Reference dataset '%s' could not be decoded", key)
        payload = {}

    if not isinstance(payload, dict):
        logger.warning(
            "Reference dataset '%s' is not an object – defaulting to empty dict", key
        )
        payload = {}

    _REFERENCE_CACHE[key] = payload
    return payload

