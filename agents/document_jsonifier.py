from __future__ import annotations

import re
from typing import Any, Dict, List


def convert_document_to_json(text: str, doc_type: str | None = None) -> Dict[str, Any]:
    """Convert raw document text into a JSON-like dictionary.

    The implementation is intentionally lightweight. It scans the text for
    ``key: value`` patterns and returns them under ``header_data``. More
    sophisticated extraction (e.g. using an LLM) can be plugged in later.

    Parameters
    ----------
    text:
        Raw text extracted from the document.
    doc_type:
        Optional hint of document type such as ``"Invoice"`` or
        ``"Purchase_Order"``. The current implementation does not make use of
        this but it is provided for future extension.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing ``header_data`` and ``line_items`` keys.
    """
    header: Dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = re.sub(r"\s+", "_", key.strip().lower())
            header[key] = value.strip()
    # Line item extraction can be plugged in here if needed
    return {"header_data": header, "line_items": []}
