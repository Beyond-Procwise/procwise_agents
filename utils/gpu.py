"""GPU configuration utilities for the ProcWise agentic framework.

This module centralises GPU-related environment setup so that all agents
and services can rely on a single, consistent configuration.  The
``configure_gpu`` function is idempotent – it will apply settings only
once and return the detected device (``"cuda"`` or ``"cpu"``).
"""

from __future__ import annotations

import os
from typing import Optional

try:  # ``torch`` is optional at import time for some environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

_CONFIGURED: bool = False
_DEVICE: Optional[str] = None


def configure_gpu() -> str:
    """Configure GPU environment variables and default device.

    Returns
    -------
    str
        The device string (``"cuda"`` or ``"cpu"``) that should be used by
        downstream libraries.
    """
    global _CONFIGURED, _DEVICE
    if _CONFIGURED:
        return _DEVICE or "cpu"

    # Ensure GPU visibility and enablement for libraries that honour these
    # environment variables. Defaults are chosen to utilise the first GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OLLAMA_USE_GPU", "1")

    if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.set_default_device("cuda")
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"

    # Many libraries such as ``sentence_transformers`` respect this variable.
    os.environ.setdefault("SENTENCE_TRANSFORMERS_DEFAULT_DEVICE", _DEVICE)

    _CONFIGURED = True
    return _DEVICE
