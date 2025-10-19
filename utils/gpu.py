"""GPU configuration utilities for the ProcWise agentic framework.

This module centralises GPU-related environment setup so that all agents
and services can rely on a single, consistent configuration.  The
``configure_gpu`` function is idempotent – it will apply settings only
once and return the detected device (``"cuda"`` or ``"cpu"``).

The module also exposes :func:`load_cross_encoder` which initialises
``sentence_transformers`` cross encoders with a graceful fallback for the
``meta`` tensor initialisation error introduced in newer versions of
PyTorch.  When this happens the model is first constructed on CPU and
then moved to the requested GPU, ensuring that GPU acceleration remains
available without crashing the agent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Any

try:  # ``torch`` is optional at import time for some environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

_CONFIGURED: bool = False
_DEVICE: Optional[str] = None
_CROSS_ENCODER_CACHE: dict[tuple[str, str], Any] = {}

logger = logging.getLogger(__name__)


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
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OLLAMA_USE_GPU", "1")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.set_default_device("cuda")
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"

    # Many libraries such as ``sentence_transformers`` respect this variable.
    os.environ.setdefault("SENTENCE_TRANSFORMERS_DEFAULT_DEVICE", _DEVICE)
    os.environ.setdefault("PROCWISE_DEVICE", _DEVICE)

    _CONFIGURED = True
    return _DEVICE


def load_cross_encoder(
    model_name: str,
    cross_encoder_cls: Any,
    device: Any | None,
):
    """Initialise a cross encoder on the desired device with GPU fallback.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.
    cross_encoder_cls:
        The class (typically ``sentence_transformers.CrossEncoder``) used to
        construct the reranker.
    device:
        Preferred device descriptor supplied by the agent.  It can be a
        string (``"cuda"``) or :class:`torch.device` instance.

    Returns
    -------
    Any
        An initialised cross encoder instance.

    Notes
    -----
    Recent PyTorch releases raise ``NotImplementedError`` when moving a
    module containing ``meta`` tensors directly to CUDA.  Some Hugging
    Face models trigger this pathway even though the system has a GPU
    available.  When this happens we retry the initialisation on CPU and
    then move the fully materialised model to the requested device.
    """

    target_device = None if device is None else str(device)
    cache_key = (model_name, (target_device or "cpu"))
    cached = _CROSS_ENCODER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        encoder = cross_encoder_cls(model_name, device=target_device)
        _CROSS_ENCODER_CACHE[cache_key] = encoder
        return encoder
    except NotImplementedError as exc:  # pragma: no cover - hardware dependent
        if "meta tensor" not in str(exc):
            raise
        logger.warning(
            "Cross encoder initialisation failed on device %s due to meta tensor copy; "
            "retrying via CPU fallback.",
            target_device,
        )
        encoder = cross_encoder_cls(model_name, device="cpu")
        if target_device and target_device != "cpu":
            try:
                encoder.to(target_device)
            except NotImplementedError:
                logger.exception(
                    "Cross encoder could not be moved to device %s after CPU initialisation; "
                    "continuing on CPU.",
                    target_device,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Unexpected error moving cross encoder to device %s; continuing on CPU.",
                    target_device,
                )
        _CROSS_ENCODER_CACHE[cache_key] = encoder
        return encoder
