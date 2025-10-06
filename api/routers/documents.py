"""Document-centric API endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from services.document_extractor import DocumentExtractor

# Ensure GPU-related environment variables align with the rest of the API.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_agent_nick(request: Request):
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if not agent_nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return agent_nick


class S3DocumentExtractionRequest(BaseModel):
    """Input payload for S3-backed document extraction."""

    object_key: str = Field(..., description="S3 object key for the procurement document")
    document_type: Optional[str] = Field(
        default=None,
        description="Optional explicit document type override",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("object_key", mode="before")
    @classmethod
    def _strip_object_key(cls, value: Any) -> str:
        if value is None:
            raise ValueError("object_key must be provided")
        if not isinstance(value, str):
            value = str(value)
        key = value.strip()
        if not key:
            raise ValueError("object_key must not be empty")
        return key


@router.post("/extract-from-s3")
def extract_document_from_s3(
    req: S3DocumentExtractionRequest,
    agent_nick=Depends(get_agent_nick),
):
    """Download a document from S3, extract it, and persist the structured payload."""

    bucket = getattr(agent_nick.settings, "s3_bucket_name", None)
    if not bucket:
        raise HTTPException(status_code=503, detail="S3 bucket is not configured")

    object_key = req.object_key
    suffix = os.path.splitext(object_key)[1] or ".bin"

    try:
        with agent_nick.reserve_s3_connection() as s3_client:
            response = s3_client.get_object(Bucket=bucket, Key=object_key)
    except ClientError as exc:  # pragma: no cover - network failures in prod
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = (error or {}).get("Code")
        status = 404 if code in {"NoSuchKey", "404"} else 500
        detail = error.get("Message") or f"Unable to download {object_key}"
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to download %s from S3", object_key)
        raise HTTPException(status_code=500, detail="Failed to download document from S3") from exc

    body = response.get("Body") if isinstance(response, dict) else None
    if body is None:
        raise HTTPException(status_code=500, detail="S3 object body was empty")

    try:
        file_bytes = body.read()
    finally:
        try:
            body.close()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to close S3 streaming body for %s", object_key, exc_info=True)

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    connection_factory = lambda: agent_nick.get_db_connection()
    chat_options = {}
    if hasattr(agent_nick, "ollama_options") and callable(agent_nick.ollama_options):
        try:
            chat_options = agent_nick.ollama_options()
        except Exception:  # pragma: no cover - fall back to defaults
            logger.debug("Failed to obtain Ollama chat options", exc_info=True)
            chat_options = {}

    extractor = DocumentExtractor(
        connection_factory,
        preferred_models=("qwen3", "phi4"),
        chat_options=chat_options,
    )

    metadata = dict(req.metadata or {})
    metadata.setdefault("ingestion_mode", "s3")
    metadata.setdefault("s3_bucket", bucket)
    metadata.setdefault("s3_object_key", object_key)

    try:
        result = extractor.extract(
            tmp_path,
            document_type=req.document_type,
            metadata=metadata,
            source_label=object_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Document extraction failed for %s", object_key)
        raise HTTPException(status_code=500, detail="Document extraction failed") from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to remove temporary file %s", tmp_path, exc_info=True)

    return {"status": "completed", "document": result.to_json()}


__all__ = ["router"]

