"""Document-centric API endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
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

    object_key: Optional[str] = Field(
        default=None,
        description="Optional S3 object key for extracting a single document",
    )
    s3_path: Optional[str] = Field(
        default=None,
        description="S3 path/prefix pointing at a folder of procurement documents",
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Optional explicit document type override",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("object_key", mode="before")
    @classmethod
    def _strip_object_key(cls, value: Any) -> str:
        if value is None:
            return value  # handled by overall validation
        if not isinstance(value, str):
            value = str(value)
        key = value.strip()
        return key or None

    @field_validator("s3_path", mode="before")
    @classmethod
    def _strip_s3_path(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        path = value.strip()
        return path or None

    @model_validator(mode="after")
    def _ensure_location(cls, values: "S3DocumentExtractionRequest") -> "S3DocumentExtractionRequest":
        if not values.object_key and not values.s3_path:
            raise ValueError("Either object_key or s3_path must be provided")
        if values.object_key and values.s3_path:
            raise ValueError("Provide either object_key or s3_path, not both")
        return values


def _resolve_s3_path(s3_path: str, default_bucket: str) -> Tuple[str, str]:
    """Return bucket and prefix for the provided path."""

    value = s3_path.strip()
    if not value:
        raise ValueError("s3_path must not be empty")

    if value.startswith("s3://"):
        stripped = value[5:]
        bucket_part, _, key_part = stripped.partition("/")
        bucket_name = bucket_part or default_bucket
        prefix = key_part
    else:
        bucket_name = default_bucket
        prefix = value

    prefix = prefix.lstrip("/")
    if not prefix:
        raise ValueError("s3_path must include an object prefix")
    return bucket_name, prefix


def _list_s3_keys(client: Any, bucket: str, prefix: str) -> List[str]:
    """List all object keys for the provided prefix."""

    keys: List[str] = []
    continuation: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            params["ContinuationToken"] = continuation
        response = client.list_objects_v2(**params)
        contents = (response or {}).get("Contents", [])
        for entry in contents:
            key = entry.get("Key") if isinstance(entry, dict) else None
            if key and not key.endswith("/"):
                keys.append(key)

        if not response or not response.get("IsTruncated"):
            break
        continuation = response.get("NextContinuationToken")
        if not continuation:
            break

    return keys


def _download_s3_object(client: Any, bucket: str, key: str) -> bytes:
    """Download an S3 object and return its bytes."""

    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body") if isinstance(response, dict) else None
    if body is None:
        raise HTTPException(status_code=500, detail=f"S3 object body was empty for {key}")

    try:
        return body.read()
    finally:
        try:
            body.close()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to close S3 streaming body for %s", key, exc_info=True)


@router.post("/extract-from-s3")
def extract_document_from_s3(
    req: S3DocumentExtractionRequest,
    agent_nick=Depends(get_agent_nick),
):
    """Download a document from S3, extract it, and persist the structured payload."""

    bucket = getattr(agent_nick.settings, "s3_bucket_name", None)
    if not bucket:
        raise HTTPException(status_code=503, detail="S3 bucket is not configured")

    effective_bucket = bucket
    prefix: Optional[str] = None

    if req.s3_path:
        try:
            effective_bucket, prefix = _resolve_s3_path(req.s3_path, bucket)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    base_metadata = dict(req.metadata or {})

    documents: List[Dict[str, Any]] = []

    try:
        with agent_nick.reserve_s3_connection() as s3_client:
            object_keys: List[str]
            if prefix is not None:
                try:
                    object_keys = _list_s3_keys(s3_client, effective_bucket, prefix)
                except ClientError as exc:  # pragma: no cover - network failures in prod
                    error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
                    detail = error.get("Message") or f"Unable to list objects for {prefix}"
                    raise HTTPException(status_code=500, detail=detail) from exc
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Failed to list S3 objects for %s", prefix)
                    raise HTTPException(status_code=500, detail="Failed to list S3 objects") from exc

                if not object_keys:
                    raise HTTPException(status_code=404, detail="No documents found for provided s3_path")
            else:
                assert req.object_key is not None  # validated during model creation
                object_keys = [req.object_key]

            for object_key in object_keys:
                if not object_key:
                    continue
                suffix = os.path.splitext(object_key)[1] or ".bin"
                try:
                    file_bytes = _download_s3_object(s3_client, effective_bucket, object_key)
                except ClientError as exc:  # pragma: no cover - network failures in prod
                    error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
                    code = (error or {}).get("Code")
                    status = 404 if code in {"NoSuchKey", "404"} else 500
                    detail = error.get("Message") or f"Unable to download {object_key}"
                    raise HTTPException(status_code=status, detail=detail) from exc
                except HTTPException:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Failed to download %s from S3", object_key)
                    raise HTTPException(status_code=500, detail="Failed to download document from S3") from exc

                tmp_path = None
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = Path(tmp.name)

                    metadata = dict(base_metadata)
                    metadata.setdefault("ingestion_mode", "s3")
                    metadata.setdefault("s3_bucket", effective_bucket)
                    metadata.setdefault("s3_object_key", object_key)

                    result = extractor.extract(
                        tmp_path,
                        document_type=req.document_type,
                        metadata=metadata,
                        source_label=object_key,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except HTTPException:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Document extraction failed for %s", object_key)
                    raise HTTPException(status_code=500, detail="Document extraction failed") from exc
                finally:
                    if tmp_path:
                        try:
                            tmp_path.unlink()
                        except FileNotFoundError:
                            pass
                        except Exception:  # pragma: no cover - defensive cleanup
                            logger.debug("Failed to remove temporary file %s", tmp_path, exc_info=True)

                documents.append(result.to_json())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive cleanup
        logger.exception("Unexpected failure during S3 extraction")
        raise HTTPException(status_code=500, detail="Document extraction failed") from exc

    response_payload: Dict[str, Any] = {"status": "completed", "documents": documents}
    if documents:
        response_payload["document"] = documents[0]
    return response_payload


__all__ = ["router"]

