"""Ingest and maintain the static procurement policy corpus in Qdrant."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from qdrant_client import models

from services.document_embedding_service import (
    DocumentEmbeddingService,
    EmbeddedDocument,
)

logger = logging.getLogger(__name__)


class StaticPolicyLoader(DocumentEmbeddingService):
    """Synchronise S3-hosted policy documents into a dedicated Qdrant collection."""

    _DEFAULT_PREFIX = "Static Policy/"

    def __init__(
        self,
        agent_nick: Any,
        *,
        collection_name: Optional[str] = None,
    ) -> None:
        target_collection = collection_name or getattr(
            getattr(agent_nick, "settings", None),
            "static_policy_collection_name",
            "static_policy",
        )
        super().__init__(
            agent_nick,
            collection_name=target_collection,
            rag_service_factory=lambda *_: None,
        )
        # Disable propagation into the general RAG store – the policy corpus is
        # retrieved directly as a dedicated collection.
        self._rag_service_failed = True

        self._policy_indexes_ready = False
        self._s3_client = getattr(agent_nick, "s3_client", None)
        if self._s3_client is None:
            raise ValueError("AgentNick must expose an S3 client for policy ingestion")

        settings = getattr(agent_nick, "settings", None)
        bucket_override = getattr(settings, "static_policy_s3_bucket", None)
        default_bucket = getattr(settings, "s3_bucket_name", None)
        self._bucket = (bucket_override or default_bucket or "").strip()
        prefix = getattr(settings, "static_policy_s3_prefix", self._DEFAULT_PREFIX)
        self._prefix = prefix.lstrip("/") if prefix else self._DEFAULT_PREFIX
        self._auto_ingest = bool(getattr(settings, "static_policy_auto_ingest", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sync_static_policy(
        self,
        *,
        s3_uri: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Load policy documents from S3 and persist them into Qdrant.

        Parameters
        ----------
        s3_uri:
            Optional full S3 URI (``s3://bucket/prefix``) overriding the
            configured bucket and prefix for a single sync.
        force:
            When ``True`` every discovered object is re-ingested even if the
            stored ETag matches the current object metadata.
        """

        summary = {"ingested": 0, "skipped": 0, "errors": []}
        if not self._auto_ingest:
            logger.info("Static policy auto-ingest disabled; skipping sync")
            return summary

        bucket, prefix = self._resolve_target(s3_uri)
        if not bucket:
            logger.warning("Static policy bucket is not configured; skipping ingestion")
            return summary

        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        except (BotoCoreError, ClientError, NoCredentialsError) as exc:
            logger.warning(
                "Unable to list static policy objects from s3://%s/%s: %s",
                bucket,
                prefix,
                exc,
            )
            summary["errors"].append(
                {"bucket": bucket, "prefix": prefix, "error": str(exc)}
            )
            return summary

        for page in page_iterator:
            for entry in page.get("Contents", []):
                key = entry.get("Key")
                size = entry.get("Size")
                if not key or key.endswith("/") or not size:
                    continue

                etag = self._clean_etag(entry.get("ETag"))
                last_modified = entry.get("LastModified")
                source_path = f"s3://{bucket}/{key}"

                existing = self._lookup_existing(source_path)
                if not force and existing and existing.get("s3_etag") == etag:
                    summary["skipped"] += 1
                    continue

                if existing:
                    self._purge_existing(source_path)

                try:
                    file_bytes = self._fetch_object(bucket, key)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to download static policy object %s/%s", bucket, key
                    )
                    summary["errors"].append(
                        {
                            "bucket": bucket,
                            "key": key,
                            "error": str(exc),
                        }
                    )
                    continue

                try:
                    self._ingest_object(
                        bucket=bucket,
                        key=key,
                        body=file_bytes,
                        etag=etag,
                        last_modified=last_modified,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Static policy ingestion failed for s3://%s/%s", bucket, key
                    )
                    summary["errors"].append(
                        {
                            "bucket": bucket,
                            "key": key,
                            "error": str(exc),
                        }
                    )
                    continue

                summary["ingested"] += 1

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_target(self, s3_uri: Optional[str]) -> Tuple[str, str]:
        if s3_uri and s3_uri.startswith("s3://"):
            without_scheme = s3_uri[5:]
            parts = without_scheme.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            return bucket, prefix.lstrip("/")
        return self._bucket, self._prefix

    def _clean_etag(self, etag: Optional[str]) -> Optional[str]:
        if not isinstance(etag, str):
            return None
        cleaned = etag.strip('"')
        return cleaned or None

    def _lookup_existing(self, source_path: str) -> Optional[Dict[str, Any]]:
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_path",
                            match=models.MatchValue(value=source_path),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            logger.debug("Static policy lookup failed for %s", source_path, exc_info=True)
            return None

        for point in points or []:
            payload = getattr(point, "payload", None)
            if isinstance(payload, dict):
                return dict(payload)
        return None

    def _purge_existing(self, source_path: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_path",
                                match=models.MatchValue(value=source_path),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception:
            logger.warning(
                "Failed to purge stale policy payloads for %s", source_path, exc_info=True
            )

    def _fetch_object(self, bucket: str, key: str) -> bytes:
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
        except (BotoCoreError, ClientError, NoCredentialsError) as exc:
            raise RuntimeError(f"Unable to fetch s3://{bucket}/{key}: {exc}") from exc

        body = response.get("Body")
        if hasattr(body, "read"):
            data = body.read()
        else:
            data = body if isinstance(body, (bytes, bytearray)) else b""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"S3 object {bucket}/{key} returned non-bytes body")
        return bytes(data)

    def _ingest_object(
        self,
        *,
        bucket: str,
        key: str,
        body: bytes,
        etag: Optional[str],
        last_modified: Optional[Any],
    ) -> EmbeddedDocument:
        filename = Path(key).name
        extracted = self._extract_text(file_bytes=body, filename=filename)
        text = extracted.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Document {filename} produced no textual content")

        document_id = self._derive_document_id(bucket, key, etag)
        chunk_seed: Dict[str, Any] = {
            "document_id": document_id,
            "doc_name": Path(filename).stem,
            "title": extracted.metadata.get("title") or Path(filename).stem,
            "document_type": "Policy",
            "source_type": "Policy",
            "effective_date": extracted.metadata.get("effective_date"),
            "supplier": extracted.metadata.get("supplier"),
            "doc_version": extracted.metadata.get("doc_version"),
            "round_id": extracted.metadata.get("round_id"),
        }

        chunks = self._build_document_chunks(
            text,
            filename=filename,
            document_type_hint="Policy",
            extraction_metadata=extracted.metadata,
            base_metadata=chunk_seed,
        )
        if not chunks:
            raise ValueError(f"No embeddable chunks produced for {filename}")

        vectors = self._encode_chunks(chunks)
        if not vectors:
            raise ValueError(f"Embedding model returned no vectors for {filename}")

        vector_size = len(vectors[0])
        self._ensure_policy_collection(vector_size)

        namespace_uuid = uuid.UUID(document_id)
        ingested_at = datetime.utcnow().isoformat(timespec="seconds")
        last_modified_iso = (
            last_modified.isoformat() if hasattr(last_modified, "isoformat") else None
        )

        base_metadata: Dict[str, Any] = {
            "document_id": document_id,
            "policy_document_id": document_id,
            "record_id": document_id,
            "collection_name": self.collection_name,
            "policy_source": "static_policy",
            "official_policy": True,
            "document_type": "Policy",
            "doc_name": Path(filename).stem,
            "filename": filename,
            "file_extension": Path(filename).suffix.lower(),
            "uploaded_at": ingested_at,
            "ingested_at": ingested_at,
            "source_path": f"s3://{bucket}/{key}",
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_etag": etag,
            "s3_last_modified": last_modified_iso,
            "extraction_method": extracted.method,
        }
        base_metadata.update(chunk_seed)
        base_metadata.update(extracted.metadata or {})
        base_metadata.setdefault("source_type", "Policy")
        base_metadata.setdefault("title", chunk_seed.get("title"))
        base_metadata.setdefault("effective_date", chunk_seed.get("effective_date"))
        base_metadata.setdefault("supplier", chunk_seed.get("supplier"))
        base_metadata.setdefault("doc_version", chunk_seed.get("doc_version"))
        base_metadata.setdefault("round_id", chunk_seed.get("round_id"))

        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = dict(base_metadata)
            payload.update(chunk.metadata)
            payload["document_type"] = "policy"
            payload["chunk_id"] = idx
            payload["chunk_index"] = idx
            payload["content"] = chunk.content
            payload["text_summary"] = chunk.content
            payload["embedding_model"] = self._embedding_model_name

            point_id = uuid.uuid5(namespace_uuid, str(idx))
            points.append(
                models.PointStruct(id=str(point_id), vector=vector, payload=payload)
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return EmbeddedDocument(
            document_id=document_id,
            collection=self.collection_name,
            chunk_count=len(points),
            metadata=base_metadata,
        )

    def _ensure_policy_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("Vector size must be positive for policy ingestion")
        self._ensure_collection(vector_size)
        if self._policy_indexes_ready:
            return

        indexed_fields = (
            "document_type",
            "policy_source",
            "policy_document_id",
            "section",
            "source_path",
            "record_id",
        )
        for field in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception:
                logger.debug(
                    "Failed to create payload index for static policy field %s",
                    field,
                    exc_info=True,
                )

        self._policy_indexes_ready = True

    def _derive_document_id(self, bucket: str, key: str, etag: Optional[str]) -> str:
        base = f"s3://{bucket}/{key}"
        if etag:
            base = f"{base}?etag={etag}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

