"""Ensure ``source_type`` payload indexes exist across all Qdrant collections used
for retrieval and backfill missing values where a deterministic default is known.

The migration is idempotent and safe to run multiple times.  It mirrors the
runtime safeguards introduced in :mod:`services.rag_service` so that freshly
provisioned environments (or legacy clusters) align with the expected payload
schema required for policy-aware retrieval.
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient, models

from config.settings import settings


logger = logging.getLogger(__name__)


_COLLECTION_FIELDS = {
    getattr(settings, "qdrant_collection_name", "procwise_document_embeddings"): None,
    getattr(settings, "uploaded_documents_collection_name", "uploaded_documents"): "Upload",
    getattr(settings, "static_policy_collection_name", "static_policy"): "Policy",
    getattr(settings, "learning_collection_name", "learning"): None,
}


def _ensure_index(client: QdrantClient, collection_name: str) -> None:
    try:
        collection = client.get_collection(collection_name=collection_name)
        payload_schema = getattr(collection, "payload_schema", {}) or {}
    except Exception:
        logger.info("Collection '%s' missing; skipping index creation", collection_name)
        return

    if "source_type" in payload_schema:
        return

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        logger.info(
            "Created 'source_type' index on collection '%s'", collection_name
        )
    except Exception:
        logger.exception(
            "Failed to create 'source_type' index on collection '%s'", collection_name
        )


def _backfill_source_type(
    client: QdrantClient,
    collection_name: str,
    expected_value: str,
    *,
    document_type: Optional[str] = None,
    batch_size: int = 256,
) -> None:
    """Populate ``source_type`` for legacy points when a safe default exists."""

    scroll_filter = models.Filter(must=[], must_not=[])
    if document_type:
        scroll_filter.must.append(
            models.FieldCondition(
                key="document_type", match=models.MatchValue(value=document_type.lower())
            )
        )
    if expected_value:
        scroll_filter.must.append(models.IsNullCondition(key="source_type"))

    offset: Optional[list[int]] = None
    total_updated = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            scroll_filter=scroll_filter if (scroll_filter.must or scroll_filter.must_not) else None,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        point_ids = []
        for point in points:
            if point.id is None:
                continue
            payload = getattr(point, "payload", {}) or {}
            if payload.get("source_type"):
                continue
            point_ids.append(point.id)
        if not point_ids:
            continue
        try:
            client.set_payload(
                collection_name=collection_name,
                payload={"source_type": expected_value},
                points=point_ids,
            )
            total_updated += len(point_ids)
        except Exception:
            logger.exception(
                "Failed to backfill 'source_type' for %d points in '%s'",
                len(point_ids),
                collection_name,
            )
            break

    if total_updated:
        logger.info(
            "Backfilled 'source_type' for %d points in '%s'", total_updated, collection_name
        )


def ensure_source_type_indexes(client: QdrantClient) -> None:
    for collection_name, default_value in _COLLECTION_FIELDS.items():
        if not collection_name:
            continue
        _ensure_index(client, collection_name)
        if default_value:
            _backfill_source_type(
                client,
                collection_name,
                expected_value=default_value,
                document_type="policy" if default_value == "Policy" else None,
            )


def run() -> None:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_source_type_indexes(client)


if __name__ == "__main__":
    run()
