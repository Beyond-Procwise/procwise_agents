"""Create the ``learning`` Qdrant collection used for workflow learnings.

The migration is idempotent and can be re-run safely.  It mirrors the runtime
initialisation performed by :class:`services.learning_repository.LearningRepository`
so that environments without the full application stack can provision the
collection independently.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models

from config.settings import settings


logger = logging.getLogger(__name__)


COLLECTION_NAME = getattr(settings, "learning_collection_name", "learning")
VECTOR_SIZE = int(getattr(settings, "vector_size", 768))

REQUIRED_INDEXES = {
    "document_type": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "event_type": models.PayloadSchemaType.KEYWORD,
    "workflow_id": models.PayloadSchemaType.KEYWORD,
    "rfq_id": models.PayloadSchemaType.KEYWORD,
    "supplier_id": models.PayloadSchemaType.KEYWORD,
}


def ensure_learning_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        logger.info("Learning collection '%s' already present", COLLECTION_NAME)
    except Exception:
        logger.info("Creating learning collection '%s'", COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    try:
        collection = client.get_collection(collection_name=COLLECTION_NAME)
        payload_schema: dict[str, Any] = getattr(collection, "payload_schema", {}) or {}
    except Exception:
        payload_schema = {}

    existing = set(payload_schema.keys())
    for field_name, schema in REQUIRED_INDEXES.items():
        if field_name in existing:
            continue
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            logger.info("Created payload index '%s' on '%s'", field_name, COLLECTION_NAME)
        except Exception:
            logger.exception(
                "Failed to create payload index '%s' on '%s'", field_name, COLLECTION_NAME
            )


def run() -> None:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_learning_collection(client)


if __name__ == "__main__":
    run()
