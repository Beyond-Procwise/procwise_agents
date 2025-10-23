from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach request and idempotency identifiers to each request context."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request, call_next):  # type: ignore[override]
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        ikey = request.headers.get("Idempotency-Key")

        request.state.request_id = rid
        request.state.idempotency_key = ikey

        response = await call_next(request)

        response.headers.setdefault("X-Request-ID", rid)
        if ikey:
            response.headers.setdefault("Idempotency-Key", ikey)

        return response
