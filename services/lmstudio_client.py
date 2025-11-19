"""Lightweight client for interacting with the local LM Studio server."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Iterable, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class LMStudioClientError(RuntimeError):
    """Raised when the LM Studio server returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LMStudioClient:
    """Simple wrapper around the LM Studio OpenAI-compatible HTTP API."""

    _SAFE_OPTION_KEYS = {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "stop",
    }

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        url = base_url or getattr(settings, "lmstudio_base_url", "http://127.0.0.1:1234")
        if not url.startswith("http"):
            url = f"http://{url}"
        self.base_url = url.rstrip("/")
        self.timeout = timeout or getattr(settings, "lmstudio_timeout", 120)
        self.api_key = api_key or getattr(settings, "lmstudio_api_key", None)
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update(self._headers())
        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            message = exc.response.text if exc.response is not None else str(exc)
            raise LMStudioClientError(message, status_code=status_code) from exc
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise LMStudioClientError(str(exc)) from exc

    @staticmethod
    def _serialise_prompt(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        try:
            return json.dumps(prompt, ensure_ascii=False)
        except Exception:
            return str(prompt)

    def _coerce_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not options:
            return {}
        filtered: Dict[str, Any] = {}
        for key, value in options.items():
            if key in self._SAFE_OPTION_KEYS:
                filtered[key] = value
        return filtered

    @staticmethod
    def _coerce_message_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        for key in ("text", "response", "output", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_models(self) -> List[str]:
        """Return the list of models exposed by LM Studio."""

        response = self._request("GET", "/v1/models")
        payload = response.json() if response.content else {}
        models: List[str] = []
        data = payload.get("data") or payload.get("models") or []
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("id") or entry.get("model") or entry.get("name")
                if name:
                    models.append(name)
            elif isinstance(entry, str):
                models.append(entry)
        return models

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke the chat completions endpoint."""

        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": bool(stream),
        }
        payload.update(self._coerce_options(options))
        if response_format:
            payload["response_format"] = response_format
        response = self._request("POST", "/v1/chat/completions", json=payload, stream=False)
        data = response.json() if response.content else {}
        content = self._coerce_message_content(data)
        return {
            "message": {"role": "assistant", "content": content},
            "response": content,
            "raw": data,
        }

    def generate(
        self,
        *,
        model: str,
        prompt: Any,
        format: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around prompt-only chat completions."""

        response_format = None
        if format == "json":
            response_format = {"type": "json_object"}
        prompt_text = self._serialise_prompt(prompt)
        messages = [{"role": "user", "content": prompt_text}]
        return self.chat(
            model=model,
            messages=messages,
            response_format=response_format,
            stream=stream,
            options=options,
        )

    def embed(self, *, model: str, inputs: Iterable[str]) -> List[List[float]]:
        """Return embeddings for ``inputs``."""

        payload = {"model": model, "input": list(inputs)}
        response = self._request("POST", "/v1/embeddings", json=payload)
        data = response.json() if response.content else {}
        vectors: List[List[float]] = []
        for row in data.get("data", []):
            if not isinstance(row, dict):
                continue
            vector = row.get("embedding")
            if isinstance(vector, list):
                vectors.append(vector)
        return vectors


_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional[LMStudioClient] = None


def get_lmstudio_client() -> LMStudioClient:
    """Return a process-wide LM Studio client."""

    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = LMStudioClient()
    return _CLIENT


__all__ = ["LMStudioClient", "LMStudioClientError", "get_lmstudio_client"]
