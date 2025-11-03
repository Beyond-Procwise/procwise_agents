"""Prompt utilities backed by the ``proc.prompt`` database table.

The historical implementation of :class:`PromptEngine` focused solely on
LLM-based query decomposition and expected callers to manually feed prompt
libraries loaded from JSON fixtures.  Runtime environments now require that
prompt configuration is sourced directly from PostgreSQL so that updates made
through the governance UI immediately influence agent behaviour.  This module
therefore combines both responsibilities – sourcing prompts from the database
and exposing them to downstream consumers while retaining the query
decomposition helper.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import ollama
import torch

from config.settings import settings

logger = logging.getLogger(__name__)


class PromptEngine:
    """Load prompt metadata from ``proc.prompt`` and expose helper utilities."""

    _DEFAULT_COLUMNS: Sequence[str] = (
        "prompt_id",
        "prompt_name",
        "prompt_type",
        "prompt_linked_agents",
        "prompts_desc",
        "prompts_status",
        "created_date",
        "created_by",
        "last_modified_date",
        "last_modified_by",
        "version",
    )

    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        prompt_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        fallback_model = getattr(settings, "extraction_model", "gpt-oss")
        resolved_model: Optional[str] = None
        if agent_nick is not None:
            resolver = getattr(agent_nick, "get_agent_model", None)
            if callable(resolver):
                try:
                    candidate = resolver("prompt_engine", fallback=None)
                except Exception:  # pragma: no cover - defensive logging only
                    logger.debug(
                        "PromptEngine model resolution failed; using fallback",
                        exc_info=True,
                    )
                else:
                    if isinstance(candidate, str) and candidate.strip():
                        resolved_model = candidate.strip()

        self.llm_model = resolved_model or fallback_model

        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None

        self.agent_nick = agent_nick

        self._prompts: List[Dict[str, Any]] = []
        self._prompt_index: Dict[int, Dict[str, Any]] = {}
        self._agent_index: Dict[str, List[Dict[str, Any]]] = {}

        self._build_cache(prompt_rows)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self):
        factory = self._connection_factory
        if factory is None:
            yield None
            return

        resource = factory() if callable(factory) else factory
        if resource is None:
            yield None
            return

        if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
            with resource as conn:
                yield conn
            return

        try:
            yield resource
        finally:  # pragma: no cover - defensive cleanup
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close prompt connection")

    def _fetch_prompt_rows(self) -> List[Dict[str, Any]]:
        columns: Sequence[str] = self._DEFAULT_COLUMNS
        with self._connect() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT prompt_id, prompt_name, prompt_type, prompt_linked_agents,
                               prompts_desc, prompts_status, created_date, created_by,
                               last_modified_date, last_modified_by, version
                        FROM proc.prompt
                        WHERE COALESCE(prompts_status, 1) = 1
                        """
                    )
                    rows = cursor.fetchall()
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
            except Exception:
                logger.exception("Failed to load prompts from database")
                return []

        return [dict(zip(columns, row)) for row in rows] if rows else []

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = re.sub(r"(?<!^)(?=[A-Z0-9])", "_", text)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text).lower().strip("_")
        return slug

    @classmethod
    def _coerce_row(cls, row: Any) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        if isinstance(row, dict):
            return dict(row)
        if isinstance(row, (list, tuple)):
            length = len(row)
            if length == len(cls._DEFAULT_COLUMNS):
                return dict(zip(cls._DEFAULT_COLUMNS, row))
            if length == 5:
                columns = (
                    "prompt_id",
                    "prompt_name",
                    "prompt_type",
                    "prompt_linked_agents",
                    "prompts_desc",
                )
                return dict(zip(columns, row))
            if length == 3:
                columns = ("prompt_id", "prompts_desc", "prompt_linked_agents")
                return dict(zip(columns, row))
            result: Dict[str, Any] = {}
            if length:
                result["prompt_id"] = row[0]
            if length > 1:
                result["prompts_desc"] = row[1]
            if length > 2:
                result["prompt_linked_agents"] = row[2]
            return result
        return None

    @classmethod
    def _coerce_linked_agents(cls, payload: Any) -> List[str]:
        tokens: List[str] = []
        if payload is None:
            return tokens
        if isinstance(payload, str):
            payload = re.findall(r"[A-Za-z0-9_]+", payload)
        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                slug = cls._slugify(item)
                if slug:
                    tokens.append(slug)
        else:
            slug = cls._slugify(payload)
            if slug:
                tokens.append(slug)
        return tokens

    @staticmethod
    def _parse_prompt_payload(payload: Any) -> Dict[str, Any]:
        raw_text: str = ""
        parsed: Any = None
        template: Optional[str] = None
        metadata: Any = None
        prompt_config: Any = None
        instructions: Any = None

        original = payload
        if isinstance(original, (bytes, bytearray)):
            original = original.decode(errors="ignore")

        if isinstance(original, str):
            raw_text = original.strip()
            if raw_text:
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    parsed = None
        elif isinstance(original, dict):
            parsed = dict(original)
            raw_text = json.dumps(parsed, ensure_ascii=False)
        elif original is not None:
            raw_text = str(original)

        if isinstance(parsed, dict):
            template = parsed.get("prompt_template") or parsed.get("template")
            metadata = parsed.get("metadata")
            prompt_config = parsed.get("prompt_config")
            instructions = parsed.get("instructions")
        elif isinstance(parsed, list):
            instructions = parsed

        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "template": template,
            "metadata": metadata,
            "prompt_config": prompt_config,
            "instructions": instructions,
        }

    def _normalise_prompt(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not row:
            return None

        prompt_id = row.get("prompt_id") or row.get("promptId")
        try:
            prompt_id = int(prompt_id)
        except (TypeError, ValueError):
            logger.debug("Skipping prompt with invalid id: %r", row)
            return None

        payload = self._parse_prompt_payload(
            row.get("prompts_desc")
            or row.get("promptsDesc")
            or row.get("description")
        )

        entry: Dict[str, Any] = {
            "promptId": prompt_id,
            "promptName": row.get("prompt_name") or row.get("promptName"),
            "promptType": row.get("prompt_type") or row.get("promptType"),
            "prompts_desc": payload["parsed"] if payload["parsed"] is not None else payload["raw_text"],
            "raw_prompts_desc": payload["raw_text"],
            "template": payload["template"] or (payload["raw_text"] or None),
            "metadata": payload["metadata"],
            "prompt_config": payload["prompt_config"],
            "instructions": payload["instructions"],
            "linked_agents": self._coerce_linked_agents(
                row.get("prompt_linked_agents")
                or row.get("linked_agents")
                or row.get("agents")
            ),
        }

        for key in (
            "prompts_status",
            "created_date",
            "created_by",
            "last_modified_date",
            "last_modified_by",
            "version",
        ):
            if key in row and row[key] is not None:
                entry[key] = row[key]

        parsed_payload = payload["parsed"]
        if isinstance(parsed_payload, dict):
            templates = parsed_payload.get("templates")
            if isinstance(templates, list):
                entry["templates"] = templates
            if not payload["template"]:
                template_value = parsed_payload.get("prompt_template") or parsed_payload.get("template")
                if template_value:
                    entry["template"] = template_value

        if entry["template"] is None:
            entry.pop("template")

        return entry

    def _build_cache(self, prompt_rows: Optional[Iterable[Dict[str, Any]]]) -> None:
        rows = prompt_rows if prompt_rows is not None else self._fetch_prompt_rows()

        prompts: List[Dict[str, Any]] = []
        index: Dict[int, Dict[str, Any]] = {}
        agent_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for raw in rows:
            row = self._coerce_row(raw)
            if not row:
                continue
            prompt = self._normalise_prompt(row)
            if not prompt:
                continue
            prompts.append(prompt)
            index[prompt["promptId"]] = prompt
            for slug in prompt.get("linked_agents", []):
                agent_index[slug].append(prompt)

        self._prompts = prompts
        self._prompt_index = index
        self._agent_index = {slug: list(items) for slug, items in agent_index.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload prompts from the database."""

        self._build_cache(None)

    def prompts_by_id(self) -> Dict[int, Dict[str, Any]]:
        """Return a mapping of prompt identifiers to prompt metadata."""

        return {pid: dict(prompt) for pid, prompt in self._prompt_index.items()}

    def all_prompts(self) -> List[Dict[str, Any]]:
        """Return a list of all known prompts."""

        return [dict(prompt) for prompt in self._prompts]

    def get_prompt(self, prompt_id: Any) -> Optional[Dict[str, Any]]:
        """Return a single prompt by its identifier."""

        try:
            pid = int(prompt_id)
        except (TypeError, ValueError):
            return None
        prompt = self._prompt_index.get(pid)
        return dict(prompt) if prompt else None

    def prompts_for_agent(self, agent_type: Any) -> List[Dict[str, Any]]:
        """Return prompts linked to the supplied agent slug."""

        slug = self._slugify(agent_type)
        if not slug:
            return []
        return [dict(prompt) for prompt in self._agent_index.get(slug, [])]

    def prompt_library(self) -> Dict[str, Any]:
        """Aggregate prompt templates suitable for query decomposition."""

        templates: List[Any] = []
        seen: set[str] = set()

        for prompt in self._prompts:
            candidates: Iterator[Any] = iter(())
            if isinstance(prompt.get("templates"), list):
                candidates = iter(prompt["templates"])
            elif isinstance(prompt.get("prompts_desc"), dict):
                payload_templates = prompt["prompts_desc"].get("templates")
                if isinstance(payload_templates, list):
                    candidates = iter(payload_templates)

            for candidate in candidates:
                try:
                    serialised = json.dumps(candidate, sort_keys=True, default=str)
                except TypeError:
                    serialised = str(candidate)
                if serialised in seen:
                    continue
                seen.add(serialised)
                templates.append(candidate)

        return {"templates": templates}

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def deconstruct_query(
        self,
        user_query: str,
        prompt_library: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Use the configured LLM to parse ``user_query`` into structured data."""

        prompt_library = prompt_library or self.prompt_library()
        templates = prompt_library.get("templates") if isinstance(prompt_library, dict) else None
        prompt_library_context = json.dumps(templates or [], indent=2)

        logger.info("PromptEngine: Deconstructing query -> '%s'", user_query)

        prompt = f"""You are a query analysis expert. Your task is to deconstruct a user's natural language query into a structured JSON object based on a provided library of known query templates.

**Instructions:**
1.  Analyze the user's query.
2.  Infer values for `category`, `criteria`, `time_period`, and `filters`.
3.  If the query mentions ranking but no specific criteria are named (e.g., "rank the suppliers"), you MUST default the `criteria` to `["price", "delivery", "risk"]`.
4.  If no time period is mentioned, `time_period` must be `null`.
5.  Find the best matching template from the "PROMPT LIBRARY" below.
6.  Infer the values for the template's input parameters (like `category`, `criteria`, `time_period`, `filters`) from the user's query.
7.  Your response MUST be a single, valid JSON object containing the `template_id` and the extracted `parameters`. If a parameter cannot be inferred, use `null`.

**PROMPT LIBRARY:**
---
{prompt_library_context}
---

**USER QUERY:**
"{user_query}"

**JSON OUTPUT:**"""

        try:
            options = {"temperature": 0.0}
            if torch.cuda.is_available():
                options["num_gpu_layers"] = -1
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                format="json",
                options=options,
            )
            parsed_query = json.loads(response.get("response"))
            logger.info("PromptEngine: Query deconstructed successfully -> %s", parsed_query)
            return parsed_query
        except Exception as exc:  # pragma: no cover - runtime/network issues
            logger.exception("PromptEngine failed to deconstruct query: %s", exc)
            return None


__all__ = ["PromptEngine"]

