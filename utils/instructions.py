"""Utilities for parsing prompt and policy instruction payloads.

These helpers translate loosely structured prompt and policy descriptions
into normalised dictionaries that agents can consume when adapting their
behaviour at runtime.  Instruction sources may be JSON blobs, YAML-like
``key: value`` directives or nested dictionaries.  The functions in this
module attempt to coerce those representations into a single flattened map
with snake_case keys so callers can look up overrides deterministically.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Iterator, MutableMapping


def _try_parse_json(text: str) -> Any:
    """Return the parsed JSON value for ``text`` when possible."""

    try:
        return json.loads(text)
    except Exception:  # pragma: no cover - defensive parsing
        return None


def _coerce_literal(value: Any) -> Any:
    """Attempt to coerce ``value`` into a primitive literal."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""

        lowered = text.lower()
        if lowered in {"true", "yes", "y"}:
            return True
        if lowered in {"false", "no", "n"}:
            return False

        # Integers and floats are handled explicitly so downstream agents can
        # safely perform arithmetic without needing to repeat the parsing
        # logic.  ``float`` is attempted after ``int`` so values such as
        # "10.5" are preserved as non-integers.
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    return value


def _parse_kv_lines(text: str) -> Dict[str, Any]:
    """Parse ``key: value`` pairs from ``text`` into a dictionary."""

    entries: Dict[str, Any] = {}
    for line in text.replace(";", "\n").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = normalize_instruction_key(key)
        if not key:
            continue
        entries[key] = _coerce_literal(raw)
    return entries


def normalize_instruction_key(key: Any) -> str:
    """Return a snake_case representation of ``key`` suitable for lookups."""

    if key is None:
        return ""
    text = str(key).strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def _iter_instruction_payloads(source: Any) -> Iterator[MutableMapping[str, Any]]:
    """Yield dictionaries embedded within ``source``."""

    if source is None:
        return

    if isinstance(source, MutableMapping):
        yield source
        for value in source.values():
            yield from _iter_instruction_payloads(value)
        return

    if isinstance(source, (list, tuple, set)):
        for item in source:
            yield from _iter_instruction_payloads(item)
        return

    if isinstance(source, str):
        text = source.strip()
        if not text:
            return

        parsed = _try_parse_json(text)
        if parsed is None:
            parsed = _parse_kv_lines(text)

        if isinstance(parsed, MutableMapping):
            yield from _iter_instruction_payloads(parsed)
        elif isinstance(parsed, (list, tuple, set)):
            for item in parsed:
                yield from _iter_instruction_payloads(item)


def parse_instruction_sources(sources: Iterable[Any]) -> Dict[str, Any]:
    """Return a flattened dictionary of instructions from ``sources``.

    Parameters
    ----------
    sources:
        Iterable of raw instruction payloads.  Each element may be a dict,
        list, JSON string or ``key: value`` directive block.
    """

    instructions: Dict[str, Any] = {}

    for source in sources:
        for payload in _iter_instruction_payloads(source):
            for raw_key, raw_value in payload.items():
                key = normalize_instruction_key(raw_key)
                if not key:
                    continue

                # Attempt to parse nested JSON blobs expressed as strings so
                # callers do not need to repeat deserialisation logic.
                if isinstance(raw_value, str):
                    text = raw_value.strip()
                    if text:
                        parsed = _try_parse_json(text)
                        if isinstance(parsed, (dict, list, tuple, set)):
                            raw_value = parsed
                        else:
                            kv_payload = _parse_kv_lines(text)
                            if kv_payload:
                                raw_value = kv_payload
                            else:
                                raw_value = _coerce_literal(text)
                    else:
                        continue

                if raw_value is None:
                    continue

                # Prefer richer values (dict/list) over simple scalars when the
                # same key appears multiple times.  This ensures that policy
                # rules such as ``default_weights`` override earlier scalar
                # placeholders pulled from descriptions.
                if key in instructions:
                    existing = instructions[key]
                    if isinstance(existing, (dict, list, tuple, set)) and not isinstance(
                        raw_value, (dict, list, tuple, set)
                    ):
                        continue

                instructions[key] = raw_value

    return instructions

