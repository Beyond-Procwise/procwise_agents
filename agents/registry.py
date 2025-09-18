"""Agent registry with alias resolution for ProcWise agents."""

from __future__ import annotations

from typing import Any, Dict, Optional


class AgentRegistry(dict):
    """Dictionary-like registry that supports alias lookups.

    The registry stores agents keyed by their canonical slug (typically the
    snake_case variant of the agent class name).  Additional aliases can be
    registered to allow legacy identifiers such as CamelCase class names to
    resolve to the same instance without duplicating entries in the mapping.
    """

    def __init__(
        self,
        initial: Optional[Dict[str, Any]] = None,
        *,
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(initial or {})
        self._aliases: Dict[str, str] = {}
        if aliases:
            self.add_aliases(aliases)

    # ------------------------------------------------------------------
    # Alias management helpers
    # ------------------------------------------------------------------
    def add_alias(self, alias: Optional[str], target: Optional[str]) -> None:
        """Register ``alias`` as an alternative key for ``target``."""

        if not alias or not target:
            return
        if target not in self:
            return
        # Preserve the original alias and a lower-case variant for
        # case-insensitive lookup.  ``alias`` may already be lower-case so the
        # two assignments will harmlessly overlap.
        self._aliases[str(alias)] = target
        if isinstance(alias, str):
            self._aliases[alias.lower()] = target

    def add_aliases(self, mapping: Dict[str, str]) -> None:
        for alias, target in (mapping or {}).items():
            self.add_alias(alias, target)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def _resolve_key(self, key: Any) -> Optional[str]:
        if key is None:
            return None
        if super().__contains__(key):
            return key
        if isinstance(key, str):
            lower = key.lower()
            if super().__contains__(lower):
                return lower
            alias_target = self._aliases.get(key) or self._aliases.get(lower)
            if alias_target and super().__contains__(alias_target):
                return alias_target
        return None

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return self._resolve_key(key) is not None

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        resolved = self._resolve_key(key)
        if resolved is None:
            raise KeyError(key)
        return super().__getitem__(resolved)

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        resolved = self._resolve_key(key)
        if resolved is None:
            return default
        return super().get(resolved, default)

    def keys(self):  # type: ignore[override]
        return super().keys()

    def values(self):  # type: ignore[override]
        return super().values()

    def items(self):  # type: ignore[override]
        return super().items()

    def aliases(self) -> Dict[str, str]:
        """Return a copy of the registered alias mapping."""

        return dict(self._aliases)

