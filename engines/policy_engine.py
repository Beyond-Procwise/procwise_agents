"""Database-backed policy loading utilities.

The original implementation read bundled JSON fixtures for supplier ranking
and opportunity policies.  Runtime environments now mandate that policy
configuration is sourced directly from the PostgreSQL ``proc.policy`` table
so that agent behaviour reflects the latest governance rules without
requiring code deploys.  This module therefore provides a lightweight
repository for policy metadata with convenience helpers for common lookups.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Load and cache policy definitions from ``proc.policy``."""

    SUPPLIER_POLICY_SLUGS = {
        "weight_allocation_policy",
        "categorical_scoring_policy",
        "normalization_direction_policy",
    }

    OPPORTUNITY_KEYWORDS = {
        "opportunity",
        "oppfinderpolicy",
        "contract_expiry",
        "maverick_spend",
        "volume_consolidation",
        "supplier_risk",
        "duplicate_supplier",
        "category_overspend",
        "inflation_pass_through",
        "unused_contract_value",
        "supplier_performance",
        "esg_opportunity",
        "price_benchmark_variance",
    }

    def __init__(
        self,
        agent_nick: Optional[Any] = None,
        connection_factory: Optional[Any] = None,
        policy_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the engine and load policies.

        Parameters
        ----------
        agent_nick:
            When provided, the agent container is used to obtain a live
            database connection via ``get_db_connection``.
        connection_factory:
            Optional explicit factory returning a DB-API compatible
            connection.  This parameter is primarily intended for unit
            tests where lightweight stubs are preferable.
        policy_rows:
            Iterable of dictionaries mirroring the ``proc.policy`` schema.
            When supplied the rows are used instead of querying the
            database, enabling deterministic fixtures in tests.
        """

        self.agent_nick = agent_nick
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif agent_nick is not None:
            self._connection_factory = getattr(agent_nick, "get_db_connection", None)
        else:
            self._connection_factory = None

        self._policies: List[Dict[str, Any]] = self._load_policies(policy_rows)
        self._slug_index: Dict[str, Dict[str, Any]] = {}
        for policy in self._policies:
            for alias in policy.get("aliases", set()):
                if alias not in self._slug_index:
                    self._slug_index[alias] = policy

        self.supplier_policies = self._collect_supplier_policies()
        self.opportunity_policies = self._collect_opportunity_policies()
        self._normalise_weight_policy()
        logger.info("PolicyEngine loaded %d policies", len(self._policies))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = str(value)
        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"(?<!^)(?=[A-Z0-9])", "_", text)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text).lower().strip("_")
        return slug or None

    @staticmethod
    def _coerce_details(payload: Any) -> Dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode(errors="ignore")
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except Exception:
                logger.debug("Unable to parse policy_details JSON: %s", text)
                return {"text": text}
            if isinstance(parsed, dict):
                return parsed
            return {}
        logger.debug("Unsupported policy_details payload: %r", payload)
        return {}

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
        finally:
            close = getattr(resource, "close", None)
            if callable(close):  # pragma: no cover - defensive cleanup
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close policy connection")

    def _fetch_policy_rows(self) -> List[Dict[str, Any]]:
        columns = [
            "policy_id",
            "policy_name",
            "policy_type",
            "policy_desc",
            "policy_details",
            "policy_linked_agents",
        ]
        with self._connect() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT policy_id, policy_name, policy_type, policy_desc,
                               policy_details, policy_linked_agents
                        FROM proc.policy
                        WHERE COALESCE(policy_status, 1) = 1
                        """
                    )
                    rows = cursor.fetchall()
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
            except Exception:
                logger.exception("Failed to load policies from database")
                return []
        return [dict(zip(columns, row)) for row in rows] if rows else []

    def _normalise_policy_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        record = dict(row)
        details = self._coerce_details(record.get("policy_details"))
        policy_id = record.get("policy_id")
        policy_name = record.get("policy_name")
        policy_desc = record.get("policy_desc")
        slug = (
            self._slugify(policy_name)
            or self._slugify(details.get("policy_name"))
            or self._slugify(details.get("identifier"))
            or self._slugify(policy_desc)
            or self._slugify(policy_id)
        )
        aliases = {
            value
            for value in (
                self._slugify(policy_name),
                self._slugify(policy_desc),
                self._slugify(policy_id),
                self._slugify(details.get("policy_name")),
                self._slugify(details.get("identifier")),
                self._slugify(details.get("policy_identifier")),
            )
            if value
        }
        rules = details.get("rules") if isinstance(details, dict) else {}
        if isinstance(rules, dict):
            aliases.update(
                {
                    alias
                    for alias in (
                        self._slugify(rules.get("name")),
                        self._slugify(rules.get("policy_name")),
                    )
                    if alias
                }
            )
        linked_agents = self._coerce_linked_agents(record.get("policy_linked_agents"))
        aliases.update(linked_agents)
        if slug:
            aliases.add(slug)
        identifier = (
            details.get("policy_identifier")
            or details.get("identifier")
            or policy_id
        )
        policy_identifier = str(identifier) if identifier is not None else None
        policy = {
            "policyId": policy_identifier,
            "policyName": policy_name,
            "policy_desc": policy_desc,
            "policy_type": record.get("policy_type"),
            "details": details if isinstance(details, dict) else {},
            "aliases": aliases,
            "slug": slug or self._slugify(policy_identifier) or "",
            "policy_linked_agents": linked_agents,
            "raw_row": record,
        }
        return policy

    def _load_policies(
        self, override_rows: Optional[Iterable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        rows = list(override_rows) if override_rows is not None else self._fetch_policy_rows()
        policies: List[Dict[str, Any]] = []
        for row in rows:
            policy = self._normalise_policy_row(row)
            if policy:
                policies.append(policy)
        return policies

    def _collect_supplier_policies(self) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for slug in self.SUPPLIER_POLICY_SLUGS:
            policy = self.get_policy(slug)
            if policy:
                collected.append(policy)
        if collected:
            return collected
        for policy in self._policies:
            if any(token.startswith("supplier") for token in policy.get("aliases", [])):
                collected.append(policy)
        return collected

    def _collect_opportunity_policies(self) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for policy in self._policies:
            aliases = policy.get("aliases", set())
            if aliases & self.OPPORTUNITY_KEYWORDS:
                collected.append(policy)
        return collected

    def _normalise_weight_policy(self) -> None:
        """Ensure default supplier ranking weights sum to 1."""
        weight_policy = self.get_policy("weight_allocation_policy")
        if not weight_policy:
            return
        rules = weight_policy.get("details", {}).get("rules", {})
        weights = rules.get("default_weights", {}) if isinstance(rules, dict) else {}
        if not isinstance(weights, dict):
            return
        total = sum(float(value) for value in weights.values())
        if total and abs(total - 1.0) > 1e-6:
            for key, value in list(weights.items()):
                try:
                    weights[key] = round(float(value) / total, 4)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    weights[key] = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_policies(self) -> List[Dict[str, Any]]:
        return list(self._policies)

    def iter_policies(self) -> Iterator[Dict[str, Any]]:
        return iter(self._policies)

    def get_policy(self, slug: str) -> Optional[Dict[str, Any]]:
        key = self._slugify(slug)
        if not key:
            return None
        policy = self._slug_index.get(key)
        if policy:
            return policy
        for candidate in self._policies:
            if key in candidate.get("aliases", set()):
                return candidate
        return None

    def validate_workflow(self, workflow_name: str, user_id: str, input_data: dict) -> dict:
        """Validate a workflow against policy rules."""

        if workflow_name == "supplier_ranking":
            criteria = input_data.get("criteria")
            if not criteria:
                weight_policy = self.get_policy("weight_allocation_policy") or {}
                rules = weight_policy.get("details", {}).get("rules", {})
                default_weights = (
                    rules.get("default_weights", {}) if isinstance(rules, dict) else {}
                )
                criteria = list(default_weights.keys())
            intent = {
                "template_id": "rank_by_criteria",
                "parameters": {"criteria": criteria},
            }
            allowed, reason, _ = self.validate_and_apply(intent)
            return {"allowed": allowed, "reason": reason}

        return {"allowed": True, "reason": "No policy checks"}

    def validate_and_apply(self, intent: dict) -> tuple[bool, str, dict]:
        logger.debug("PolicyEngine validating intent: %s", intent)
        if not intent or not intent.get("parameters"):
            return False, "Query intent could not be determined.", intent

        if intent.get("template_id") == "rank_by_criteria":
            criteria = intent.get("parameters", {}).get("criteria")
            if not criteria or not isinstance(criteria, list):
                reason = (
                    "Policy validation failed: Ranking requires at least one criterion"
                )
                return False, reason, intent

            weight_policy = self.get_policy("weight_allocation_policy")
            if not weight_policy:
                return False, "Weight allocation policy not found.", intent

            rules = weight_policy.get("details", {}).get("rules", {})
            weight_map = rules.get("default_weights", {}) if isinstance(rules, dict) else {}
            defined_weights = set(weight_map.keys())
            for criterion in criteria:
                if criterion not in defined_weights:
                    reason = (
                        f"Policy validation failed: Ranking criterion '{criterion}' "
                        "has no defined weight."
                    )
                    return False, reason, intent

        logger.debug("PolicyEngine intent validated successfully")
        return True, "Validation successful.", intent
