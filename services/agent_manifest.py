"""Agent manifest service exposing task, policy, and knowledge bundles.

The manifest combines agent definitions, linked policies, and procurement
schema knowledge so that each agent receives a consistent operating picture in
both standalone and orchestrated executions.  Knowledge payloads are derived
from the canonical ``proc.*`` table definitions captured in
``docs/procurement_table_reference.md`` and from the normalised schema objects
available via :mod:`utils.procurement_schema`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.procurement_schema import PROCUREMENT_SCHEMAS

logger = logging.getLogger(__name__)


_PROC_RELATIONSHIPS: List[Dict[str, str]] = [
    {
        "from": "proc.contracts.supplier_id",
        "to": "proc.supplier.supplier_id",
        "description": "Contracts reference suppliers for supplier_name resolution.",
    },
    {
        "from": "proc.supplier.supplier_name",
        "to": "proc.purchase_order_agent.supplier_name",
        "description": "Supplier names link into the purchase order fact table.",
    },
    {
        "from": "proc.purchase_order_agent.po_id",
        "to": "proc.po_line_items_agent.po_id",
        "description": "Purchase orders own the associated line items.",
    },
    {
        "from": "proc.purchase_order_agent.po_id",
        "to": "proc.invoice_agent.po_id",
        "description": "Invoices reference the purchase order they belong to.",
    },
    {
        "from": "proc.invoice_agent.invoice_id",
        "to": "proc.invoice_line_items_agent.invoice_id",
        "description": "Invoice line items extend invoice level details.",
    },
    {
        "from": "proc.po_line_items_agent.item_description",
        "to": "proc.cat_product_mapping.product",
        "description": "Line items map onto the category taxonomy for enriched spend insights.",
    },
]


@dataclass
class AgentDefinition:
    agent_type: str
    description: str
    dependencies: List[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        token = self.agent_type
        token = token.replace("Agent", "") if token.endswith("Agent") else token
        normalised = []
        for char in token:
            if char.isupper() and normalised:
                normalised.append("_")
            normalised.append(char.lower())
        slug = "".join(normalised).strip("_")
        return slug or self.agent_type.lower()


class AgentManifestService:
    """Centralises agent metadata for context-aware execution."""

    def __init__(self, agent_nick) -> None:
        self.agent_nick = agent_nick
        self._definitions = self._load_definitions()
        self._policy_engine = getattr(agent_nick, "policy_engine", None)
        if self._policy_engine is None:
            logger.warning("AgentNick does not expose a policy engine; manifest policies may be empty")
        self._table_profiles = self._build_table_profiles()

    # ------------------------------------------------------------------
    # Definition loading
    # ------------------------------------------------------------------
    def _load_definitions(self) -> Dict[str, AgentDefinition]:
        definitions_path = Path(__file__).resolve().parents[1] / "agent_definitions.json"
        entries: Iterable[Dict[str, Any]] = []
        try:
            with open(definitions_path, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unable to load agent definitions from %s", definitions_path)
            return {}

        mapping: Dict[str, AgentDefinition] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            agent_type = entry.get("agentType") or entry.get("agent_type")
            description = entry.get("description") or ""
            dependencies = entry.get("dependencies") or []
            if not agent_type:
                continue
            definition = AgentDefinition(
                agent_type=str(agent_type),
                description=str(description or ""),
                dependencies=[str(dep) for dep in (dependencies or [])],
            )
            mapping[definition.slug] = definition
        return mapping

    # ------------------------------------------------------------------
    # Knowledge helpers
    # ------------------------------------------------------------------
    def _build_table_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        for table, schema in PROCUREMENT_SCHEMAS.items():
            profiles[table] = {
                "columns": list(schema.columns),
                "required": list(schema.required),
                "synonyms": {key: list(value) for key, value in schema.synonyms.items()},
            }
        # Contracts and supplier tables are referenced in the markdown but not in
        # ``PROCUREMENT_SCHEMAS``. Provide lightweight descriptors so manifests
        # can surface the relationship chain end-to-end.
        profiles.setdefault(
            "proc.contracts",
            {
                "columns": [
                    "contract_id",
                    "contract_title",
                    "supplier_id",
                    "contract_start_date",
                    "contract_end_date",
                    "total_contract_value",
                ],
                "required": ["contract_id", "supplier_id"],
                "synonyms": {"contract_id": ["contract number", "agreement number"]},
            },
        )
        profiles.setdefault(
            "proc.supplier",
            {
                "columns": [
                    "supplier_id",
                    "supplier_name",
                    "supplier_type",
                    "risk_score",
                    "default_currency",
                    "country",
                ],
                "required": ["supplier_id", "supplier_name"],
                "synonyms": {
                    "supplier_name": ["vendor", "trading name"],
                    "supplier_id": ["vendor id", "supplier number"],
                },
            },
        )
        profiles.setdefault(
            "proc.cat_product_mapping",
            {
                "columns": [
                    "product",
                    "category_level_2",
                    "category_level_3",
                    "category_level_4",
                    "category_level_5",
                ],
                "required": ["product", "category_level_2"],
                "synonyms": {"product": ["item", "description"]},
            },
        )
        return profiles

    def _policy_bundle_for_agent(self, slug: str) -> List[Dict[str, Any]]:
        engine = self._policy_engine
        if engine is None:
            return []
        try:
            policies = getattr(engine, "_policies", [])
        except Exception:  # pragma: no cover - defensive
            logger.exception("Policy engine did not expose policy cache")
            return []
        bundle: List[Dict[str, Any]] = []
        for policy in policies or []:
            linked = policy.get("policy_linked_agents") or []
            if slug in linked:
                bundle.append({
                    "policyId": policy.get("policyId"),
                    "policyName": policy.get("policyName"),
                    "policy_desc": policy.get("policy_desc"),
                    "policy_type": policy.get("policy_type"),
                })
        return bundle

    @staticmethod
    @lru_cache(maxsize=64)
    def _derive_workflow_hint(slug: str) -> Optional[str]:
        if not slug:
            return None
        if slug.endswith("_agent"):
            slug = slug[:-6]
        mapping = {
            "data_extraction": "document_extraction",
            "supplier_ranking": "supplier_ranking",
            "quote_evaluation": "quote_evaluation",
            "opportunity_miner": "opportunity_mining",
            "quote_comparison": "quote_comparison",
            "supplier_interaction": "supplier_interaction",
            "negotiation": "negotiation",
            "email_drafting": "email_drafting",
            "approvals": "approvals",
        }
        return mapping.get(slug)

    def build_manifest(self, agent_key: str) -> Dict[str, Any]:
        slug = self._normalise(agent_key)
        definition = self._definitions.get(slug)
        policy_bundle = self._policy_bundle_for_agent(slug)
        workflow = self._derive_workflow_hint(slug)
        knowledge = {
            "tables": self._table_profiles,
            "relationships": list(_PROC_RELATIONSHIPS),
            "workflow": workflow,
        }
        task_profile = {
            "agent_type": definition.agent_type if definition else agent_key,
            "description": definition.description if definition else "",
            "dependencies": definition.dependencies if definition else [],
            "workflow": workflow,
        }
        return {
            "task": task_profile,
            "policies": policy_bundle,
            "knowledge": knowledge,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(agent_key: str) -> str:
        if not agent_key:
            return ""
        token = agent_key.strip()
        if not token:
            return ""
        token = token.replace("-", "_")
        if token.lower() in ("data_extraction_agent", "data_extraction"):
            return "data_extraction"
        cleaned = []
        for char in token:
            if char.isupper():
                cleaned.append("_")
                cleaned.append(char.lower())
            else:
                cleaned.append(char)
        slug = "".join(cleaned).replace("__", "_").strip("_")
        if slug.endswith("_agent"):
            slug = slug[:-6]
        return slug


__all__ = ["AgentManifestService"]
