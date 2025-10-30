"""Render negotiation and approval emails from fixed workflow templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple


_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "resources" / "workflow"
_DEFAULT_CONTEXT_FILE = _TEMPLATE_DIR / "negotiation_email_context.json"


@dataclass(frozen=True)
class SupplierProfile:
    """Simple container for supplier-specific workflow data."""

    supplier_id: str
    name: str
    contact_name: str
    email: Optional[str] = None


class NegotiationEmailTemplateRenderer:
    """Utility that renders workflow emails using the mandated templates."""

    def __init__(self, context_data: Mapping[str, Any]):
        self._context = dict(context_data)
        suppliers = self._context.get("suppliers", {})
        self._supplier_profiles: MutableMapping[str, SupplierProfile] = {}
        for supplier_id, payload in suppliers.items():
            if not isinstance(payload, Mapping):
                continue
            name = str(payload.get("name") or supplier_id).strip()
            contact_name = str(
                payload.get("contact_name") or payload.get("name") or "Supplier"
            ).strip()
            email = payload.get("email")
            self._supplier_profiles[supplier_id] = SupplierProfile(
                supplier_id=supplier_id,
                name=name,
                contact_name=contact_name,
                email=str(email).strip() if isinstance(email, str) else None,
            )

    @classmethod
    def from_default(cls) -> "NegotiationEmailTemplateRenderer":
        """Load the renderer with the repository default workflow context."""

        with _DEFAULT_CONTEXT_FILE.open("r", encoding="utf-8") as handle:
            context_data = json.load(handle)
        return cls(context_data)

    # ------------------------------------------------------------------
    # Public render helpers
    # ------------------------------------------------------------------
    def initial_quote_request(
        self, supplier: Mapping[str, Any]
    ) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        specific_items = self._resolve_specific_items(supplier)
        subject = f"Request for Pricing Review – {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            f"As part of our regular review of supplier arrangements, we are assessing the current pricing and "
            f"commercial terms for {commodity_name}. This review is intended to ensure consistency and competitiveness "
            "across our supply base.\n\n"
            f"We would appreciate it if you could review your existing pricing and provide your most up-to-date and "
            f"competitive proposal for {specific_items}. To assist in our evaluation, please complete the attached pricing "
            "table with the relevant details, including any applicable delivery schedules or payment terms.\n\n"
            "Thank you for your continued support and collaboration. We value the relationship with your organisation and "
            "look forward to receiving your updated submission.\n\n"
            f"Kind regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def renegotiation_request(self, supplier: Mapping[str, Any]) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        targets = self._context.get("negotiation_targets", {})
        unit_pct = self._format_percent(targets.get("unit_price_reduction_pct", 5))
        volume_pct = self._format_percent(targets.get("volume_increase_pct", 10))
        subject = f"Request for Revised Proposal – {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            "Thank you for providing your proposal. Following our review, we have identified a few areas where we "
            "believe there may be scope for further improvement to enhance the competitiveness of your offer.\n\n"
            "In particular, please consider whether adjustments could be made in the following areas:\n\n"
            f"Unit pricing: a potential reduction of around {unit_pct}, based on current market benchmarks.\n\n"
            f"Volume commitment: we may be able to increase order volumes by approximately {volume_pct}, subject to more favourable pricing.\n\n"
            "Commercial terms: please review whether delivery lead times or payment terms could be further optimised.\n\n"
            "We would be grateful if you could review these points and provide an updated proposal reflecting any possible adjustments.\n\n"
            "Thank you once again for your continued engagement and support.\n\n"
            f"Best regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def final_review_request(self, supplier: Mapping[str, Any]) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        targets = self._context.get("negotiation_targets", {})
        low = self._format_percent(targets.get("final_range_low_pct", 4))
        high = self._format_percent(targets.get("final_range_high_pct", 6))
        subject = f"Final Proposal Review – {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            "Thank you for submitting your revised proposal and for your continued cooperation during this review process.\n\n"
            "After a detailed comparison against market data and other supplier submissions, we believe your offer remains "
            "very competitive, though there may still be limited potential for further refinement in the following areas:\n\n"
            f"Unit price alignment, where prevailing market levels indicate a potential range of {low}–{high} lower.\n\n"
            "Overall cost efficiency, which could be achieved through minor adjustments to the pricing structure or terms.\n\n"
            "If you are able to incorporate any final revisions within these parameters, we would be pleased to consider your proposal for award.\n\n"
            "Thank you once again for your time and partnership.\n\n"
            f"Kind regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def follow_up_neutral(
        self,
        supplier: Mapping[str, Any],
        *,
        original_email_date: date,
    ) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        subject = f"Follow-Up: Request for Pricing Review – {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            "I hope you are well. I wanted to follow up on my earlier email regarding the pricing review for "
            f"{commodity_name}, sent on {original_email_date:%d %B %Y}. We haven’t yet received your response and wanted to ensure that it has reached you.\n\n"
            "If you have already replied, please simply reply to this email (rather than starting a new thread) so we can easily track your submission. If not, we would be grateful if you could confirm your latest pricing or advise when we can expect your response.\n\n"
            "Your input is important as we are currently finalising our evaluation and would like to ensure your proposal is considered.\n\n"
            "Thank you for your attention and cooperation — I look forward to your reply.\n\n"
            f"Kind regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def follow_up_firm(
        self,
        supplier: Mapping[str, Any],
        *,
        original_email_date: date,
        deadline: date,
    ) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        subject = f"Follow-Up Reminder: Response Required – {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            "I hope this message finds you well. I’m following up on my earlier email dated "
            f"{original_email_date:%d %B %Y} regarding the pricing review for {commodity_name}. We haven’t yet received a response and wanted to check whether you have had the opportunity to review our request.\n\n"
            "We are in the process of finalising our supplier evaluation and would appreciate your confirmation or updated proposal no later than "
            f"{deadline:%d %B %Y}.\n\n"
            "If you have already replied, please simply respond to this email thread so that your submission is properly captured in our records.\n\n"
            "Thank you for your prompt attention to this matter. We value your partnership and would like to ensure your organisation is included in this review.\n\n"
            f"Kind regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def follow_up_collaborative(
        self,
        supplier: Mapping[str, Any],
        *,
        original_email_date: date,
    ) -> Tuple[str, str]:
        supplier_id = self._extract_supplier_id(supplier)
        profile = self._supplier_profiles.get(supplier_id)
        contact_name = self._resolve_contact_name(supplier, profile)
        commodity_name = self._resolve_commodity_name(supplier)
        subject = f"Gentle Reminder: Pricing Review for {commodity_name}"
        body = (
            f"Dear {contact_name},\n\n"
            "I hope you’re keeping well. I wanted to gently follow up on my earlier note regarding the pricing review for "
            f"{commodity_name}. We’re currently progressing through our evaluation and would very much like to include your updated proposal as part of this process.\n\n"
            "If you’ve already shared your response, please reply directly to this email so we can confirm receipt. If not, any update or indication of timing would be greatly appreciated.\n\n"
            "We truly value the relationship we have with your team and look forward to continuing our collaboration on this.\n\n"
            "Warm regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}"
        )
        return subject, body

    def approval_request_email(self) -> Tuple[str, str]:
        commodity_name = self._context.get("commodity_name", "Sourcing Request")
        approval = self._context.get("approval", {})
        supplier_name = approval.get("supplier_name", "Preferred Supplier")
        reference_id = approval.get("reference_id", "WF-REFERENCE")
        subject = (
            "Request for Approval – "
            f"{commodity_name} / {supplier_name} / {reference_id}"
        )
        approver_name = self._context.get("approver_name", "Approver")
        project_name = approval.get("project_name", commodity_name)
        suppliers_compared = approval.get("suppliers_compared", [])
        suppliers_text = ", ".join(map(str, suppliers_compared)) if suppliers_compared else "N/A"
        total_value = approval.get("total_value", "Value TBC")
        key_justification = approval.get(
            "key_justification", "Most competitive price with favourable delivery terms"
        )
        link_id = approval.get("link_id", reference_id)
        body = (
            f"Dear {approver_name},\n\n"
            "I hope you’re well. I’m writing to request your approval for proceeding with the recommended proposal relating to "
            f"{project_name}.\n\n"
            "As part of the evaluation process, we have reviewed multiple supplier quotations through our quote comparison dashboard in the Procwise Agentic Procurement System. Based on this assessment, "
            f"{supplier_name} has been identified as providing the most competitive and commercially balanced offer.\n\n"
            "A brief summary of the evaluation is provided below:\n\n"
            f"Suppliers Compared: {suppliers_text}\n\n"
            f"Recommended Supplier: {supplier_name}\n\n"
            f"Total Proposed Value: {total_value}\n\n"
            f"Key Justification: {key_justification}\n\n"
            "For full details, please review the quote comparison summary and supporting documentation available here:\n"
            f"View in Procwise Approval Page: https://procwise.app/approvals/{link_id}\n\n"
            "If you’re satisfied with the recommendation, please either:\n\n"
            "Approve directly in the Procwise system using the link above, or\n\n"
            "Simply reply to this email with “Approved”, and we will record it in the approval log.\n\n"
            "Thank you for taking the time to review this request. Please let me know if you’d like any additional information or supporting analysis before approving.\n\n"
            "Kind regards,\n"
            f"{self._context['buyer_name']}\n"
            f"{self._context['buyer_position']}\n"
            f"{self._context['buyer_organisation']}\n"
            f"{self._context['buyer_contact_information']}"
        )
        return subject, body

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_supplier_id(self, supplier: Mapping[str, Any]) -> str:
        candidate = supplier.get("supplier_id") or supplier.get("id") or "SUP-UNKNOWN"
        return str(candidate)

    def _resolve_contact_name(
        self,
        supplier: Mapping[str, Any],
        profile: Optional[SupplierProfile],
    ) -> str:
        for key in ("contact", "contact_name", "primary_contact"):
            value = supplier.get(key)
            if value:
                return str(value)
        if profile and profile.contact_name:
            return profile.contact_name
        name = supplier.get("name") or (profile.name if profile else None)
        return str(name) if name else "Supplier Team"

    def _resolve_commodity_name(self, supplier: Mapping[str, Any]) -> str:
        return str(
            supplier.get("commodity")
            or supplier.get("item_name")
            or self._context.get("commodity_name", "Sourcing Requirement")
        )

    def _resolve_specific_items(self, supplier: Mapping[str, Any]) -> str:
        value = (
            supplier.get("specific_items")
            or supplier.get("items")
            or self._context.get("specific_items")
            or self._resolve_commodity_name(supplier)
        )
        return str(value)

    @staticmethod
    def _format_percent(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "X%"
        return f"{numeric:.0f}%"


__all__ = ["NegotiationEmailTemplateRenderer"]
