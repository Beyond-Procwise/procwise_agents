"""Pydantic models for email dispatch endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class EmailDraftV2(BaseModel):
    """Email draft payload without legacy ``rfq_id`` exposure."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    identifier: Optional[str] = Field(
        default=None, description="Draft identifier used during dispatch."
    )
    supplier_id: Optional[str] = None
    recipients: Optional[List[EmailStr]] = None
    sender: Optional[EmailStr] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    action_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy(cls, value: Any) -> Any:
        """Map legacy keys to the canonical ``identifier`` field."""

        if isinstance(value, dict):
            if not value.get("identifier"):
                if value.get("unique_id"):
                    value["identifier"] = value["unique_id"]
                elif value.get("rfq_id"):
                    value["identifier"] = value["rfq_id"]
            value.pop("rfq_id", None)
        return value

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def resolve_identifier(self) -> Optional[str]:
        return self.identifier

    def resolved_identifier(self) -> Optional[str]:  # pragma: no cover - alias
        return self.resolve_identifier()

    def resolve_recipients(self) -> Optional[List[EmailStr]]:
        return self.recipients or None

    def resolved_recipients(self) -> Optional[List[EmailStr]]:  # pragma: no cover
        return self.resolve_recipients()

    def resolve_sender(self) -> Optional[EmailStr]:
        return self.sender or None

    def resolved_sender(self) -> Optional[EmailStr]:  # pragma: no cover
        return self.resolve_sender()

    def resolve_subject(self) -> Optional[str]:
        return self.subject or None

    def resolved_subject(self) -> Optional[str]:  # pragma: no cover
        return self.resolve_subject()

    def resolve_body(self) -> Optional[str]:
        return self.body or None

    def resolved_body(self) -> Optional[str]:  # pragma: no cover
        return self.resolve_body()

    @property
    def unique_id(self) -> Optional[str]:  # pragma: no cover - legacy alias
        return self.identifier


class EmailDispatchRequestV2(BaseModel):
    """Email dispatch request supporting V2 semantics without ``rfq_id``."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    identifier: Optional[str] = Field(
        default=None,
        alias="unique_id",
        description="Draft identifier (alias: unique_id).",
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Workflow context identifier for context-aware dispatch.",
    )
    recipients: Optional[List[EmailStr]] = None
    sender: Optional[EmailStr] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    action_id: Optional[str] = None

    draft: Optional[EmailDraftV2] = None
    drafts: Optional[List[EmailDraftV2]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        if not value.get("identifier") and not value.get("unique_id") and value.get(
            "rfq_id"
        ):
            value["identifier"] = value["rfq_id"]
        value.pop("rfq_id", None)

        drafts_value = value.get("drafts")
        if drafts_value and isinstance(drafts_value, dict):
            value["drafts"] = [drafts_value]

        return value

    # ------------------------------------------------------------------
    # Helper methods used by router logic
    # ------------------------------------------------------------------
    def resolve_identifier(self) -> Optional[str]:
        if self.identifier:
            return self.identifier
        if self.draft:
            identifier = self.draft.resolve_identifier()
            if identifier:
                return identifier
        if self.drafts:
            for entry in self.drafts:
                identifier = entry.resolve_identifier()
                if identifier:
                    return identifier
        return None

    def get_identifier(self) -> str:
        return self.resolve_identifier() or ""

    def resolved_identifier(self) -> Optional[str]:  # pragma: no cover - alias
        return self.resolve_identifier()

    def resolve_recipients(self) -> List[EmailStr]:
        if self.recipients:
            return list(self.recipients)
        if self.draft and self.draft.recipients:
            return list(self.draft.recipients)
        if self.drafts:
            for entry in self.drafts:
                recipients = entry.resolve_recipients()
                if recipients:
                    return list(recipients)
        return []

    def resolved_recipients(self) -> Optional[List[EmailStr]]:  # pragma: no cover
        recipients = self.resolve_recipients()
        return recipients or None

    def resolve_sender(self) -> Optional[EmailStr]:
        if self.sender:
            return self.sender
        if self.draft:
            return self.draft.resolve_sender()
        if self.drafts:
            for entry in self.drafts:
                sender = entry.resolve_sender()
                if sender:
                    return sender
        return None

    def resolved_sender(self) -> Optional[EmailStr]:  # pragma: no cover
        return self.resolve_sender()

    def resolve_subject(self) -> Optional[str]:
        if self.subject:
            return self.subject
        if self.draft:
            return self.draft.resolve_subject()
        if self.drafts:
            for entry in self.drafts:
                subject = entry.resolve_subject()
                if subject:
                    return subject
        return None

    def resolved_subject(self) -> Optional[str]:  # pragma: no cover
        return self.resolve_subject()

    def resolve_body(self) -> Optional[str]:
        if self.body:
            return self.body
        if self.draft:
            return self.draft.resolve_body()
        if self.drafts:
            for entry in self.drafts:
                body = entry.resolve_body()
                if body:
                    return body
        return None

    def resolved_body(self) -> Optional[str]:  # pragma: no cover
        return self.resolve_body()

    def resolve_action_id(self) -> Optional[str]:
        return self.action_id

    @property
    def unique_id(self) -> Optional[str]:  # pragma: no cover - legacy alias
        return self.identifier

    @unique_id.setter  # pragma: no cover - legacy alias
    def unique_id(self, value: Optional[str]) -> None:
        self.identifier = value

