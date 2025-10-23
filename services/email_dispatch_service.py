"""Email dispatch service for sending stored RFQ drafts."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.email_tracking import (
    build_tracking_comment,
    embed_unique_id_in_email_body,
    ensure_tracking_prefix,
    strip_tracking_comment,
)
from utils.gpu import configure_gpu

from services.backend_scheduler import BackendScheduler

from .email_dispatch_chain_store import (
    record_dispatch as record_workflow_dispatch,
    register_dispatch as register_dispatch_chain,
)
from .email_service import EmailService
from .email_thread_store import (
    DEFAULT_THREAD_TABLE,
    ensure_thread_table,
    record_thread_mapping,
    sanitise_thread_table_name,
)

configure_gpu()

logger = logging.getLogger(__name__)

_DEFAULT_THREAD_TABLE = DEFAULT_THREAD_TABLE


class DraftNotFoundError(ValueError):
    """Raised when a draft cannot be located for dispatch."""


class EmailDispatchService:
    """Send persisted RFQ drafts via Amazon SES and update their status."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.email_service = EmailService(agent_nick)
        self.settings = agent_nick.settings
        self.logger = logging.getLogger(__name__)
        self._thread_table_name = sanitise_thread_table_name(
            getattr(self.settings, "email_thread_table", None),
            logger=self.logger,
        )
        self._thread_table_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve_workflow_id(self, identifier: str) -> Optional[str]:
        """Best-effort lookup of the workflow identifier for ``identifier``.

        The email dispatch endpoint must not generate ad-hoc workflow
        identifiers.  Instead it should re-use the identifier that was
        attached to the drafted email.  This helper mirrors the first phase
        of :meth:`send_draft` – fetching and hydrating the latest draft – but
        bails out before attempting to send the message.  It returns ``None``
        when the draft cannot be located or when the stored payload does not
        expose a workflow token.
        """

        identifier = (identifier or "").strip()
        if not identifier:
            return None

        try:
            with self.agent_nick.get_db_connection() as conn:
                draft_row = self._fetch_latest_draft(conn, identifier)
        except Exception:  # pragma: no cover - database connectivity
            self.logger.exception(
                "Failed to resolve workflow for dispatch identifier %s", identifier
            )
            return None

        if not draft_row:
            return None

        draft = self._hydrate_draft(draft_row)

        workflow_token = self._extract_workflow_identifier(draft)
        if workflow_token:
            return workflow_token

        metadata = draft.get("metadata")
        if isinstance(metadata, dict):
            workflow_token = self._extract_workflow_identifier(metadata)
            if workflow_token:
                return workflow_token

        dispatch_meta = draft.get("dispatch_metadata")
        if isinstance(dispatch_meta, dict):
            workflow_token = self._extract_workflow_identifier(dispatch_meta)
            if workflow_token:
                return workflow_token

        return None

    def dispatch_from_context(
        self,
        workflow_id: Optional[str],
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch a draft resolved from ``workflow_id`` with optional overrides."""

        overrides = dict(overrides or {})

        identifier = self._normalise_identifier(overrides.pop("identifier", None))
        if not identifier:
            identifier = self._normalise_identifier(overrides.pop("unique_id", None))
        if not identifier:
            identifier = self._normalise_identifier(overrides.pop("rfq_id", None))

        workflow_hint = self._normalise_identifier(
            overrides.pop("workflow_id", None) or workflow_id
        )

        recipients_override = overrides.pop("recipients", None)
        if isinstance(recipients_override, str) or isinstance(
            recipients_override, Sequence
        ) and not isinstance(recipients_override, (bytes, bytearray)):
            recipients = self._normalise_recipients(
                recipients_override if isinstance(recipients_override, str) else recipients_override
            )
        else:
            recipients = None

        sender_override = overrides.pop("sender", None)
        if sender_override is not None:
            sender_override = str(sender_override).strip() or None

        subject_override = overrides.pop("subject_override", None)
        if subject_override is not None:
            subject_override = str(subject_override).strip() or None

        body_override = overrides.pop("body_override", None)
        if body_override is not None:
            body_override = str(body_override)

        if overrides:
            self.logger.debug(
                "Ignoring unsupported dispatch overrides: %s", sorted(overrides.keys())
            )

        if not identifier:
            if not workflow_hint:
                raise ValueError(
                    "workflow_id or identifier is required to dispatch an email draft"
                )
            identifier = self._resolve_identifier_for_workflow(workflow_hint)
            if not identifier:
                raise DraftNotFoundError(
                    f"No dispatchable draft found for workflow {workflow_hint}"
                )

        return self.send_draft(
            identifier=identifier,
            recipients=recipients,
            sender=sender_override,
            subject_override=subject_override,
            body_override=body_override,
        )

    def send_draft(
        self,
        identifier: str,
        recipients: Optional[Iterable[str]] = None,
        sender: Optional[str] = None,
        subject_override: Optional[str] = None,
        body_override: Optional[str] = None,
        attachments: Optional[List[Tuple[bytes, str]]] = None,
    ) -> Dict[str, Any]:
        """Send the latest draft for ``identifier`` (unique_id preferred)."""

        identifier = (identifier or "").strip()
        if not identifier:
            raise ValueError("unique_id is required to send an email draft")

        with self.agent_nick.get_db_connection() as conn:
            draft_row = self._fetch_latest_draft(conn, identifier)
            if draft_row is None:
                raise DraftNotFoundError(
                    f"No stored draft found for identifier {identifier}. "
                    "Please use unique_id format (PROC-WF-XXXXXX) or ensure the draft exists."
                )

            draft = self._hydrate_draft(draft_row)

            unique_id = draft.get("unique_id")
            if not unique_id:
                raise ValueError(
                    "Draft found but missing unique_id. Draft data may be corrupted."
                )

            review_status = str(draft.get("review_status") or "").upper()
            if review_status not in {"APPROVED", "SENT"}:
                raise ValueError(
                    "Draft must be approved (review_status=APPROVED) before dispatch"
                )

            rfq_identifier = self._normalise_identifier(draft.get("rfq_id"))

            recipient_source = (
                recipients if recipients is not None else draft.get("recipients")
            )
            recipient_list = self._normalise_recipients(recipient_source)
            if not recipient_list and draft.get("receiver"):
                recipient_list = self._normalise_recipients([draft["receiver"]])

            if not recipient_list:
                raise ValueError(
                    "At least one recipient email is required to send the draft"
                )

            sender_candidate = sender if sender is not None else draft.get("sender")
            sender_email = (
                str(sender_candidate).strip() if sender_candidate is not None else ""
            )
            if not sender_email:
                default_sender = getattr(
                    self.settings, "DEFAULT_SENDER", None
                ) or getattr(self.settings, "ses_default_sender", None)
                sender_email = str(default_sender).strip() if default_sender else ""
            if not sender_email:
                raise ValueError(
                    "Sender email address is required and DEFAULT_SENDER is not configured"
                )

            if subject_override is not None:
                subject_candidate = subject_override
            else:
                subject_candidate = draft.get("subject")
            subject_str = str(subject_candidate).strip() if subject_candidate else ""
            subject = subject_str or f"{unique_id} – Request for Quotation"

            body_source = (
                body_override if body_override is not None else draft.get("body")
            )
            body_text = str(body_source).strip() if body_source else ""

            draft_metadata_source = (
                draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
            )
            draft_metadata = dict(draft_metadata_source)
            dispatch_run_id = uuid.uuid4().hex
            draft_metadata["dispatch_token"] = dispatch_run_id
            draft_metadata["run_id"] = dispatch_run_id
            try:
                body, backend_metadata = self._ensure_tracking_annotation(
                    body_text,
                    unique_id=unique_id,
                    supplier_id=draft.get("supplier_id"),
                    dispatch_token=dispatch_run_id,
                    run_id=draft.get("run_id") or dispatch_run_id,
                    workflow_id=draft.get("workflow_id"),
                )
            except ValueError as err:
                self.logger.warning(
                    "Skipping tracking annotation for unique_id %s: %s",
                    unique_id,
                    err,
                )
                body = body_text
                backend_metadata = {"unique_id": identifier}

            backend_metadata.setdefault("unique_id", unique_id)
            supplier_id_value = draft.get("supplier_id")
            workflow_id_value = draft.get("workflow_id")
            if supplier_id_value:
                backend_metadata.setdefault("supplier_id", supplier_id_value)
            if workflow_id_value:
                backend_metadata.setdefault("workflow_id", workflow_id_value)
            backend_metadata["run_id"] = dispatch_run_id

            dispatch_payload = dict(draft)
            dispatch_payload["metadata"] = draft_metadata
            dispatch_payload.update(
                {
                    "subject": subject,
                    "body": body,
                    "recipients": recipient_list,
                    "receiver": (
                        recipient_list[0] if recipient_list else draft.get("receiver")
                    ),
                    "contact_level": 1 if recipient_list else 0,
                    "sender": sender_email,
                    "dispatch_metadata": backend_metadata,
                    "dispatch_run_id": dispatch_run_id,
                }
            )
            dispatch_payload.setdefault(
                "workflow_id", backend_metadata.get("workflow_id")
            )
            dispatch_payload.setdefault("unique_id", unique_id)
            if backend_metadata.get("run_id"):
                dispatch_payload.setdefault("run_id", backend_metadata.get("run_id"))
            if backend_metadata.get("supplier_id"):
                dispatch_payload.setdefault(
                    "supplier_id", backend_metadata.get("supplier_id")
                )
            dispatch_payload.setdefault("metadata", {})
            if isinstance(dispatch_payload["metadata"], dict):
                dispatch_payload["metadata"].setdefault("run_id", dispatch_run_id)
                dispatch_payload["metadata"]["dispatch_token"] = dispatch_run_id

            headers = {
                "X-Procwise-Workflow-Id": backend_metadata.get("workflow_id"),
                "X-Procwise-Unique-Id": unique_id,
            }
            mailbox_header = draft.get("mailbox") or getattr(
                self.settings, "supplier_mailbox", None
            )
            if mailbox_header:
                backend_metadata["mailbox"] = mailbox_header
                headers["X-Procwise-Mailbox"] = mailbox_header
            if backend_metadata.get("supplier_id"):
                headers["X-Procwise-Supplier-Id"] = backend_metadata.get("supplier_id")

            payload_metadata = dict(backend_metadata)
            provider_payload: Dict[str, Any] = {
                "to": list(recipient_list),
                "from": sender_email,
                "subject": subject,
                "metadata": payload_metadata,
            }
            if re.search(r"<[^>]+>", body):
                provider_payload["html"] = body
            else:
                provider_payload["text"] = body

            dispatch_payload["payload"] = provider_payload

            send_result = self.email_service.send_email(
                subject,
                body,
                recipient_list,
                sender_email,
                attachments,
                headers=headers,
            )

            sent = send_result.success
            message_id = send_result.message_id

            self._update_draft_status(
                conn,
                draft_row,
                dispatch_payload,
                recipient_list,
                sent,
            )

            self._update_action_sent_status(
                conn,
                dispatch_payload,
                unique_id,
                bool(sent),
            )

            conn.commit()

            dispatch_payload["sent_status"] = bool(sent)
            if sent:
                dispatch_payload["sent_on"] = datetime.utcnow().isoformat()
                dispatch_payload["message_id"] = message_id
                logger.info(
                    "Dispatched supplier email workflow=%s unique_id=%s supplier=%s message_id=%s",
                    backend_metadata.get("workflow_id"),
                    unique_id,
                    draft.get("supplier_id"),
                    message_id,
                )
                canonical_rfq_identifier = (
                    rfq_identifier.upper() if isinstance(rfq_identifier, str) else None
                )

                try:
                    try:
                        self._record_thread_mapping(
                            conn,
                            message_id,
                            unique_id,
                            draft.get("supplier_id"),
                            recipient_list,
                            rfq_id=unique_id,
                        )
                    except TypeError:
                        self._record_thread_mapping(  # type: ignore[misc]
                            conn,
                            message_id,
                            unique_id,
                            draft.get("supplier_id"),
                            recipient_list,
                        )
                    conn.commit()
                except Exception:  # pragma: no cover - defensive
                    self.logger.exception(
                        "Failed to persist thread mapping for unique_id %s", unique_id
                    )
                try:
                    register_dispatch_chain(
                        conn,
                        rfq_id=canonical_rfq_identifier or unique_id,
                        message_id=message_id,
                        subject=subject,
                        body=body,
                        thread_index=draft.get("thread_index"),
                        supplier_id=draft.get("supplier_id"),
                        workflow_ref=draft.get("action_id"),
                        recipients=recipient_list,
                        metadata=backend_metadata,
                    )
                    conn.commit()
                except Exception:  # pragma: no cover - best effort logging
                    self.logger.exception(
                        "Failed to register dispatch chain for unique_id %s", unique_id
                    )

                workflow_identifier = backend_metadata.get("workflow_id")
                if workflow_identifier and unique_id:
                    try:
                        record_workflow_dispatch(
                            workflow_id=workflow_identifier,
                            unique_id=unique_id,
                            supplier_id=str(
                                backend_metadata.get("supplier_id")
                                or draft.get("supplier_id")
                                or ""
                            )
                            or None,
                            supplier_email=(
                                recipient_list[0]
                                if recipient_list
                                else draft.get("receiver")
                            ),
                            message_id=message_id,
                            subject=subject,
                            dispatched_at=datetime.now(timezone.utc),
                        )
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "Failed to record workflow email dispatch for workflow %s",
                            workflow_identifier,
                        )
                    else:
                        try:
                            BackendScheduler.ensure(
                                self.agent_nick
                            ).notify_email_dispatch(workflow_identifier)
                        except Exception:  # pragma: no cover - defensive logging
                            logger.exception(
                                "Failed to trigger email watcher for workflow %s",
                                workflow_identifier,
                            )
            elif message_id:
                dispatch_payload["message_id"] = message_id

            return {
                "unique_id": unique_id,
                "sent": bool(sent),
                "recipients": recipient_list,
                "sender": sender_email,
                "subject": subject,
                "body": body,
                "message_id": message_id,
                "thread_index": dispatch_payload.get("thread_index"),
                "workflow_id": backend_metadata.get("workflow_id")
                or dispatch_payload.get("workflow_id"),
                "run_id": backend_metadata.get("run_id")
                or dispatch_payload.get("run_id"),
                "payload": provider_payload,
                "draft": dispatch_payload,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_latest_draft(self, conn, identifier: str) -> Optional[Tuple]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, rfq_id, supplier_id, supplier_name, subject, body, sent,
                       review_status, recipient_email, contact_level, thread_index, payload, sender, sent_on,
                       workflow_id, run_id, unique_id, mailbox, dispatch_run_id, dispatched_at
                FROM proc.draft_rfq_emails
                WHERE unique_id = %s
                ORDER BY sent ASC, thread_index DESC, id DESC
                LIMIT 1
                """,
                (identifier,),
            )
            row = cur.fetchone()

        if row:
            return row

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, rfq_id, supplier_id, supplier_name, subject, body, sent,
                       review_status, recipient_email, contact_level, thread_index, payload, sender, sent_on,
                       workflow_id, run_id, unique_id, mailbox, dispatch_run_id, dispatched_at
                FROM proc.draft_rfq_emails
                WHERE rfq_id = %s
                ORDER BY sent ASC, thread_index DESC, id DESC
                LIMIT 1
                """,
                (identifier,),
            )
            return cur.fetchone()

    def _resolve_identifier_for_workflow(self, workflow_id: str) -> Optional[str]:
        workflow_id = self._normalise_identifier(workflow_id)
        if not workflow_id:
            return None

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT unique_id, rfq_id
                        FROM proc.draft_rfq_emails
                        WHERE workflow_id = %s
                          AND review_status IN ('APPROVED', 'SENT')
                        ORDER BY sent ASC, dispatched_at DESC NULLS LAST, updated_on DESC, id DESC
                        LIMIT 1
                        """,
                        (workflow_id,),
                    )
                    row = cur.fetchone()
        except Exception:
            self.logger.exception(
                "Failed to resolve dispatch identifier for workflow %s", workflow_id
            )
            return None

        if not row:
            return None

        unique_id, rfq_id = (row + (None, None))[:2]
        return self._normalise_identifier(unique_id) or self._normalise_identifier(rfq_id)

    def _hydrate_draft(self, row: Tuple) -> Dict[str, Any]:
        values = list(row)
        if len(values) < 20:
            values.extend([None] * (20 - len(values)))

        (
            draft_id,
            rfq_id,
            supplier_id,
            supplier_name,
            subject,
            body,
            sent,
            review_status,
            recipient_email,
            contact_level,
            thread_index,
            payload,
            sender,
            sent_on,
            workflow_id,
            run_id,
            unique_id,
            mailbox,
            dispatch_run_id,
            dispatched_at,
        ) = values[:20]

        hydrated: Dict[str, Any]
        if isinstance(payload, dict):
            hydrated = dict(payload)
        else:
            try:
                hydrated = json.loads(payload) if payload else {}
            except Exception:
                hydrated = {}

        defaults = {
            "id": draft_id,
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "subject": subject,
            "body": body,
            "sent_status": bool(sent),
            "review_status": (review_status or "PENDING"),
            "receiver": recipient_email,
            "contact_level": contact_level,
            "thread_index": thread_index,
            "sender": sender,
            "recipients": hydrated.get("recipients")
            or ([recipient_email] if recipient_email else []),
            "workflow_id": workflow_id,
            "run_id": run_id,
            "unique_id": unique_id,
            "mailbox": mailbox,
            "dispatch_run_id": dispatch_run_id,
            "dispatched_at": dispatched_at,
        }
        for key, value in defaults.items():
            hydrated.setdefault(key, value)
        if sent_on and "sent_on" not in hydrated:
            hydrated["sent_on"] = (
                sent_on
                if isinstance(sent_on, str)
                else getattr(sent_on, "isoformat", lambda: sent_on)()
            )
        recipients_value = hydrated.get("recipients")
        if isinstance(recipients_value, str):
            hydrated["recipients"] = self._normalise_recipients([recipients_value])
        elif isinstance(recipients_value, Iterable):
            hydrated["recipients"] = self._normalise_recipients(recipients_value)
        else:
            hydrated["recipients"] = []
        if not hydrated.get("sender"):
            hydrated["sender"] = getattr(self.settings, "ses_default_sender", "")
        if not hydrated.get("review_status"):
            hydrated["review_status"] = review_status or "PENDING"
        return hydrated

    @staticmethod
    def _extract_workflow_identifier(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        for key in (
            "workflow_id",
            "workflowId",
            "workflowID",
            "process_workflow_id",
        ):
            value = payload.get(key)
            if isinstance(value, str):
                token = value.strip()
                if token:
                    return token
        return None

    def _update_draft_status(
        self,
        conn,
        row: Tuple,
        payload: Dict[str, Any],
        recipients: Sequence[str],
        sent: bool,
    ) -> None:
        values = list(row)
        if len(values) < 20:
            values.extend([None] * (20 - len(values)))

        draft_id = values[0]
        recipient = recipients[0] if recipients else payload.get("receiver")
        try:
            contact_level = int(payload.get("contact_level", 1 if recipients else 0))
        except Exception:
            contact_level = 1 if recipients else 0

        payload["sent_status"] = bool(sent)
        payload_json = json.dumps(payload, default=str)

        supplier_id = payload.get("supplier_id") or values[2]
        supplier_name = payload.get("supplier_name") or values[3]
        rfq_value = payload.get("rfq_id") or values[1]
        workflow_id = payload.get("workflow_id") or values[14]
        run_id = payload.get("run_id") or payload.get("dispatch_run_id") or values[15]
        unique_id = payload.get("unique_id") or values[16] or uuid.uuid4().hex
        mailbox = payload.get("mailbox") or values[17]
        dispatch_run_id = payload.get("dispatch_run_id") or values[18]

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE proc.draft_rfq_emails
                SET sent = %s,
                    subject = %s,
                    body = %s,
                    recipient_email = %s,
                    contact_level = %s,
                    supplier_id = %s,
                    supplier_name = %s,
                    rfq_id = %s,
                    payload = %s,
                    workflow_id = %s,
                    run_id = %s,
                    unique_id = %s,
                    mailbox = %s,
                    dispatch_run_id = %s,
                    review_status = CASE WHEN %s THEN 'SENT' ELSE review_status END,
                    dispatched_at = CASE WHEN %s THEN NOW() ELSE dispatched_at END,
                    sent_on = CASE WHEN %s THEN NOW() ELSE sent_on END,
                    updated_on = NOW()
                WHERE id = %s
                """,
                (
                    bool(sent),
                    payload.get("subject"),
                    payload.get("body"),
                    recipient,
                    contact_level,
                    supplier_id,
                    supplier_name,
                    rfq_value,
                    payload_json,
                    workflow_id,
                    run_id,
                    unique_id,
                    mailbox,
                    dispatch_run_id,
                    bool(sent),
                    bool(sent),
                    bool(sent),
                    draft_id,
                ),
            )

    def _update_action_sent_status(
        self,
        conn,
        payload: Dict[str, Any],
        unique_id: str,
        sent: bool,
    ) -> None:
        action_id = self._extract_action_id(payload)
        if not action_id:
            logger.debug(
                "No action identifier found for dispatch (unique_id=%s); skipping sent_status update",
                unique_id,
            )
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT process_output FROM proc.action WHERE action_id = %s",
                    (action_id,),
                )
                row = cur.fetchone()
            if not row:
                return

            process_output = self._load_json_field(row[0])
            if process_output is None:
                return

            if not self._mark_sent_status(process_output, unique_id, sent):
                return

            updated_payload = json.dumps(process_output, default=str)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE proc.action
                    SET process_output = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE action_id = %s
                    """,
                    (updated_payload, action_id),
                )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to update sent_status for action %s (unique_id=%s)",
                action_id,
                unique_id,
            )

    # ------------------------------------------------------------------
    # Outbound thread mapping helpers
    # ------------------------------------------------------------------
    def _record_thread_mapping(
        self,
        conn,
        message_id: Optional[str],
        unique_id: str,
        supplier_id: Optional[str],
        recipients: Sequence[str],
        *,
        rfq_id: Optional[str] = None,
    ) -> None:
        if not message_id or conn is None or not self._thread_table_name:
            return

        self._ensure_thread_table(conn)

        rfq_reference = self._normalise_identifier(
            rfq_id
        ) or self._normalise_identifier(unique_id)
        if not rfq_reference:
            return

        record_thread_mapping(
            conn,
            self._thread_table_name,
            message_id=str(message_id),
            rfq_id=rfq_reference,
            supplier_id=str(supplier_id) if supplier_id else None,
            recipients=recipients,
            logger=self.logger,
        )

    def _ensure_thread_table(self, conn) -> None:
        if self._thread_table_ready:
            return

        ensure_thread_table(conn, self._thread_table_name, logger=self.logger)
        self._thread_table_ready = True

    @staticmethod
    def _extract_action_id(payload: Dict[str, Any]) -> Optional[str]:
        """Return the best effort action identifier from ``payload``."""

        for key in ("action_id", "draft_action_id", "email_action_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _normalise_recipients(self, recipients: Optional[Iterable[str]]) -> List[str]:
        if recipients is None:
            return []
        values: List[str] = []
        seen: set[str] = set()
        for value in recipients:
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            values.append(candidate)
        return values

    @staticmethod
    def _normalise_identifier(value: Optional[Any]) -> Optional[str]:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
        return None

    def _ensure_tracking_annotation(
        self,
        body: str,
        *,
        unique_id: str,
        supplier_id: Optional[str] = None,
        dispatch_token: Optional[str] = None,
        run_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        text = body or ""
        existing_metadata, cleaned_body = strip_tracking_comment(text)

        resolved_workflow = workflow_id or (
            existing_metadata.workflow_id if existing_metadata else None
        )
        if not resolved_workflow:
            raise ValueError(
                "workflow_id is required to annotate outbound supplier emails"
            )

        resolved_supplier = supplier_id or (
            existing_metadata.supplier_id if existing_metadata else None
        )
        resolved_token = dispatch_token or (
            existing_metadata.token if existing_metadata else None
        )
        resolved_run = run_id or (
            existing_metadata.run_id if existing_metadata else None
        )

        resolved_unique = unique_id or (
            existing_metadata.unique_id if existing_metadata else None
        )
        if not resolved_unique:
            raise ValueError(
                "unique_id is required to annotate outbound supplier emails"
            )

        comment, tracking_meta = build_tracking_comment(
            workflow_id=resolved_workflow,
            unique_id=resolved_unique,
            supplier_id=resolved_supplier,
            token=resolved_token,
            run_id=resolved_run,
        )

        base_body = (cleaned_body or text).strip()
        annotated_body = ensure_tracking_prefix(base_body, comment)
        annotated_body = embed_unique_id_in_email_body(
            annotated_body,
            tracking_meta.unique_id,
        )

        metadata: Dict[str, Any] = {
            "workflow_id": tracking_meta.workflow_id,
            "unique_id": tracking_meta.unique_id,
        }
        if tracking_meta.supplier_id:
            metadata["supplier_id"] = tracking_meta.supplier_id
        if tracking_meta.token:
            metadata["dispatch_token"] = tracking_meta.token
        if tracking_meta.run_id:
            metadata["run_id"] = tracking_meta.run_id

        return annotated_body, metadata

    @staticmethod
    def _load_json_field(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return None

    @staticmethod
    def _mark_sent_status(data: Any, unique_id: str, sent: bool) -> bool:
        """Update ``sent_status`` in ``data`` for drafts matching ``unique_id``."""

        updated = False
        desired_status = "True" if sent else "False"

        if isinstance(data, dict):
            updated |= EmailDispatchService._update_draft_collection(
                data.get("drafts"), unique_id, sent
            )
            if data.get("unique_id") == unique_id or data.get("rfq_id") == unique_id:
                current = EmailDispatchService._normalise_sent_status(
                    data.get("sent_status")
                )
                if current != desired_status:
                    data["sent_status"] = desired_status
                    updated = True
                if sent and "sent_on" not in data:
                    data["sent_on"] = datetime.utcnow().isoformat()
        elif isinstance(data, list):
            for item in data:
                updated |= EmailDispatchService._mark_sent_status(item, unique_id, sent)

        return updated

    @staticmethod
    def _update_draft_collection(value: Any, unique_id: str, sent: bool) -> bool:
        if not isinstance(value, list):
            return False
        updated = False
        desired_status = "True" if sent else "False"
        for draft in value:
            if not isinstance(draft, dict):
                continue
            if draft.get("unique_id") == unique_id or draft.get("rfq_id") == unique_id:
                current = EmailDispatchService._normalise_sent_status(
                    draft.get("sent_status")
                )
                if current != desired_status:
                    draft["sent_status"] = desired_status
                    updated = True
                if sent and "sent_on" not in draft:
                    draft["sent_on"] = datetime.utcnow().isoformat()
        return updated

    @staticmethod
    def _normalise_sent_status(value: Any) -> Optional[str]:
        """Coerce ``value`` to a canonical string representation."""

        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, str):
            stripped = value.strip()
            lower = stripped.lower()
            if lower == "true":
                return "True"
            if lower == "false":
                return "False"
        return None


def find_recent_sent_for_supplier(
    connection,
    message: Dict[str, object],
    *,
    mailbox: Optional[str] = None,
    window_minutes: int = 10,
) -> Optional[Dict[str, object]]:
    """Return the most recent dispatch-chain entry for ``message``'s supplier."""

    supplier_value = message.get("supplier_id")
    if supplier_value in (None, ""):
        return None

    try:
        supplier_norm = str(supplier_value).strip().lower()
    except Exception:
        return None

    if not supplier_norm:
        return None

    try:
        interval = max(int(window_minutes), 1)
    except Exception:
        interval = 10

    mailbox_filter = mailbox or None

    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT rfq_id, supplier_id, dispatch_metadata, message_id, thread_index, created_at
            FROM proc.email_dispatch_chains
            WHERE awaiting_response = TRUE
              AND created_at >= NOW() - (%s || ' minutes')::INTERVAL
              AND LOWER(supplier_id) = %s
              AND (%s IS NULL
                   OR dispatch_metadata ->> 'mailbox' IS NULL
                   OR dispatch_metadata ->> 'mailbox' = %s)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(interval), supplier_norm, mailbox_filter, mailbox_filter),
        )
        row = cur.fetchone()

    if not row:
        return None

    metadata = row[2]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = None

    return {
        "rfq_id": row[0],
        "supplier_id": row[1],
        "dispatch_metadata": metadata,
        "message_id": row[3],
        "thread_index": row[4],
        "created_at": row[5],
    }
