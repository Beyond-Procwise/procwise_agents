"""Email dispatch service for sending stored RFQ drafts."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.gpu import configure_gpu

from .email_dispatch_chain_store import register_dispatch as register_dispatch_chain
from .email_service import EmailService
from .email_thread_store import (
    DEFAULT_THREAD_TABLE,
    ensure_thread_table,
    record_thread_mapping,
    sanitise_thread_table_name,
)

configure_gpu()

logger = logging.getLogger(__name__)

_RFQ_ID_PATTERN = re.compile(r"RFQ-ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)


_DEFAULT_THREAD_TABLE = DEFAULT_THREAD_TABLE


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
    def send_draft(
        self,
        rfq_id: str,
        recipients: Optional[Iterable[str]] = None,
        sender: Optional[str] = None,
        subject_override: Optional[str] = None,
        body_override: Optional[str] = None,
        attachments: Optional[List[Tuple[bytes, str]]] = None,
    ) -> Dict[str, Any]:
        """Send the latest draft for ``rfq_id`` and mark it as sent."""

        rfq_id = (rfq_id or "").strip()
        if not rfq_id:
            raise ValueError("rfq_id is required to send an email draft")

        with self.agent_nick.get_db_connection() as conn:
            draft_row = self._fetch_latest_draft(conn, rfq_id)
            if draft_row is None:
                raise ValueError(f"No stored draft found for RFQ {rfq_id}")

            draft = self._hydrate_draft(draft_row)

            recipient_list = self._normalise_recipients(
                recipients if recipients is not None else draft.get("recipients")
            )
            if not recipient_list and draft.get("receiver"):
                recipient_list = self._normalise_recipients([draft["receiver"]])

            if not recipient_list:
                raise ValueError("At least one recipient email is required to send the draft")

            sender_candidate = (
                sender
                if sender is not None
                else draft.get("sender")
                or getattr(self.settings, "ses_default_sender", "")
            )
            sender_email = str(sender_candidate).strip()
            if not sender_email:
                raise ValueError("Sender email address is required")

            if subject_override is not None:
                subject_candidate = subject_override
            else:
                subject_candidate = draft.get("subject")
            subject_str = str(subject_candidate).strip() if subject_candidate else ""
            subject = subject_str or f"{rfq_id} – Request for Quotation"

            body_source = body_override if body_override is not None else draft.get("body")
            body_text = str(body_source).strip() if body_source else ""
            body, backend_metadata = self._ensure_rfq_annotation(body_text, rfq_id)

            dispatch_payload = dict(draft)
            dispatch_payload.update(
                {
                    "subject": subject,
                    "body": body,
                    "recipients": recipient_list,
                    "receiver": recipient_list[0] if recipient_list else draft.get("receiver"),
                    "contact_level": 1 if recipient_list else 0,
                    "sender": sender_email,
                    "dispatch_metadata": backend_metadata,
                }
            )

            headers = {"X-Procwise-RFQ-ID": rfq_id}
            mailbox_header = draft.get("mailbox") or getattr(self.settings, "supplier_mailbox", None)
            if mailbox_header:
                headers["X-Procwise-Mailbox"] = mailbox_header

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
                rfq_id,
                bool(sent),
            )

            conn.commit()

            dispatch_payload["sent_status"] = bool(sent)
            if sent:
                dispatch_payload["sent_on"] = datetime.utcnow().isoformat()
                dispatch_payload["message_id"] = message_id
                try:
                    self._record_thread_mapping(
                        conn,
                        message_id,
                        rfq_id,
                        draft.get("supplier_id"),
                        recipient_list,
                    )
                    conn.commit()
                except Exception:  # pragma: no cover - defensive
                    self.logger.exception(
                        "Failed to persist thread mapping for RFQ %s", rfq_id
                    )
                try:
                    register_dispatch_chain(
                        conn,
                        rfq_id=rfq_id,
                        message_id=message_id,
                        subject=subject,
                        body=body,
                        thread_index=draft.get("thread_index"),
                        supplier_id=draft.get("supplier_id"),
                        workflow_ref=draft.get("action_id"),
                        recipients=recipient_list,
                        metadata=backend_metadata,
                    )
                except Exception:  # pragma: no cover - best effort logging
                    self.logger.exception(
                        "Failed to register dispatch chain for RFQ %s", rfq_id
                    )
            elif message_id:
                dispatch_payload["message_id"] = message_id

            return {
                "rfq_id": rfq_id,
                "sent": bool(sent),
                "recipients": recipient_list,
                "sender": sender_email,
                "subject": subject,
                "body": body,
                "message_id": message_id,
                "thread_index": dispatch_payload.get("thread_index"),
                "draft": dispatch_payload,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_latest_draft(self, conn, rfq_id: str) -> Optional[Tuple]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, rfq_id, supplier_id, supplier_name, subject, body, sent,
                       recipient_email, contact_level, thread_index, payload, sender, sent_on
                FROM proc.draft_rfq_emails
                WHERE rfq_id = %s
                ORDER BY sent ASC, thread_index DESC, id DESC
                LIMIT 1
                """,
                (rfq_id,),
            )
            return cur.fetchone()

    def _hydrate_draft(self, row: Tuple) -> Dict[str, Any]:
        (
            draft_id,
            rfq_id,
            supplier_id,
            supplier_name,
            subject,
            body,
            sent,
            recipient_email,
            contact_level,
            thread_index,
            payload,
            sender,
            sent_on,
        ) = row

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
            "receiver": recipient_email,
            "contact_level": contact_level,
            "thread_index": thread_index,
            "sender": sender,
            "recipients": hydrated.get("recipients") or ([recipient_email] if recipient_email else []),
        }
        for key, value in defaults.items():
            hydrated.setdefault(key, value)
        if sent_on and "sent_on" not in hydrated:
            hydrated["sent_on"] = sent_on if isinstance(sent_on, str) else getattr(sent_on, "isoformat", lambda: sent_on)()
        recipients_value = hydrated.get("recipients")
        if isinstance(recipients_value, str):
            hydrated["recipients"] = self._normalise_recipients([recipients_value])
        elif isinstance(recipients_value, Iterable):
            hydrated["recipients"] = self._normalise_recipients(recipients_value)
        else:
            hydrated["recipients"] = []
        if not hydrated.get("sender"):
            hydrated["sender"] = getattr(self.settings, "ses_default_sender", "")
        return hydrated

    def _update_draft_status(
        self,
        conn,
        row: Tuple,
        payload: Dict[str, Any],
        recipients: Sequence[str],
        sent: bool,
    ) -> None:
        draft_id = row[0]
        recipient = recipients[0] if recipients else payload.get("receiver")
        try:
            contact_level = int(payload.get("contact_level", 1 if recipients else 0))
        except Exception:
            contact_level = 1 if recipients else 0

        payload["sent_status"] = bool(sent)
        payload_json = json.dumps(payload, default=str)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE proc.draft_rfq_emails
                SET sent = %s,
                    subject = %s,
                    body = %s,
                    recipient_email = %s,
                    contact_level = %s,
                    payload = %s,
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
                    payload_json,
                    bool(sent),
                    draft_id,
                ),
            )

    def _update_action_sent_status(
        self,
        conn,
        payload: Dict[str, Any],
        rfq_id: str,
        sent: bool,
    ) -> None:
        action_id = self._extract_action_id(payload)
        if not action_id:
            logger.debug(
                "No action identifier found in dispatch payload for RFQ %s; skipping sent_status update",
                rfq_id,
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

            if not self._mark_sent_status(process_output, rfq_id, sent):
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
                "Failed to update sent_status for action %s and RFQ %s",
                action_id,
                rfq_id,
            )

    # ------------------------------------------------------------------
    # Outbound thread mapping helpers
    # ------------------------------------------------------------------
    def _record_thread_mapping(
        self,
        conn,
        message_id: Optional[str],
        rfq_id: str,
        supplier_id: Optional[str],
        recipients: Sequence[str],
    ) -> None:
        if not message_id or conn is None or not self._thread_table_name:
            return

        self._ensure_thread_table(conn)

        record_thread_mapping(
            conn,
            self._thread_table_name,
            message_id=str(message_id),
            rfq_id=str(rfq_id),
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
        for value in recipients:
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if not candidate:
                continue
            if candidate.lower() not in {item.lower() for item in values}:
                values.append(candidate)
        return values

    def _ensure_rfq_annotation(self, body: str, rfq_id: str) -> tuple[str, Dict[str, Any]]:
        text = body or ""
        metadata: Dict[str, Any] = {"rfq_id": rfq_id}
        if text and rfq_id:
            text = re.sub(r"(?i)\bRFQ[-\s:]*[A-Z0-9]{2,}[^\s]*", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip(), metadata

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
    def _mark_sent_status(data: Any, rfq_id: str, sent: bool) -> bool:
        """Update ``sent_status`` in ``data`` for drafts matching ``rfq_id``."""

        updated = False
        desired_status = "True" if sent else "False"

        if isinstance(data, dict):
            updated |= EmailDispatchService._update_draft_collection(
                data.get("drafts"), rfq_id, sent
            )
            if data.get("rfq_id") == rfq_id:
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
                updated |= EmailDispatchService._mark_sent_status(item, rfq_id, sent)

        return updated

    @staticmethod
    def _update_draft_collection(value: Any, rfq_id: str, sent: bool) -> bool:
        if not isinstance(value, list):
            return False
        updated = False
        desired_status = "True" if sent else "False"
        for draft in value:
            if isinstance(draft, dict) and draft.get("rfq_id") == rfq_id:
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
