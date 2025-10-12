from __future__ import annotations
import email
import imaplib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import Message
from typing import Any, Dict, List, Optional, Set, Tuple

from repositories.draft_rfq_emails_repo import (
    init_schema as init_draft_schema,
    expected_unique_ids_and_last_dispatch,
)
from repositories.supplier_responses_repo import (
    init_schema as init_resp_schema,
    upsert_response,
    fetch_latest_for_workflow,
    mark_processed_by_ids,
)

logger = logging.getLogger(__name__)

# Hidden marker + headers (unique id primary matcher; supplier id secondary)
MARKER_RE = re.compile(r"<!--\s*PROCWISE:(\{.*?\})\s*-->", re.DOTALL | re.IGNORECASE)
HDR_WORKFLOW = "X-Procwise-Workflow-Id"
HDR_UNIQUE = "X-Procwise-Unique-Id"
HDR_SUPPLIER = "X-Procwise-Supplier-Id"

@dataclass
class InboundEmail:
    uid: str
    subject: str
    from_addr: str
    to_addrs: List[str]
    date_utc: datetime
    message_id: Optional[str]
    workflow_id: Optional[str]
    unique_id: Optional[str]
    supplier_id: Optional[str]
    body_text: str
    raw_rfc822: bytes

# -------------------- IMAP helpers --------------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default

def _connect_imap() -> imaplib.IMAP4:
    host = _env("IMAP_HOST", "imap.example.com")
    port = int(_env("IMAP_PORT", "993"))
    use_ssl = _env("IMAP_SSL", "true").lower() in ("1", "true", "yes", "y", "on")
    user = _env("IMAP_USERNAME")
    pwd = _env("IMAP_PASSWORD")
    if use_ssl:
        imap = imaplib.IMAP4_SSL(host, port)
    else:
        imap = imaplib.IMAP4(host, port)
    if not user or not pwd:
        raise RuntimeError("IMAP_USERNAME/IMAP_PASSWORD must be set")
    typ, _ = imap.login(user, pwd)
    if typ != "OK":
        raise RuntimeError("IMAP login failed")
    folder = _env("IMAP_FOLDER", "INBOX")
    typ, _ = imap.select(folder, readonly=True)
    if typ != "OK":
        raise RuntimeError(f"Unable to select IMAP folder: {folder}")
    return imap

def _decode_header_val(val: Optional[str]) -> str:
    if not val:
        return ""
    try:
        return str(make_header(decode_header(val)))
    except Exception:
        return val

def _get_text_from_message(msg: Message) -> str:
    if msg.is_multipart():
        # prefer text/plain
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    pass
        # fallback html
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/html" and "attachment" not in disp:
                try:
                    html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", html)
                except Exception:
                    pass
        return ""
    else:
        try:
            return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            return ""

def _extract_marker(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = MARKER_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}

def _parse_email(raw: bytes) -> InboundEmail:
    msg: Message = email.message_from_bytes(raw)
    subj = _decode_header_val(msg.get("Subject"))
    frm = _decode_header_val(msg.get("From"))
    tos = _decode_header_val(msg.get("To"))
    to_addrs = [t.strip() for t in re.split(r"[;,]", tos) if t.strip()] if tos else []
    mid = (msg.get("Message-ID") or msg.get("Message-Id") or "").strip("<> \t\r\n")
    # headers first
    wf_hdr = (msg.get(HDR_WORKFLOW) or "").strip()
    uid_hdr = (msg.get(HDR_UNIQUE) or "").strip()
    sid_hdr = (msg.get(HDR_SUPPLIER) or "").strip()
    # body marker as fallback/confirmation
    text = _get_text_from_message(msg)
    marker = _extract_marker(text)
    wf = wf_hdr or marker.get("workflow_id")
    uniq = uid_hdr or marker.get("unique_id")
    supplier = sid_hdr or marker.get("supplier_id")
    # date normalize
    try:
        d = msg.get("Date")
        dt = email.utils.parsedate_to_datetime(d) if d else None
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)

    return InboundEmail(
        uid="",
        subject=subj or "",
        from_addr=frm or "",
        to_addrs=to_addrs,
        date_utc=dt or datetime.now(timezone.utc),
        message_id=mid or None,
        workflow_id=wf or None,
        unique_id=uniq or None,
        supplier_id=supplier or None,
        body_text=text or "",
        raw_rfc822=raw,
    )

def _imap_search_since(imap: imaplib.IMAP4, since_utc: datetime) -> List[str]:
    since_str = since_utc.strftime("%d-%b-%Y")
    typ, data = imap.uid("SEARCH", None, f'(SINCE {since_str})')
    if typ != "OK":
        return []
    ids = (data[0] or b"").decode("utf-8").strip()
    return [i for i in ids.split() if i]

# -------------------- Core flow --------------------

def _start_processing_worker(
    *,
    row: Dict[str, Any],
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
) -> Dict[str, Any]:
    payload = {
        "message": row.get("body_text", ""),
        "email_headers": {
            "message_id": row.get("message_id"),
            "subject": row.get("subject"),
            "from": row.get("from_addr"),
            "to": (row.get("to_addrs") or "").split(","),
            "workflow_id": row.get("workflow_id"),
            "unique_id": row.get("unique_id"),
            "supplier_id": row.get("supplier_id"),
        },
        "workflow_id": row.get("workflow_id"),
        "run_id": row.get("run_id"),
        "action_id": "supplier_interaction",
    }
    # Orchestrator preferred; supplier_interaction agent fallback
    if orchestrator and hasattr(orchestrator, "execute_workflow"):
        try:
            return orchestrator.execute_workflow("supplier_interaction", payload)
        except Exception as exc:
            logger.exception("Orchestrator supplier_interaction failed")
            return {"status": "failed", "error": str(exc)}
    if agent_registry and "supplier_interaction" in agent_registry:
        agent = agent_registry["supplier_interaction"]
        try:
            if hasattr(agent, "create_context"):
                ctx = agent.create_context(input_data=payload)
                out = agent.execute(ctx)
                return {"status": "completed", "result": getattr(out, "data", out)}
            if hasattr(agent, "run"):
                return {"status": "completed", "result": agent.run(payload)}
            return {"status": "completed", "result": agent(payload)}  # type: ignore
        except Exception as exc:
            logger.exception("Direct supplier_interaction failed")
            return {"status": "failed", "error": str(exc)}
    return {"status": "failed", "error": "No orchestrator or supplier_interaction agent available"}

def run_email_watcher_for_workflow(
    *,
    workflow_id: str,
    run_id: Optional[str],
    wait_seconds_after_last_dispatch: int = 90,
    lookback_minutes: int = 240,
    mailbox_name: Optional[str] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Implements the new rules:
      - No RFQ-id matching: primary=unique_id (header/body), secondary=supplier_id.
      - Counter is authoritative: use draft_rfq_emails to determine expected unique_ids and last dispatch time.
      - Trigger AFTER all dispatches; wait 90s; scan IMAP; proceed only when counts match.
      - Persist to proc.supplier_responses; verify ALL expected unique_ids are matched; process ALL in parallel.
      - Works for both initial drafting and negotiation replies.
    """
    # Ensure schemas exist (safe in prod, creates in dev)
    init_draft_schema()
    init_resp_schema()

    # 1) Fetch expected unique_ids and last dispatch time from draft table
    expected_uids, map_uid_to_supplier, last_dt = expected_unique_ids_and_last_dispatch(workflow_id=workflow_id, run_id=run_id)
    expected_count = len(expected_uids)
    if expected_count == 0:
        msg = f"No dispatched emails found for workflow={workflow_id} run={run_id}"
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "expected": 0, "found": 0, "processed": 0}

    # 2) Wait 90 seconds after last dispatch
    if last_dt is not None:
        target = last_dt + timedelta(seconds=max(0, int(wait_seconds_after_last_dispatch)))
        now = datetime.now(timezone.utc)
        if target > now:
            wait_for = (target - now).total_seconds()
            logger.info("Waiting %.1f seconds after last dispatch (workflow=%s)", wait_for, workflow_id)
            time.sleep(wait_for)

    # 3) Scan IMAP since (last_dt - small buffer) or lookback window
    since = (last_dt - timedelta(minutes=5)) if last_dt else (datetime.now(timezone.utc) - timedelta(minutes=max(5, lookback_minutes)))
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    imap = _connect_imap()
    stored = 0
    matched_unique_ids: Set[str] = set()
    fallback_supplier_hits: Set[str] = set()  # supplier_ids that matched without unique_id

    try:
        uids = _imap_search_since(imap, since)
        for uid in uids:
            typ, payload = imap.uid("FETCH", uid, "(RFC822)")
            if typ != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            raw = payload[0][1]
            mail = _parse_email(raw)
            mail.uid = uid

            # Only consider emails declaring this workflow in header/body marker
            if (mail.workflow_id or "").strip() != workflow_id:
                continue

            # Count match: we collect candidates; primary match on unique_id, else supplier fallback
            candidate_unique_id: Optional[str] = None
            candidate_supplier_id: Optional[str] = None

            if mail.unique_id and mail.unique_id in expected_uids:
                candidate_unique_id = mail.unique_id
            else:
                # fallback: supplier-based only if that supplier_id is uniquely associated to exactly one expected unique_id
                sid = (mail.supplier_id or "").strip()
                if sid:
                    # reverse mapping: find any expected unique_id bound to this supplier
                    # (in normal dispatch, 1 unique_id per supplier under a workflow)
                    for uid_expected, sid_expected in map_uid_to_supplier.items():
                        if sid_expected == sid and uid_expected in expected_uids:
                            candidate_unique_id = uid_expected
                            candidate_supplier_id = sid
                            break

            # Persist every candidate response row (even if we didn't resolve unique_id yet, store supplier_id/message)
            headers = {
                "Message-ID": mail.message_id,
                HDR_WORKFLOW: mail.workflow_id,
                HDR_UNIQUE: mail.unique_id,
                HDR_SUPPLIER: mail.supplier_id,
                "Mailbox": mailbox_name or _env("IMAP_USERNAME"),
                "IMAP_UID": uid,
            }
            upsert_response(
                workflow_id=workflow_id,
                run_id=str(run_id) if run_id is not None else None,
                unique_id=candidate_unique_id or mail.unique_id,  # prefer resolved
                supplier_id=candidate_supplier_id or mail.supplier_id,
                message_id=mail.message_id,
                mailbox=mailbox_name or _env("IMAP_USERNAME"),
                imap_uid=uid,
                from_addr=mail.from_addr,
                to_addrs=mail.to_addrs,
                subject=mail.subject,
                body_text=mail.body_text,
                headers=headers,
                received_at=mail.date_utc,
                extra={}
            )
            stored += 1

            if candidate_unique_id:
                matched_unique_ids.add(candidate_unique_id)
                if candidate_supplier_id:
                    fallback_supplier_hits.add(candidate_supplier_id)

        # 4) Gate: counts must match (number of distinct matched unique_ids == expected_count)
        found_count = len(matched_unique_ids)
        logger.info("Watcher reconciliation workflow=%s expected=%d found=%d", workflow_id, expected_count, found_count)
        if found_count != expected_count:
            missing = sorted(list(expected_uids - matched_unique_ids))
            reason = f"Not all responses received yet. Missing unique_ids={','.join(missing) if missing else 'none'}"
            return {
                "status": "not_ready",
                "reason": reason,
                "expected": expected_count,
                "found": found_count,
                "stored": stored,
                "missing_unique_ids": missing,
            }

    finally:
        try:
            imap.close()
        except Exception:
            pass
        try:
            imap.logout()
        except Exception:
            pass

    # 5) Fetch all pending rows for this workflow and process ALL (parallel)
    rows = fetch_latest_for_workflow(workflow_id=workflow_id)
    # Keep only rows whose unique_id is in the expected set (safety)
    rows = [r for r in rows if (r.get("unique_id") in expected_uids)]
    results: List[Dict[str, Any]] = []

    if rows:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = [
                pool.submit(
                    _start_processing_worker,
                    row=row,
                    agent_registry=agent_registry,
                    orchestrator=orchestrator,
                )
                for row in rows
            ]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.exception("Processing worker failed")
                    results.append({"status": "failed", "error": str(exc)})

        # Mark processed
        mark_processed_by_ids([int(r["id"]) for r in rows if "id" in r])

    return {
        "status": "processed",
        "workflow_id": workflow_id,
        "expected": expected_count,
        "found": len(matched_unique_ids),
        "stored": stored,
        "processed": len(results),
        "results": results,
    }
