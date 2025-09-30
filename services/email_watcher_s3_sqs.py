from __future__ import annotations
import os, json, re, base64, quopri, logging, unicodedata
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import List, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError

# ---------- Config ----------
REGION = os.getenv("AWS_REGION", "ap-south-1")
QUEUE_URL = os.environ["SQS_QUEUE_URL"]
PROCESSED_TABLE = os.getenv("PROCESSED_TABLE", "procwise_email_processed")
DELETE_ON_SUCCESS = os.getenv("DELETE_ON_SUCCESS", "true").lower() == "true"
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "15"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Compile RFQ patterns
_default_patterns = [r"[A-Z]{3}\d{5,}", r"RFQ-\d{4}-\d{3,}[A-Z]?"]
PATTERNS_STR = os.getenv("RFQ_ID_PATTERNS")
RFQ_PATTERNS = json.loads(PATTERNS_STR) if PATTERNS_STR else _default_patterns
_DASH_CLASS = r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"
_ZERO_WIDTH = "".join(["\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060", "\u200E", "\u200F"])

RFQ_ID_RE = re.compile(r"\\b(?:" + "|".join(RFQ_PATTERNS) + r")\\b", re.IGNORECASE)

# Subject noise (Re:, Fwd:, [EXTERNAL], etc.)
NOISE = re.compile(r"^(?:(re|fwd|fw)\\s*:\\s*)+|\\[[^\\]]+\\]\\s*", re.IGNORECASE)

# ---------- AWS ----------
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)

# ---------- Logging ----------
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
log = logging.getLogger("email_watcher")

# ---------- Helpers ----------
def clean_subject(s: Optional[str]) -> str:
    s = (s or "").strip()
    while True:
        s2 = NOISE.sub("", s).strip()
        if s2 == s:
            return s
        s = s2


def normalize_text(value: Optional[str]) -> str:
    s = unicodedata.normalize("NFKC", value or "")
    s = re.sub(f"[{re.escape(_ZERO_WIDTH)}]", "", s)
    s = re.sub(_DASH_CLASS, "-", s)
    return re.sub(r"\s+", " ", s).strip()


def canonicalize_rfq_id(token: str) -> str:
    return normalize_text(token).upper()


def extract_rfq_ids_with_positions(text: str) -> List[Tuple[str, int]]:
    seen: Set[str] = set()
    ordered: List[Tuple[str, int]] = []
    for match in RFQ_ID_RE.finditer(text or ""):
        canon = canonicalize_rfq_id(match.group(0))
        if canon in seen:
            continue
        seen.add(canon)
        ordered.append((canon, match.start()))
    return ordered

def extract_text_from_message(msg) -> Tuple[str, str]:
    """
    Returns (text_plain, text_html_as_text_fallback)
    """
    text_plain = None
    html = None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and text_plain is None:
                try:
                    text_plain = part.get_content()
                except:  # charset quirks
                    text_plain = _decode_bytes(part)
            elif ctype == "text/html" and html is None:
                try:
                    html = part.get_content()
                except:
                    html = _decode_bytes(part)
    else:
        ctype = msg.get_content_type()
        try:
            if ctype == "text/plain":
                text_plain = msg.get_content()
            elif ctype == "text/html":
                html = msg.get_content()
        except:
            body = _decode_bytes(msg)
            if ctype == "text/plain": text_plain = body
            elif ctype == "text/html": html = body

    if (not text_plain) and html:
        # basic HTML -> text fallback (no external deps)
        text_plain = _html_to_text(html)
    return (text_plain or ""), (html or "")

def _decode_bytes(part) -> str:
    payload = part.get_payload(decode=True) or b""
    # Try quoted-printable and base64 fallbacks
    try:
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        try:
            return quopri.decodestring(payload).decode("utf-8", "replace")
        except Exception:
            try:
                return base64.b64decode(payload).decode("utf-8", "replace")
            except Exception:
                return payload.decode("utf-8", "replace")

def _html_to_text(html: str) -> str:
    # very minimal strip; preserve RFQ tokens
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    txt = re.sub(r"(?is)<br\\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</p>", "\n", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    return re.sub(r"\\s+", " ", txt).strip()

def extract_rfq_ids(text: str) -> Set[str]:
    return {cid for cid, _ in extract_rfq_ids_with_positions(text)}

def body_tag_extract(body_text: str, body_html: str) -> Optional[str]:
    # Look for hidden tag <!-- PROCWISE:RFQ_ID=... -->
    m = re.search(r"PROCWISE:RFQ_ID=([A-Za-z0-9\-]+)", body_html or "", re.IGNORECASE)
    if m:
        return canonicalize_rfq_id(m.group(1))
    m = re.search(r"PROCWISE:RFQ_ID=([A-Za-z0-9\-]+)", body_text or "", re.IGNORECASE)
    if m:
        return canonicalize_rfq_id(m.group(1))
    return None

def header_extract(msg) -> Optional[str]:
    val = msg.get("X-Procwise-RFQ-Id")
    if val:
        return canonicalize_rfq_id(val)
    return None

def lookup_parent_thread_rfq(msg) -> Optional[str]:
    # TODO: integrate with thread storage to map In-Reply-To / References headers to RFQ IDs
    return None


def pick_rfq_id_from_subject(
    subject_norm: str,
    known_ids_canon: Optional[Set[str]],
    parent_thread_rfq: Optional[str],
) -> Tuple[Optional[str], str, List[str]]:
    candidates = extract_rfq_ids_with_positions(subject_norm)
    if not candidates:
        return None, "none", []

    if len(candidates) == 1:
        return candidates[0][0], "subject_id", []

    ordered = sorted(candidates, key=lambda item: item[1])
    others_all = [cid for cid, _ in ordered]

    if known_ids_canon:
        narrowed = [(cid, pos) for cid, pos in ordered if cid in known_ids_canon]
        if len(narrowed) == 1:
            chosen = narrowed[0][0]
            others = [cid for cid in others_all if cid != chosen]
            return chosen, "subject_id_known", others
        if len(narrowed) > 1:
            if parent_thread_rfq:
                parent_canon = canonicalize_rfq_id(parent_thread_rfq)
                for cid, _ in narrowed:
                    if cid == parent_canon:
                        others = [candidate for candidate in others_all if candidate != cid]
                        return cid, "subject_id_thread", others
            narrowed = sorted(narrowed, key=lambda item: item[1])
            chosen = narrowed[0][0]
            others = [cid for cid in others_all if cid != chosen]
            return chosen, "subject_id_leftmost", others

    chosen = ordered[0][0]
    others = [cid for cid in others_all if cid != chosen]
    return chosen, "subject_id_leftmost", others


def audit_multi_id(subject_norm: str, chosen: str, others: List[str], via: str, s3_key: Optional[str]):
    if not others:
        return
    log.info(
        "Multi-RFQ subject handled",
        extra={
            "s3_key": s3_key or "",
            "chosen_rfq": chosen,
            "other_rfq_candidates": others,
            "matched_via": via,
            "subject_norm": subject_norm[:300],
        },
    )


def match_email_to_rfq(
    known_ids: Optional[Set[str]],
    subject_norm: str,
    body_text_norm: str,
    body_html_norm: str,
    msg,
    *,
    s3_key: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Returns (rfq_id, matched_via)
    matched_via ∈ {'header','body','subject_id','subject_id_known','subject_id_thread','subject_id_leftmost','body_id','none'}
    """
    known_ids = known_ids or set()
    has_known = bool(known_ids)

    # 1) Header
    hdr = header_extract(msg)
    if hdr and (not has_known or hdr in known_ids):
        return hdr, "header"

    # 2) Body tag
    tag = body_tag_extract(body_text_norm, body_html_norm)
    if tag and (not has_known or tag in known_ids):
        return tag, "body"

    parent_thread_rfq = lookup_parent_thread_rfq(msg)
    subject_choice, via, others = pick_rfq_id_from_subject(
        subject_norm, known_ids if has_known else None, parent_thread_rfq
    )
    if subject_choice:
        audit_multi_id(subject_norm, subject_choice, others, via, s3_key)
        return subject_choice, via

    # 4) Body fallback by visible text when no subject match
    body_candidates = extract_rfq_ids_with_positions(body_text_norm)
    if body_candidates:
        return body_candidates[0][0], "body_id"

    return None, "none"

def already_processed(s3_key: str) -> bool:
    try:
        resp = ddb.get_item(
            TableName=PROCESSED_TABLE,
            Key={"s3_key": {"S": s3_key}},
            ConsistentRead=True,
        )
        return "Item" in resp
    except ClientError as e:
        log.error(f"DDB get_item failed: {e}")
        return False

def mark_processed(s3_key: str, message_id: str, provider_id: str, rfq_id: str):
    try:
        ddb.put_item(
            TableName=PROCESSED_TABLE,
            Item={
                "s3_key": {"S": s3_key},
                "message_id": {"S": message_id or ""},
                "provider_id": {"S": provider_id or ""},
                "rfq_id": {"S": rfq_id or ""},
                "processed_at": {"S": datetime.now(timezone.utc).isoformat()},
            },
            ConditionExpression="attribute_not_exists(s3_key)",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise

def dead_letter(reason: str, details: dict):
    log.warning(f"DEAD-LETTER reason={reason} details={json.dumps(details)[:1000]}")

def load_known_rfq_ids() -> Set[str]:
    # TODO: Replace with your data source (DB query, API call). For now, assume all tokens are acceptable.
    # If you want strict matching: fetch active/open RFQ IDs here.
    return set()

# ---------- Core processing ----------
def process_s3_record(bucket: str, key: str) -> bool:
    """
    Returns True if processed successfully (or safely skipped), False if should retry.
    """
    if already_processed(key):
        log.info(f"Skip already-processed {key}")
        return True

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        log.error(f"Cannot fetch s3://{bucket}/{key}: {e}")
        return False

    raw = obj["Body"].read()
    # In SES->S3 pipeline, object is the raw RFC822 email
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    message_id = (msg.get("Message-Id") or "").strip()
    provider_id = (obj.get("ETag") or "").strip('"')

    subj = clean_subject(msg.get("Subject"))
    body_text, body_html = extract_text_from_message(msg)

    subject_norm = normalize_text(subj)
    body_text_norm = normalize_text(body_text)
    body_html_norm = normalize_text(body_html)

    known_raw = load_known_rfq_ids()
    known_canon = {canonicalize_rfq_id(k) for k in known_raw if k}

    rfq_id, matched_via = match_email_to_rfq(
        known_canon if known_canon else None,
        subject_norm,
        body_text_norm,
        body_html_norm,
        msg,
        s3_key=key,
    )

    if not rfq_id:
        dead_letter("no_rfq_id_found", {
            "s3_key": key,
            "message_id": message_id,
            "subject": subj[:300],
        })
        # Mark processed to avoid infinite retries; adjust if you want to reprocess later
        mark_processed(key, message_id, provider_id, rfq_id or "")
        return True

    # >>> Hand off to your pipeline here (NegotiationAgent → DraftingAgent) <<<
    log.info(f"Matched RFQ {rfq_id} via {matched_via} | subject='{subj}'")

    # Example hand-off stub:
    # strategy = negotiation_agent.analyze_email(rfq_id=rfq_id, email_raw=raw)
    # draft = drafting_agent.compose_reply(rfq_id=rfq_id, strategy=strategy)
    # send_email_with_anchors(draft, rfq_id)

    mark_processed(key, message_id, provider_id, rfq_id)
    return True

def handle_sqs_message(m: dict) -> bool:
    """
    Returns True on success (safe to delete). False to retry later.
    Expects S3 event notification format in the message body (direct or SNS-wrapped).
    """
    try:
        body = m.get("Body") or "{}"
        payload = json.loads(body)

        # SNS envelope support (if S3→SNS→SQS)
        if "Type" in payload and payload.get("TopicArn") and payload.get("Message"):
            payload = json.loads(payload["Message"])

        # S3 event
        recs = payload.get("Records") or []
        ok_all = True
        for r in recs:
            s3info = r.get("s3", {})
            bucket = s3info.get("bucket", {}).get("name")
            key = s3info.get("object", {}).get("key")
            if not bucket or not key:
                log.error(f"Malformed S3 record: {r}")
                ok_all = False
                continue
            # URL-decoding is handled by boto3, but keys in event are URL-encoded
            from urllib.parse import unquote_plus
            key = unquote_plus(key)

            ok = process_s3_record(bucket, key)
            ok_all = ok_all and ok

        return ok_all
    except Exception as e:
        log.exception(f"Failed to handle SQS message: {e}")
        return False

def run_loop():
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,       # long polling
            VisibilityTimeout=max(30, LOOKBACK_MIN * 60 // 3),
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        to_delete = []
        for m in msgs:
            ok = handle_sqs_message(m)
            if ok and DELETE_ON_SUCCESS:
                to_delete.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})

        if to_delete:
            try:
                sqs.delete_message_batch(QueueUrl=QUEUE_URL, Entries=to_delete)
            except ClientError as e:
                log.error(f"Delete batch failed: {e}")

if __name__ == "__main__":
    run_loop()
