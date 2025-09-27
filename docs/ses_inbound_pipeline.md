# SES Inbound Pipeline Hardening Checklist

The `SESEmailWatcher` expects inbound supplier replies to arrive through Amazon
SES and land in S3 with accompanying SQS notifications. Use the checklist below
when onboarding a new mailbox or troubleshooting missed replies.

> **Operational note (March 2025):** SQS polling is temporarily disabled in
> production while the `procwise-inbound-queue` resource is unavailable. The
> watcher now defaults to S3 polling only. Set
> `ses_inbound_queue_enabled=True` once the queue is restored to re-enable the
> push-based path described below.

## 1. Verify the domain and MX routing
- Confirm that the sending domain is verified in SES and that the MX record for
your mailbox points at `inbound-smtp.<region>.amazonaws.com`.
- If you operate multiple regions, update the watcher’s `ses_region` setting so
assumed roles and download clients are initialised against the same region that
handles inbound mail.

## 2. Activate an inbound rule set with S3 delivery
- Enable an inbound rule set for each domain or specific recipient address.
- Add an **S3 action** that stores the raw `.eml` payloads in the bucket the
watcher reads from. SES does not allow a Lambda-only action; S3 delivery is
mandatory.
- Optionally add a Lambda invocation after the S3 action if you need additional
side effects, but keep the S3 write as the first action in the chain.

## 3. Publish S3 notifications to SQS (push > poll)
- Configure the bucket to emit `s3:ObjectCreated:*` events to an SQS queue and
expose the queue URL through the `ses_inbound_queue_url` setting.
- Ensure the queue policy allows the bucket (and SES service principal) to send
notifications. The watcher’s `_load_from_queue()` path consumes SNS/SQS payloads
and downloads the referenced object directly from S3.
- Relying purely on periodic S3 polling is eventually consistent and can miss
bursty traffic. The queue delivers messages immediately so replies are processed
as soon as they arrive.

## 4. Align prefixes and mailbox addressing
- SES writes to the prefix supplied in the receipt rule. The watcher builds a
set of candidate prefixes from `ses_inbound_prefix` and the mailbox address
(e.g. `ses/inbound/supplier@example.com/`). Use the new initialisation logs to
confirm that the normalised prefixes match what SES is writing.
- If SES uses a different prefix (such as `raw-mails/`), update
`ses_inbound_prefix` to match or supply multiple prefixes (comma-separated) so
the watcher includes them during scans.

## 5. Validate permissions and batching parameters
- When the inbox bucket lives in another account, supply
`ses_inbound_role_arn` so the watcher can assume cross-account access.
- Tune `ses_inbound_queue_wait_seconds` and `ses_inbound_queue_max_messages` to
match the queue’s throughput characteristics if you expect a high volume of
emails.

Following the steps above ensures inbound replies flow from SES → S3 → SQS →
`SESEmailWatcher` without gaps, while keeping the configuration visible through
the watcher’s startup logs.
