# SES Inbound Pipeline Checklist (S3 → SQS → Lambda)

Supplier responses flow through an event-driven pipeline that starts with SES
writing raw `.eml` payloads into the `s3://procwisemvp/emails/` prefix. Every
object creation triggers an S3 notification which fans out to an SQS queue. A
dedicated Lambda (`EmailWatcher`) reads from the queue, normalises the payload
path, and tags the object with the resolved RFQ identifier before downstream
processing. Use the checklist below when onboarding a new mailbox or
troubleshooting delayed replies.

## 1. Confirm S3 delivery
- Ensure the SES receipt rule writes the raw `.eml` payloads into the
  `procwisemvp` bucket under the `emails/` prefix.
- When using a different bucket or prefix for testing, update the watcher
  settings accordingly or copy the files into `procwisemvp/emails/`.
- Grant the watcher’s execution role `s3:GetObject` and `s3:ListBucket`
  permissions for the chosen prefix.
- Configure an S3 Event Notification on `procwisemvp/emails/` that targets the
  SQS queue used by the Lambda.

## 2. Validate watcher configuration
- The Lambda is configured via environment variables (`EMAIL_INGEST_BUCKET`,
  `EMAIL_INGEST_PREFIX`, `EMAIL_THREAD_TABLE`). Defaults point to
  `procwisemvp`, `emails/`, and `procwise_outbound_emails` respectively.
- The function copies every processed object into
  `emails/<RFQ-ID>/ingest/<original-name>` and applies the tag `rfq-id=<RFQ-ID>`.
  Downstream processors should read from the normalised prefix or filter by tag.
- Each processed message is upserted into Postgres (`proc.supplier_replies`) so
  the orchestrator can reconcile supplier replies without re-downloading from
  S3.

- Replies that cannot be matched to an RFQ are copied into
  `emails/_unmatched/` with the tag `needs-review=true` for manual triage.

## 3. Keep RFQ identifiers intact
- The watcher extracts the RFQ ID from the subject, body, or inline HTML
  comments. Remind suppliers to keep the identifier in their replies so the
  polling loop stops as soon as a matching message is detected.
- When testing, upload sample emails that include a valid `RFQ-YYYYMMDD-xxxx` ID
  in the body or subject.

## 4. Monitoring tips
- CloudWatch metrics on the bucket’s `PutObject` events provide early visibility
  that a supplier reply has been stored.
- Track SQS queue depth to ensure the Lambda is draining events quickly. Any
  spikes typically indicate IAM issues preventing S3 access.
- The Lambda logs whether an RFQ was resolved or whether the message moved to
  the `_unmatched` prefix. Enable `INFO` logging when debugging.

With the event-driven flow there are no polling delays: once SES writes the
file to `procwisemvp/emails/`, the Lambda processes it immediately and exposes a
stable key structure for downstream jobs.
