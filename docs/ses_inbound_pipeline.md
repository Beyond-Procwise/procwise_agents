# SES Inbound Pipeline Checklist (S3 Only)

Supplier responses are now processed exclusively from the `s3://procwisemvp/emails/`
prefix. The `SESEmailWatcher` no longer contacts IMAP inboxes or SQS queues; it
simply polls the bucket at regular intervals until a matching RFQ is found. Use
the checklist below when onboarding a new mailbox or troubleshooting delayed
replies.

## 1. Confirm S3 delivery
- Ensure the SES receipt rule writes the raw `.eml` payloads into the
  `procwisemvp` bucket under the `emails/` prefix.
- When using a different bucket or prefix for testing, update the watcher
  settings accordingly or copy the files into `procwisemvp/emails/`.
- Grant the watcher’s execution role `s3:GetObject` and `s3:ListBucket`
  permissions for the chosen prefix.

## 2. Validate watcher configuration
- `SESEmailWatcher` defaults to bucket `procwisemvp` and prefix `emails/`. No
  additional configuration is required unless you deliberately store replies
  elsewhere.
- Check the startup log entry `Email watcher will poll s3://...` to confirm the
  watcher is observing the expected location.
- Adjust `email_response_poll_seconds` to tune how frequently the bucket is
  scanned for new objects.

## 3. Keep RFQ identifiers intact
- The watcher extracts the RFQ ID from the subject, body, or inline HTML
  comments. Remind suppliers to keep the identifier in their replies so the
  polling loop stops as soon as a matching message is detected.
- When testing, upload sample emails that include a valid `RFQ-YYYYMMDD-xxxx` ID
  in the body or subject.

## 4. Monitoring tips
- CloudWatch metrics on the bucket’s `PutObject` events provide early visibility
  that a supplier reply has been stored.
- The watcher logs `Scanning S3 bucket ...` on every poll and `Processed message`
  once a reply has been handled. Enable `INFO` logging when debugging.
- Use the cached results (`email_processed_cache_limit`) to avoid reprocessing
  the same object when polling for a specific RFQ multiple times.

With the S3-only flow there are fewer moving parts: once SES writes the file to
`procwisemvp/emails/`, the watcher will pick it up without waiting on queues or
mailbox connections.
