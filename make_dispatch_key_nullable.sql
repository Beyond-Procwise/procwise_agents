-- Make workflow_email_tracking.dispatch_key optional
BEGIN;

ALTER TABLE proc.workflow_email_tracking
    ADD COLUMN IF NOT EXISTS dispatch_key TEXT;

ALTER TABLE proc.workflow_email_tracking
    ALTER COLUMN dispatch_key DROP NOT NULL;

COMMIT;
