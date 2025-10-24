-- Make workflow_email_tracking.dispatch_key optional
BEGIN;

ALTER TABLE proc.workflow_email_tracking
    ADD COLUMN IF NOT EXISTS dispatch_key TEXT;

DO $$
DECLARE
    constraint_name text;
    columns text[];
BEGIN
    SELECT
        tc.constraint_name,
        array_agg(kcu.column_name ORDER BY kcu.ordinal_position)
    INTO constraint_name, columns
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema = 'proc'
      AND tc.table_name = 'workflow_email_tracking'
      AND tc.constraint_type = 'PRIMARY KEY'
    GROUP BY tc.constraint_name;

    IF constraint_name IS NOT NULL AND (
        columns <> ARRAY['workflow_id', 'unique_id']
        OR 'dispatch_key' = ANY(columns)
    ) THEN
        EXECUTE format(
            'ALTER TABLE proc.workflow_email_tracking DROP CONSTRAINT %I',
            constraint_name
        );
        EXECUTE 'ALTER TABLE proc.workflow_email_tracking ADD PRIMARY KEY (workflow_id, unique_id)';
    ELSIF constraint_name IS NULL THEN
        EXECUTE 'ALTER TABLE proc.workflow_email_tracking ADD PRIMARY KEY (workflow_id, unique_id)';
    END IF;
END;
$$;

ALTER TABLE proc.workflow_email_tracking
    ALTER COLUMN dispatch_key DROP NOT NULL;

COMMIT;
