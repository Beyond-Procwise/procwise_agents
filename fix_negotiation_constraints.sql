BEGIN;

-- Drop any existing problematic constraints
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_schema = 'proc'
          AND table_name = 'negotiation_sessions'
          AND constraint_type = 'UNIQUE'
    LOOP
        EXECUTE 'ALTER TABLE proc.negotiation_sessions DROP CONSTRAINT IF EXISTS ' || quote_ident(r.constraint_name);
    END LOOP;
END $$;

-- Drop stray unique indexes so we can enforce the canonical key
DO $$
DECLARE
    idx RECORD;
BEGIN
    FOR idx IN
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'proc'
          AND tablename = 'negotiation_sessions'
          AND indexdef ILIKE 'CREATE UNIQUE INDEX%'
          AND indexname NOT IN (
              'uq_negotiation_sessions_workflow_supplier_round',
              'negotiation_sessions_unique_supplier_round_idx'
          )
    LOOP
        EXECUTE 'DROP INDEX IF EXISTS ' || quote_ident('proc') || '.' || quote_ident(idx.indexname);
    END LOOP;
END $$;

-- Ensure all required columns exist
ALTER TABLE proc.negotiation_sessions
    ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255);

-- Create the correct unique constraint aligned to workflow_id
CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_sessions_workflow_supplier_round
    ON proc.negotiation_sessions (workflow_id, supplier_id, round)
    WHERE workflow_id IS NOT NULL;

-- Fix negotiation_session_state constraints
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN 
        SELECT constraint_name 
        FROM information_schema.table_constraints 
        WHERE table_schema = 'proc'
          AND table_name = 'negotiation_session_state'
          AND constraint_type = 'UNIQUE'
    LOOP
        EXECUTE 'ALTER TABLE proc.negotiation_session_state DROP CONSTRAINT IF EXISTS ' || quote_ident(r.constraint_name);
    END LOOP;
END $$;

-- Drop legacy unique indexes that conflict with workflow_id usage
DO $$
DECLARE
    idx RECORD;
BEGIN
    FOR idx IN
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'proc'
          AND tablename = 'negotiation_session_state'
          AND indexdef ILIKE 'CREATE UNIQUE INDEX%'
          AND indexname NOT IN (
              'uq_negotiation_session_state_workflow_supplier',
              'negotiation_session_state_unique_supplier_idx'
          )
    LOOP
        EXECUTE 'DROP INDEX IF EXISTS ' || quote_ident('proc') || '.' || quote_ident(idx.indexname);
    END LOOP;
END $$;

-- Ensure required columns exist
ALTER TABLE proc.negotiation_session_state
    ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255);

-- Create correct unique constraint
CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_session_state_workflow_supplier
    ON proc.negotiation_session_state (workflow_id, supplier_id)
    WHERE workflow_id IS NOT NULL;

COMMIT;
