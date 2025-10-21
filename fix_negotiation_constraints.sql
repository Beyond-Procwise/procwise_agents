-- Fix negotiation_sessions constraints
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

-- Ensure all required columns exist
ALTER TABLE proc.negotiation_sessions 
    ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255);

-- Create the correct unique constraint
-- Choose ONE of the following based on your schema preference:

-- Option A: If using workflow_id as primary identifier
CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_sessions_workflow_supplier_round
    ON proc.negotiation_sessions (workflow_id, supplier_id, round)
    WHERE workflow_id IS NOT NULL;

-- Option B: If using rfq_id as primary identifier  
CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_sessions_rfq_supplier_round
    ON proc.negotiation_sessions (rfq_id, supplier_id, round)
    WHERE rfq_id IS NOT NULL;

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

-- Ensure required columns exist
ALTER TABLE proc.negotiation_session_state
    ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255);

-- Create correct unique constraint
CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_session_state_workflow_supplier
    ON proc.negotiation_session_state (workflow_id, supplier_id)
    WHERE workflow_id IS NOT NULL;

COMMIT;
