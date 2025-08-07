import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AuditService:
    """Service for audit logging and traceability"""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings

    def log_action(self, agent: str, action: str, context: Any,
                   output: Optional[Any] = None, workflow_id: Optional[str] = None):
        """Log agent action"""
        if not self.settings.audit_enabled:
            return

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Create audit table if not exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS proc.agent_audit_log (
                            id SERIAL PRIMARY KEY,
                            workflow_id VARCHAR(255),
                            agent_name VARCHAR(255),
                            action VARCHAR(50),
                            context_data JSONB,
                            output_data JSONB,
                            created_at TIMESTAMP DEFAULT NOW(),
                            created_by VARCHAR(100)
                        )
                    """)

                    # Insert audit record
                    cursor.execute("""
                        INSERT INTO proc.agent_audit_log 
                        (workflow_id, agent_name, action, context_data, output_data, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        workflow_id,
                        agent,
                        action,
                        json.dumps(self._serialize_context(context)),
                        json.dumps(self._serialize_output(output)) if output else None,
                        self.settings.script_user
                    ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def log_error(self, agent: str, error: str, context: Any,
                  workflow_id: Optional[str] = None):
        """Log agent error"""
        self.log_action(
            agent=agent,
            action="ERROR",
            context=context,
            output={'error': error},
            workflow_id=workflow_id
        )

    def _serialize_context(self, context: Any) -> Dict:
        """Serialize context for storage"""
        if hasattr(context, 'to_dict'):
            return context.to_dict()
        elif hasattr(context, '__dict__'):
            return {k: str(v) for k, v in context.__dict__.items()}
        return {'data': str(context)}

    def _serialize_output(self, output: Any) -> Dict:
        """Serialize output for storage"""
        if hasattr(output, '__dict__'):
            return {k: str(v) for k, v in output.__dict__.items()
                    if not k.startswith('_')}
        return {'data': str(output)}