from typing import Optional, Any, Dict

def run_email_watcher_for_workflow(
    workflow_id: str,
    run_id: Optional[str],
    wait_seconds_after_last_dispatch: int = 0,
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
    supplier_agent: Optional[Any] = None,
    negotiation_agent: Optional[Any] = None,
    workflow_memory: Optional[Dict[str, Any]] = None,
) -> None:
    """Run email watcher for a specific workflow."""
    from services.email_watcher_service import EmailWatcherService
    
    watcher = EmailWatcherService()
    watcher.watch_workflow(
        workflow_id=workflow_id,
        run_id=run_id,
        wait_seconds_after_last_dispatch=wait_seconds_after_last_dispatch,
        agent_registry=agent_registry,
        orchestrator=orchestrator, 
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        workflow_memory=workflow_memory
    )
