# Beyond ProcWise Configuration Reference

This document summarises the runtime settings that can be adjusted via
environment variables using ``pydantic-settings``.  The newly introduced
options allow teams to switch between the existing Ollama pipeline and a
LangChain/LangGraph stack while keeping human-in-the-loop (HITL) safeguards and
memory integrations aligned with procurement workflows.

## LLM provider selection

| Environment variable | Default | Description |
| -------------------- | ------- | ----------- |
| ``LLM_BACKEND`` | ``"ollama"`` | Selects the runtime for agent LLM calls. Set to ``"langchain"`` to route requests through LangChain chat models with automatic fallback to the native Ollama client on failure. |
| ``LANGCHAIN_PROVIDER`` | ``"ollama"`` | Chooses the LangChain provider shim to initialise. Supported values include ``ollama`` (local models), ``openai``/``azure-openai`` (remote APIs), or any identifier accepted by ``langchain.chat_models.init_chat_model``. |
| ``LANGCHAIN_MODEL`` | ``"qwen3:30b"`` | Model identifier forwarded to the selected provider. Required when ``LLM_BACKEND=langchain``. |
| ``LANGCHAIN_API_BASE`` | ``None`` | Optional custom API endpoint (e.g. ``http://localhost:11434`` for self-hosted Ollama, or a vendor-specific gateway). |
| ``LANGCHAIN_API_KEY`` | ``None`` | Secret token used for hosted providers such as OpenAI or Azure OpenAI. |
| ``LANGCHAIN_REQUEST_TIMEOUT`` | ``120`` | Timeout (seconds) applied to LangChain invocations via ``Runnable`` configuration. |

With ``LLM_BACKEND=langchain`` the system reuses the standard agent prompts but
converts chat payloads into LangChain message objects.  If a LangChain call
raises an exception, the agent automatically retries using the locally
installed Ollama models to preserve continuity during supplier negotiations.

## Human-in-the-loop and memory settings

| Environment variable | Default | Description |
| -------------------- | ------- | ----------- |
| ``HITL_ENABLED`` | ``True`` | Toggles enforcement of human approval gates before supplier communications are dispatched. This should remain ``True`` in production to satisfy procurement review policies. |
| ``MEMORY_STORE_URI`` | ``None`` | Connection string for an external conversational memory or vector store (e.g. Redis, Postgres, or a LangGraph memory backend). Agents can persist negotiation snippets here when enabled. |
| ``LANGGRAPH_TRACING_ENABLED`` | ``False`` | Enables LangGraph tracing/inspection hooks for advanced debugging. When ``True`` the orchestrator records LangGraph thread identifiers alongside the existing process routing metadata. |

These controls support the gradual migration toward LangGraph-driven flows by
keeping long-term memory and HITL checkpoints configurable per environment.

## Related files

- ``config/settings.py`` — canonical definition of settings and defaults.
- ``agents/base_agent.py`` — runtime logic that switches between Ollama and
  LangChain backends.
- ``services/rag_service.py`` — exposes the Qdrant-backed retriever as a
  LangChain tool for graph orchestration.
