#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_service() {
  local name="$1"
  local url="$2"
  local max_attempts=${3:-60}
  local attempt=0
  until curl -k -sSf "$url" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [[ $attempt -ge $max_attempts ]]; then
      log "ERROR: $name did not become ready at $url"
      exit 1
    fi
    log "Waiting for $name to be ready (attempt $attempt/$max_attempts)..."
    sleep 5
  done
  log "$name is ready."
}

log "Starting Docker services"
docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d

log "Waiting for Postgres"
wait_for_service "Postgres" "http://localhost:5432" 1 || true
until docker exec $(docker compose -f "$PROJECT_DIR/docker-compose.yml" ps -q postgres) pg_isready -U "${POSTGRES_USER:-procwise}" >/dev/null 2>&1; do
  log "Waiting for Postgres acceptance..."
  sleep 5
done
log "Postgres ready."

log "Waiting for Neo4j HTTP"
wait_for_service "Neo4j" "http://localhost:7474" 30

log "Waiting for Qdrant"
wait_for_service "Qdrant" "http://localhost:6333/healthz" 30

log "Waiting for Ollama"
wait_for_service "Ollama" "http://localhost:11434/api/tags" 30

log "Pulling Ollama models"
docker exec $(docker compose -f "$PROJECT_DIR/docker-compose.yml" ps -q ollama) ollama pull llama3.2 || true
docker exec $(docker compose -f "$PROJECT_DIR/docker-compose.yml" ps -q ollama) ollama pull all-minilm || true

log "Creating Python virtual environment"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

log "Testing connections"
python - <<'PYTEST'
from procurement_knowledge_graph.procurement_knowledge_graph import ProcurementKnowledgeGraph
from procurement_knowledge_graph.config import (
    AppConfig,
    app_config_from_global_settings,
)

try:
    config = app_config_from_global_settings()
except RuntimeError:
    from config.settings import settings

    config = AppConfig.from_settings(settings)

kg = ProcurementKnowledgeGraph.from_config(config)
kg.test_connections()
print("All connections OK")
PYTEST

log "Setup completed successfully"
