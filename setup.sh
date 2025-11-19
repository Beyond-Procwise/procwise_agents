#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run this project." >&2
  exit 1
fi

if ! command -v docker compose >/dev/null 2>&1; then
  echo "docker compose plugin is required." >&2
  exit 1
fi

export POSTGRES_DSN="postgresql://procwise:procwise@localhost:5432/procwise"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY=""
export QDRANT_COLLECTION_PREFIX="demo_"
export LMSTUDIO_BASE_URL="http://127.0.0.1:1234"
export LMSTUDIO_CHAT_MODEL="microsoft/phi-4-reasoning-plus"
export LMSTUDIO_EMBED_MODEL="nomic-embed-text"
export LLM_DEFAULT_MODEL="phi4:latest"

COMPOSE_FILE="docker-compose.yml"

echo "Launching infrastructure..."
docker compose -f "$COMPOSE_FILE" up -d postgres neo4j qdrant

echo "Waiting for Postgres to be available..."
until docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U procwise -d procwise >/dev/null 2>&1; do
  sleep 2
  echo "  Postgres not ready yet..."
done

echo "Seeding minimal database schema..."
docker compose -f "$COMPOSE_FILE" exec -T postgres psql -U procwise -d procwise <<'SQL'
CREATE TABLE IF NOT EXISTS suppliers (
  supplier_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  risk_rating NUMERIC,
  performance_score NUMERIC,
  coverage_regions TEXT[],
  preferred BOOLEAN DEFAULT FALSE,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS contracts (
  contract_id TEXT PRIMARY KEY,
  supplier_id TEXT REFERENCES suppliers(supplier_id),
  contract_number TEXT,
  start_date DATE,
  end_date DATE,
  total_value NUMERIC,
  terms_summary TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS categories (
  category_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  parent_category_id TEXT REFERENCES categories(category_id),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  category_id TEXT REFERENCES categories(category_id),
  sku TEXT,
  unit_price NUMERIC,
  currency TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS purchase_orders (
  po_id TEXT PRIMARY KEY,
  supplier_id TEXT REFERENCES suppliers(supplier_id),
  contract_id TEXT REFERENCES contracts(contract_id),
  order_number TEXT,
  category_id TEXT REFERENCES categories(category_id),
  product_id TEXT REFERENCES products(product_id),
  status TEXT,
  total_amount NUMERIC,
  issued_at TIMESTAMPTZ,
  delivery_due TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS quotes (
  quote_id TEXT PRIMARY KEY,
  supplier_id TEXT REFERENCES suppliers(supplier_id),
  negotiation_session_id TEXT,
  total_amount NUMERIC,
  currency TEXT,
  validity_date DATE,
  notes TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS invoices (
  invoice_id TEXT PRIMARY KEY,
  po_id TEXT REFERENCES purchase_orders(po_id),
  invoice_number TEXT,
  amount_due NUMERIC,
  currency TEXT,
  due_date DATE,
  paid_at TIMESTAMPTZ,
  status TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS negotiation_sessions (
  negotiation_session_id TEXT PRIMARY KEY,
  workflow_id TEXT NOT NULL,
  rfq_reference TEXT,
  round_number INTEGER,
  started_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,
  status TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS supplier_responses (
  response_id TEXT PRIMARY KEY,
  negotiation_session_id TEXT REFERENCES negotiation_sessions(negotiation_session_id),
  supplier_id TEXT REFERENCES suppliers(supplier_id),
  round_number INTEGER,
  response_text TEXT,
  proposed_amount NUMERIC,
  received_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO suppliers (supplier_id, name, description, risk_rating, performance_score, preferred)
VALUES
  ('sup-100', 'Apex Manufacturing', 'Industrial components', 2.1, 87.2, TRUE),
  ('sup-200', 'LogiTrans', 'Global logistics partner', 1.8, 90.5, TRUE)
ON CONFLICT (supplier_id) DO UPDATE SET updated_at = NOW();
INSERT INTO categories (category_id, name)
VALUES ('cat-transport', 'Transportation'), ('cat-raw', 'Raw Materials')
ON CONFLICT (category_id) DO UPDATE SET updated_at = NOW();
INSERT INTO products (product_id, name, category_id, unit_price, currency)
VALUES ('prod-rail', 'Rail Freight Service', 'cat-transport', 12500, 'USD')
ON CONFLICT (product_id) DO UPDATE SET updated_at = NOW();
INSERT INTO contracts (contract_id, supplier_id, contract_number, start_date, end_date, total_value, terms_summary)
VALUES ('con-1', 'sup-200', 'LOG-2025-01', '2025-01-01', '2025-12-31', 1200000, 'Priority lanes, escalation in 24h')
ON CONFLICT (contract_id) DO UPDATE SET updated_at = NOW();
INSERT INTO negotiation_sessions (negotiation_session_id, workflow_id, rfq_reference, round_number, started_at, status)
VALUES ('neg-1', 'workflow-123', 'rfq-456', 1, NOW() - INTERVAL '3 days', 'in_progress')
ON CONFLICT (negotiation_session_id) DO UPDATE SET updated_at = NOW();
INSERT INTO supplier_responses (response_id, negotiation_session_id, supplier_id, round_number, response_text, proposed_amount, received_at)
VALUES
  ('resp-1', 'neg-1', 'sup-100', 1, 'Offering 3% discount for annual commitment.', 1215000, NOW() - INTERVAL '2 days'),
  ('resp-2', 'neg-1', 'sup-200', 1, 'Can match baseline with volume guarantee.', 1190000, NOW() - INTERVAL '1 days')
ON CONFLICT (response_id) DO UPDATE SET updated_at = NOW();
SQL

echo "Checking Neo4j connectivity..."
docker compose -f "$COMPOSE_FILE" exec -T neo4j cypher-shell -u neo4j -p password "RETURN 1;" >/dev/null

echo "Ensure LM Studio is running locally (default http://127.0.0.1:1234) with the required models loaded."
echo "All services are ready. Run 'docker compose up app' to execute the demo."
