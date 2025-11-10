from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from typing import List

from loguru import logger
from sqlalchemy import text

from .config import AppConfig
from .procurement_knowledge_graph import ProcurementKnowledgeGraph
from .procwise_kg_integration import ProcWiseKnowledgeGraphAgent


class DemoDataLoader:
    def __init__(self, config: AppConfig) -> None:
        from sqlalchemy import create_engine

        self._engine = create_engine(config.postgres.dsn)

    def _exec(self, statement: str, **params) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(statement), params)

    def _exec_many(self, statement: str, rows: List[dict]) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(statement), rows)

    def prepare_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS proc.supplier (
                supplier_id SERIAL PRIMARY KEY,
                supplier_name TEXT,
                risk_score NUMERIC,
                is_preferred_supplier BOOLEAN,
                esg_certifications TEXT,
                diversity_flags TEXT,
                contact_email TEXT,
                contact_phone TEXT,
                country TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.contracts (
                contract_id SERIAL PRIMARY KEY,
                supplier_id INTEGER REFERENCES proc.supplier(supplier_id),
                contract_type TEXT,
                total_contract_value NUMERIC,
                contract_lifecycle_status TEXT,
                payment_terms TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.purchase_order_agent (
                po_id SERIAL PRIMARY KEY,
                supplier_name TEXT,
                supplier_id INTEGER,
                total_amount NUMERIC,
                po_status TEXT,
                contract_id INTEGER,
                order_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.po_line_items_agent (
                po_id INTEGER,
                line_number INTEGER,
                item_description TEXT,
                quantity NUMERIC,
                unit_price NUMERIC,
                total_amount NUMERIC,
                PRIMARY KEY (po_id, line_number)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.quote_agent (
                quote_id SERIAL PRIMARY KEY,
                supplier_id INTEGER,
                po_id INTEGER,
                quote_date TIMESTAMP,
                total_amount_incl_tax NUMERIC,
                validity_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.quote_line_items_agent (
                quote_line_id SERIAL PRIMARY KEY,
                quote_id INTEGER REFERENCES proc.quote_agent(quote_id),
                item_description TEXT,
                quantity NUMERIC,
                unit_price NUMERIC,
                total_amount NUMERIC
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.invoice_agent (
                invoice_id SERIAL PRIMARY KEY,
                po_id INTEGER,
                invoice_total_incl_tax NUMERIC,
                invoice_status TEXT,
                payment_terms TEXT,
                issued_on TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.negotiation_sessions (
                rfq_id TEXT,
                supplier_id INTEGER,
                round INTEGER,
                counter_offer NUMERIC,
                created_on TIMESTAMP,
                PRIMARY KEY (rfq_id, supplier_id, round)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.negotiation_session_state (
                rfq_id TEXT,
                supplier_id INTEGER,
                supplier_reply_count INTEGER,
                current_round INTEGER,
                status TEXT,
                awaiting_response BOOLEAN,
                PRIMARY KEY (rfq_id, supplier_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.supplier_response (
                workflow_id TEXT,
                unique_id TEXT,
                rfq_id TEXT,
                supplier_id INTEGER,
                supplier_email TEXT,
                price NUMERIC,
                payment_terms TEXT,
                lead_time TEXT,
                round_number INTEGER,
                processed BOOLEAN,
                message_id TEXT,
                responded_at TIMESTAMP,
                PRIMARY KEY (workflow_id, unique_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.workflow_email_tracking (
                workflow_id TEXT,
                unique_id TEXT,
                rfq_id TEXT,
                supplier_id INTEGER,
                message_id TEXT,
                references TEXT,
                dispatched_at TIMESTAMP,
                responded_at TIMESTAMP,
                matched BOOLEAN,
                PRIMARY KEY (workflow_id, unique_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS proc.cat_product_mapping (
                category_level_1 TEXT,
                category_level_2 TEXT,
                category_level_3 TEXT,
                category_level_4 TEXT,
                category_level_5 TEXT,
                product TEXT
            )
            """,
        ]
        for statement in statements:
            self._exec(statement)

    def load_sample_data(self) -> None:
        self.prepare_schema()
        self._exec("DELETE FROM proc.workflow_email_tracking")
        self._exec("DELETE FROM proc.supplier_response")
        self._exec("DELETE FROM proc.negotiation_session_state")
        self._exec("DELETE FROM proc.negotiation_sessions")
        self._exec("DELETE FROM proc.invoice_agent")
        self._exec("DELETE FROM proc.quote_line_items_agent")
        self._exec("DELETE FROM proc.quote_agent")
        self._exec("DELETE FROM proc.po_line_items_agent")
        self._exec("DELETE FROM proc.purchase_order_agent")
        self._exec("DELETE FROM proc.contracts")
        self._exec("DELETE FROM proc.supplier")
        self._exec("DELETE FROM proc.cat_product_mapping")

        suppliers = []
        for idx in range(1, 11):
            suppliers.append(
                {
                    "supplier_name": f"Supplier {idx}",
                    "risk_score": random.randint(1, 5),
                    "is_preferred_supplier": idx % 3 == 0,
                    "esg_certifications": "ISO14001" if idx % 2 == 0 else "",
                    "diversity_flags": "Women-owned" if idx % 4 == 0 else "",
                    "contact_email": f"supplier{idx}@example.com",
                    "contact_phone": "+1-555-0100",
                    "country": "USA" if idx % 2 == 0 else "Germany",
                }
            )
        self._exec_many(
            """
            INSERT INTO proc.supplier (
                supplier_name, risk_score, is_preferred_supplier, esg_certifications,
                diversity_flags, contact_email, contact_phone, country
            ) VALUES (
                :supplier_name, :risk_score, :is_preferred_supplier, :esg_certifications,
                :diversity_flags, :contact_email, :contact_phone, :country
            )
            """,
            suppliers,
        )

        contracts = []
        for supplier_id in range(1, 6):
            contracts.append(
                {
                    "supplier_id": supplier_id,
                    "contract_type": "Annual",
                    "total_contract_value": random.randint(100_000, 500_000),
                    "contract_lifecycle_status": "active",
                    "payment_terms": "Net 30",
                    "start_date": datetime.now() - timedelta(days=365),
                    "end_date": datetime.now() + timedelta(days=365),
                }
            )
        self._exec_many(
            """
            INSERT INTO proc.contracts (
                supplier_id, contract_type, total_contract_value, contract_lifecycle_status,
                payment_terms, start_date, end_date
            ) VALUES (
                :supplier_id, :contract_type, :total_contract_value, :contract_lifecycle_status,
                :payment_terms, :start_date, :end_date
            )
            """,
            contracts,
        )

        po_rows = []
        po_line_rows = []
        quote_rows = []
        quote_line_rows = []
        invoice_rows = []
        category_rows = []
        base_date = datetime.now() - timedelta(days=60)
        for po_id in range(1, 51):
            supplier_id = random.randint(1, 10)
            amount = random.randint(5_000, 50_000)
            po_rows.append(
                {
                    "po_id": po_id,
                    "supplier_name": f"Supplier {supplier_id}",
                    "supplier_id": supplier_id,
                    "total_amount": amount,
                    "po_status": random.choice(["draft", "approved", "closed"]),
                    "contract_id": random.randint(1, 5),
                    "order_date": base_date + timedelta(days=random.randint(0, 60)),
                }
            )
            po_line_rows.append(
                {
                    "po_id": po_id,
                    "line_number": 1,
                    "item_description": f"Item {po_id}",
                    "quantity": random.randint(10, 100),
                    "unit_price": amount / 10,
                    "total_amount": amount,
                }
            )
            quote_rows.append(
                {
                    "quote_id": po_id,
                    "supplier_id": supplier_id,
                    "po_id": po_id if po_id % 5 == 0 else None,
                    "quote_date": base_date + timedelta(days=random.randint(0, 60)),
                    "total_amount_incl_tax": amount * 1.05,
                    "validity_date": datetime.now() + timedelta(days=30),
                }
            )
            quote_line_rows.append(
                {
                    "quote_id": po_id,
                    "item_description": f"Item {po_id}",
                    "quantity": random.randint(10, 100),
                    "unit_price": amount / 10,
                    "total_amount": amount,
                }
            )
            invoice_rows.append(
                {
                    "invoice_id": po_id,
                    "po_id": po_id,
                    "invoice_total_incl_tax": amount * 1.05,
                    "invoice_status": random.choice(["pending", "paid"]),
                    "payment_terms": "Net 30",
                    "issued_on": datetime.now() - timedelta(days=random.randint(0, 30)),
                }
            )
            category_rows.append(
                {
                    "category_level_1": "Category A",
                    "category_level_2": "Subcategory B",
                    "category_level_3": "",
                    "category_level_4": "",
                    "category_level_5": "",
                    "product": f"Item {po_id}",
                }
            )
        self._exec_many(
            """
            INSERT INTO proc.purchase_order_agent (
                po_id, supplier_name, supplier_id, total_amount, po_status, contract_id, order_date
            ) VALUES (
                :po_id, :supplier_name, :supplier_id, :total_amount, :po_status, :contract_id, :order_date
            )
            """,
            po_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.po_line_items_agent (
                po_id, line_number, item_description, quantity, unit_price, total_amount
            ) VALUES (
                :po_id, :line_number, :item_description, :quantity, :unit_price, :total_amount
            )
            """,
            po_line_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.quote_agent (
                quote_id, supplier_id, po_id, quote_date, total_amount_incl_tax, validity_date
            ) VALUES (
                :quote_id, :supplier_id, :po_id, :quote_date, :total_amount_incl_tax, :validity_date
            )
            ON CONFLICT (quote_id) DO UPDATE
            SET total_amount_incl_tax = EXCLUDED.total_amount_incl_tax
            """,
            quote_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.quote_line_items_agent (
                quote_id, item_description, quantity, unit_price, total_amount
            ) VALUES (
                :quote_id, :item_description, :quantity, :unit_price, :total_amount
            )
            """,
            quote_line_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.invoice_agent (
                invoice_id, po_id, invoice_total_incl_tax, invoice_status, payment_terms, issued_on
            ) VALUES (
                :invoice_id, :po_id, :invoice_total_incl_tax, :invoice_status, :payment_terms, :issued_on
            )
            ON CONFLICT (invoice_id) DO UPDATE
            SET invoice_total_incl_tax = EXCLUDED.invoice_total_incl_tax
            """,
            invoice_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.cat_product_mapping (
                category_level_1, category_level_2, category_level_3, category_level_4, category_level_5, product
            ) VALUES (
                :category_level_1, :category_level_2, :category_level_3, :category_level_4, :category_level_5, :product
            )
            """,
            category_rows,
        )

        negotiation_rows = []
        state_rows = []
        response_rows = []
        workflow_rows = []
        workflow_id = "WF_20250101_001"
        for supplier_id in range(1, 6):
            rfq_id = "RFQ-1000"
            negotiation_rows.append(
                {
                    "rfq_id": rfq_id,
                    "supplier_id": supplier_id,
                    "round": 1,
                    "counter_offer": random.randint(7_000, 12_000),
                    "created_on": datetime.now() - timedelta(days=2),
                }
            )
            state_rows.append(
                {
                    "rfq_id": rfq_id,
                    "supplier_id": supplier_id,
                    "supplier_reply_count": random.randint(0, 3),
                    "current_round": 1,
                    "status": random.choice(["awaiting", "complete"]),
                    "awaiting_response": random.choice([True, False]),
                }
            )
            message_id = f"<msg-{supplier_id}-r1@example.com>"
            workflow_rows.append(
                {
                    "workflow_id": workflow_id,
                    "unique_id": f"{rfq_id}_{supplier_id}_1",
                    "rfq_id": rfq_id,
                    "supplier_id": supplier_id,
                    "message_id": message_id,
                    "references": message_id,
                    "dispatched_at": datetime.now() - timedelta(days=1),
                    "responded_at": datetime.now() - timedelta(hours=random.randint(1, 48)),
                    "matched": True,
                }
            )
            response_rows.append(
                {
                    "workflow_id": workflow_id,
                    "unique_id": f"{rfq_id}_{supplier_id}_1",
                    "rfq_id": rfq_id,
                    "supplier_id": supplier_id,
                    "supplier_email": f"supplier{supplier_id}@example.com",
                    "price": random.randint(7_000, 12_000),
                    "payment_terms": "Net 30",
                    "lead_time": "14 days",
                    "round_number": 1,
                    "processed": False,
                    "message_id": f"<reply-{supplier_id}-r1@example.com>",
                    "responded_at": datetime.now() - timedelta(hours=random.randint(1, 48)),
                }
            )
        self._exec_many(
            """
            INSERT INTO proc.negotiation_sessions (
                rfq_id, supplier_id, round, counter_offer, created_on
            ) VALUES (
                :rfq_id, :supplier_id, :round, :counter_offer, :created_on
            )
            ON CONFLICT (rfq_id, supplier_id, round) DO UPDATE
            SET counter_offer = EXCLUDED.counter_offer
            """,
            negotiation_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.negotiation_session_state (
                rfq_id, supplier_id, supplier_reply_count, current_round, status, awaiting_response
            ) VALUES (
                :rfq_id, :supplier_id, :supplier_reply_count, :current_round, :status, :awaiting_response
            )
            ON CONFLICT (rfq_id, supplier_id) DO UPDATE
            SET supplier_reply_count = EXCLUDED.supplier_reply_count
            """,
            state_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.workflow_email_tracking (
                workflow_id, unique_id, rfq_id, supplier_id, message_id, references, dispatched_at, responded_at, matched
            ) VALUES (
                :workflow_id, :unique_id, :rfq_id, :supplier_id, :message_id, :references, :dispatched_at, :responded_at, :matched
            )
            ON CONFLICT (workflow_id, unique_id) DO UPDATE
            SET responded_at = EXCLUDED.responded_at
            """,
            workflow_rows,
        )
        self._exec_many(
            """
            INSERT INTO proc.supplier_response (
                workflow_id, unique_id, rfq_id, supplier_id, supplier_email, price, payment_terms,
                lead_time, round_number, processed, message_id, responded_at
            ) VALUES (
                :workflow_id, :unique_id, :rfq_id, :supplier_id, :supplier_email, :price, :payment_terms,
                :lead_time, :round_number, :processed, :message_id, :responded_at
            )
            ON CONFLICT (workflow_id, unique_id) DO UPDATE
            SET price = EXCLUDED.price
            """,
            response_rows,
        )

        logger.info("Sample data loaded with workflow_id=WF_20250101_001 and rfq_id=RFQ-1000")


def run_demo(load_sample_data: bool, demo: bool) -> None:
    config = AppConfig.from_env()
    if load_sample_data:
        loader = DemoDataLoader(config)
        loader.load_sample_data()

    builder = ProcurementKnowledgeGraph.from_config(config)
    builder.full_refresh()

    agent = ProcWiseKnowledgeGraphAgent(config)

    workflow_id = "WF_20250101_001"
    rfq_id = "RFQ-1000"
    trigger = agent.should_trigger_negotiation(workflow_id, rfq_id, min_responses_required=3)
    logger.info("Negotiation trigger check: {result}", result=trigger)

    status = agent.get_workflow_status(workflow_id)
    logger.info("Workflow status: {status}", status=status)

    for supplier in status["suppliers"]:
        if supplier["has_responded"] and supplier["quoted_price"]:
            strategy = agent.generate_negotiation_strategy(
                rfq_id=rfq_id,
                supplier_id=str(supplier["supplier_id"]),
                target_price=float(supplier["quoted_price"]) * 0.95,
                current_quote=float(supplier["quoted_price"]),
            )
            logger.info("Strategy for supplier {supplier}: {strategy}", supplier=supplier["supplier_id"], strategy=strategy)
            email = agent.generate_counter_offer_email(
                rfq_id=rfq_id,
                supplier_id=str(supplier["supplier_id"]),
                counter_offer=strategy["strategy"].get("recommended_counter_offer", supplier["quoted_price"]),
                round_number=2,
            )
            logger.info("Generated email: {email}", email=email)

    if demo:
        answer = agent.natural_language_query("Which suppliers have the best win rates for electronics category?")
        logger.info("Natural language answer: {answer}", answer=answer)

    agent.close()
    builder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end knowledge graph demo")
    parser.add_argument("--load-sample-data", action="store_true", help="Seed PostgreSQL with sample data")
    parser.add_argument("--demo", action="store_true", help="Run negotiation demo interactions")
    args = parser.parse_args()
    run_demo(load_sample_data=args.load_sample_data, demo=args.demo)


if __name__ == "__main__":
    main()
