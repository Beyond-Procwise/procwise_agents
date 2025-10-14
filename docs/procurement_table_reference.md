# Procurement Table Reference

This document captures the canonical structure of key procurement tables
used by agents in the ProcWise framework. It also outlines the
end-to-end data flow combining the vector database and PostgreSQL
sources.

## Data Flow Overview
1. **Contracts ➜ Suppliers**: `proc.contracts.supplier_id` joins to
   `proc.supplier.supplier_id` to fetch each `supplier_name`.
2. **Suppliers ➜ Purchase Orders**: The retrieved `supplier_name` is
   matched against `proc.purchase_order_agent.supplier_name` to gather
   purchase orders for that supplier.
3. **Purchase Orders ➜ Line Items & Invoices**:
   - `proc.po_line_items_agent.po_id` and `proc.invoice_agent.po_id`
     align with purchase orders to retrieve line items and invoices.
4. **Line Items ➜ Categories**: `proc.po_line_items_agent.item_description`
   is mapped to `proc.cat_product_mapping.product` to resolve
   `category_level_2` through `category_level_5`.
5. **Invoices ➜ Invoice Line Items**: `proc.invoice_line_items_agent`
   joins on `invoice_id` to provide detailed invoice lines.

## Table Definitions

### `proc.contracts`
```
contract_id text NOT NULL DEFAULT ('CO'::text || lpad((nextval('proc.con_sequence'::regclass))::text, 6, '0')),
contract_title text,
contract_type text,
supplier_id text,
buyer_org_id text,
contract_start_date date,
contract_end_date date,
currency text,
total_contract_value numeric(18,2),
spend_category text,
business_unit_id text,
cost_centre_id text,
is_amendment text,
parent_contract_id text,
auto_renew_flag text,
renewal_term text,
contract_lifecycle_status text,
jurisdiction text,
governing_law text,
contract_signatory_name text,
contract_signatory_role text,
payment_terms text,
risk_assessment_completed text,
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER,
last_modified_by text DEFAULT CURRENT_USER,
last_modified_date timestamp DEFAULT CURRENT_TIMESTAMP,
PRIMARY KEY (contract_id)
```

### `proc.supplier`
```
supplier_id text NOT NULL DEFAULT ('SI'::text || lpad((nextval('proc.si_sequence'::regclass))::text, 6, '0')),
supplier_name text,
trading_name text,
supplier_type text,
legal_structure varchar(10),
tax_id varchar(20),
vat_number varchar(20),
duns_number varchar(20),
parent_company_id text,
registered_country varchar(50),
registration_number varchar(50),
is_preferred_supplier boolean,
risk_score varchar(10),
credit_limit_amount numeric(18,2),
esg_cert_iso14001 boolean,
esg_cert_sa8000 boolean,
esg_cert_ecovadis boolean,
diversity_women_owned boolean,
diversity_minority_owned boolean,
diversity_veteran_owned boolean,
insurance_coverage_type varchar(30),
insurance_coverage_amount numeric(18,2),
insurance_expiry_date date,
bank_name text,
bank_account_number varchar(30),
bank_swift varchar(30),
bank_iban varchar(30),
default_currency varchar(5),
incoterms varchar(5),
delivery_lead_time_days varchar(3),
address_line1 text,
address_line2 text,
city text,
postal_code text,
country text,
website_url text,
edi_enabled boolean,
api_enabled boolean,
ariba_integrated boolean,
contact_name_1 text,
contact_role_1 text,
contact_email_1 text,
contact_phone_1 text,
contact_name_2 text,
contact_role_2 text,
contact_email_2 text,
contact_phone_2 text,
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER,
last_modified_by text DEFAULT CURRENT_USER,
last_modified_date timestamp DEFAULT CURRENT_TIMESTAMP,
PRIMARY KEY (supplier_id)
```

### `proc.purchase_order_agent`
```
po_id text NOT NULL DEFAULT ('PO'::text || lpad((nextval('proc.po_sequence_new'::regclass))::text, 6, '0')),
supplier_name text,
buyer_id text,
requisition_id text,
requested_by text,
requested_date date,
currency varchar(3),
order_date date,
expected_delivery_date date,
ship_to_country text,
delivery_region text,
incoterm text,
incoterm_responsibility text,
total_amount numeric(18,2),
delivery_address_line1 text,
delivery_address_line2 text,
delivery_city text,
postal_code text,
default_currency varchar(3),
po_status varchar(20),
payment_terms varchar(30),
exchange_rate_to_usd numeric(18,4),
converted_amount_usd numeric(18,4),
ai_flag_required varchar(5),
trigger_type varchar(30),
trigger_context_description text,
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER,
last_modified_by text DEFAULT CURRENT_USER,
last_modified_date timestamp DEFAULT CURRENT_TIMESTAMP,
contract_id text,
PRIMARY KEY (po_id)
```
### `proc.po_line_items_agent`'
```
    po_id text COLLATE pg_catalog."default" NOT NULL,
    po_line_id text COLLATE pg_catalog."default",
    line_number integer,
    item_id text COLLATE pg_catalog."default",
    item_description text COLLATE pg_catalog."default",
    quote_number text COLLATE pg_catalog."default",
    quantity numeric(18,2),
    unit_price numeric(18,2),
    unit_of_measue text COLLATE pg_catalog."default",
    currency character varying(3) COLLATE pg_catalog."default",
    line_total numeric(18,2),
    tax_percent smallint,
    tax_amount numeric(18,2),
    total_amount numeric(18,2),
    created_date timestamp without time zone,
    created_by text COLLATE pg_catalog."default",
    last_modified_by text COLLATE pg_catalog."default",
    last_modified_date timestamp without time zone,
```

### `proc.agent_plan`
```
plan_id bigserial PRIMARY KEY,
workflow_id text,
agent_id text,
agent_name text,
action_id text,
plan text NOT NULL,
created_at timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER
```
### `proc.invoice_agent`
```
invoice_id text NOT NULL DEFAULT ('IN'::text || lpad((nextval('proc.inv_sequence_new'::regclass))::text, 6, '0')),
po_id text,
supplier_name text,
buyer_id text,
requisition_id text,
requested_by text,
requested_date date,
invoice_date date,
due_date date,
invoice_paid_date date,
payment_terms text,
currency varchar(3),
invoice_amount numeric(18,2),
tax_percent numeric(5,2),
tax_amount numeric(18,2),
invoice_total_incl_tax numeric(18,2),
exchange_rate_to_usd numeric(10,4),
converted_amount_usd numeric(18,2),
country text,
region text,
invoice_status text,
ai_flag_required text,
trigger_type text,
trigger_context_description text,
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER,
last_modified_by text DEFAULT CURRENT_USER,
last_modified_date timestamp DEFAULT CURRENT_TIMESTAMP,
PRIMARY KEY (invoice_id)
```

### `proc.invoice_line_items_agent`
```
invoice_line_id text DEFAULT ('INL'::text || lpad((nextval('proc.inl_sequence_new'::regclass))::text, 6, '0')),
invoice_id text NOT NULL,
line_no integer NOT NULL,
item_id text,
item_description text,
quantity integer,
unit_of_measure text,
unit_price numeric(10,2),
line_amount numeric(18,2),
tax_percent numeric(5,2),
tax_amount numeric(18,2),
total_amount_incl_tax numeric(18,2),
po_id text,
delivery_date date,
country text,
region text,
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by text DEFAULT CURRENT_USER,
last_modified_by text DEFAULT CURRENT_USER,
last_modified_date timestamp DEFAULT CURRENT_TIMESTAMP,
PRIMARY KEY (invoice_id, line_no)
```

### `proc.cat_product_mapping`
```
category_level_1 varchar(255),
category_level_2 varchar(255),
category_level_3 varchar(255),
category_level_4 varchar(255),
category_level_5 varchar(255),
product varchar(255),
created_date timestamp DEFAULT CURRENT_TIMESTAMP,
created_by varchar(100) DEFAULT CURRENT_USER,
lastmodified_date timestamp DEFAULT CURRENT_TIMESTAMP,
lastmodified_by varchar(100) DEFAULT CURRENT_USER
```
### `proc.quote_agent`
```
quote_id text COLLATE pg_catalog."default" NOT NULL,
    supplier_id text COLLATE pg_catalog."default",
    buyer_id text COLLATE pg_catalog."default",
    quote_date date,
    validity_date date,
    currency character varying(3) COLLATE pg_catalog."default",
    total_amount numeric(18,2),
    tax_percent numeric(5,2),
    tax_amount numeric(18,2),
    total_amount_incl_tax numeric(18,2),
    po_id text COLLATE pg_catalog."default",
    country text COLLATE pg_catalog."default",
    region text COLLATE pg_catalog."default",
    ai_flag_required character varying(5) COLLATE pg_catalog."default",
    trigger_type character varying(30) COLLATE pg_catalog."default",
    trigger_context_description text COLLATE pg_catalog."default",
    created_date timestamp without time zone DEFAULT now(),
    created_by text COLLATE pg_catalog."default",
    last_modified_by text COLLATE pg_catalog."default",
    last_modified_date timestamp without time zone,
    CONSTRAINT quote_agent_pkey PRIMARY KEY (quote_id)
```
### `proc.quote_line_items_agent`
```
quote_line_id text COLLATE pg_catalog."default" NOT NULL,
    quote_id text COLLATE pg_catalog."default" NOT NULL,
    line_number integer NOT NULL,
    item_id text COLLATE pg_catalog."default",
    item_description text COLLATE pg_catalog."default",
    quantity integer,
    unit_of_measure text COLLATE pg_catalog."default",
    unit_price numeric(18,2),
    line_total numeric(18,2),
    tax_percent numeric(5,2),
    tax_amount numeric(18,2),
    total_amount numeric(18,2),
    currency character varying(3) COLLATE pg_catalog."default",
    created_date timestamp without time zone DEFAULT now(),
    created_by text COLLATE pg_catalog."default",
    last_modified_by text COLLATE pg_catalog."default",
    last_modified_date timestamp without time zone,
    CONSTRAINT quote_line_items_agent_pkey PRIMARY KEY (quote_line_id)
```
### `proc.policy`
```
policy_id integer NOT NULL DEFAULT nextval('proc.policy_policy_id_seq'::regclass),
    policy_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    policy_type character varying(100) COLLATE pg_catalog."default",
    policy_linked_agents integer[],
    policy_desc text COLLATE pg_catalog."default",
    policy_details text COLLATE pg_catalog."default",
    policy_status integer DEFAULT 1,
    created_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100) COLLATE pg_catalog."default",
    last_modified_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_modified_by character varying(100) COLLATE pg_catalog."default",
    version text COLLATE pg_catalog."default",
    CONSTRAINT policy_pkey1 PRIMARY KEY (policy_id)
```
### `proc.prompt`
```
prompt_id integer NOT NULL DEFAULT nextval('proc.prompt_prompt_id_seq'::regclass),
    prompt_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    prompt_type character varying(100) COLLATE pg_catalog."default",
    prompt_linked_agents integer[],
    prompts_desc text COLLATE pg_catalog."default",
    prompts_status integer DEFAULT 1,
    created_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100) COLLATE pg_catalog."default",
    last_modified_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_modified_by character varying(100) COLLATE pg_catalog."default",
    version text COLLATE pg_catalog."default",
    CONSTRAINT prompt_pkey PRIMARY KEY (prompt_id)
```

### `proc.agent`
```
agent_id character varying(255) COLLATE pg_catalog."default" NOT NULL,
    user_id character varying(100) COLLATE pg_catalog."default" NOT NULL,
    user_name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    agent_name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    agent_property jsonb NOT NULL DEFAULT '{}'::jsonb,
    agent_type character varying(50) COLLATE pg_catalog."default" NOT NULL,
    created_by character varying(100) COLLATE pg_catalog."default" NOT NULL,
    created_time timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    modified_by character varying(100) COLLATE pg_catalog."default",
    modified_time timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    agent_description text COLLATE pg_catalog."default",
    CONSTRAINT agent_pkey PRIMARY KEY (agent_id)
```
### `proc.routing`
```
process_id integer NOT NULL DEFAULT nextval('proc.process_process_id_seq'::regclass),
    process_name text COLLATE pg_catalog."default" NOT NULL,
    process_details jsonb,
    process_status integer DEFAULT 0,
    created_on timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_by text COLLATE pg_catalog."default",
    modified_on timestamp without time zone,
    modified_by text COLLATE pg_catalog."default",
    user_id text COLLATE pg_catalog."default",
    user_name text COLLATE pg_catalog."default",
    raw_data json,
    CONSTRAINT process_pkey PRIMARY KEY (process_id)
```
### `proc.draft_rfq_emails`
```
id bigint NOT NULL DEFAULT nextval('proc.draft_rfq_emails_id_seq'::regclass),
    rfq_id text COLLATE pg_catalog."default" NOT NULL,
    supplier_id text COLLATE pg_catalog."default",
    subject text COLLATE pg_catalog."default" NOT NULL,
    body text COLLATE pg_catalog."default" NOT NULL,
    created_on timestamp with time zone NOT NULL DEFAULT now(),
    sent boolean NOT NULL DEFAULT false,
    supplier_name text COLLATE pg_catalog."default",
    sent_on timestamp with time zone,
    recipient_email text COLLATE pg_catalog."default",
    contact_level integer NOT NULL DEFAULT 0,
    thread_index integer NOT NULL DEFAULT 1,
    sender text COLLATE pg_catalog."default",
    payload jsonb,
    updated_on timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT draft_rfq_emails_pkey PRIMARY KEY (id)
```

### `proc.workflow_email_tracking`
```
workflow_id text NOT NULL,
    unique_id text NOT NULL,
    supplier_id text,
    supplier_email text,
    message_id text,
    subject text,
    dispatched_at timestamp with time zone NOT NULL,
    responded_at timestamp with time zone,
    response_message_id text,
    matched boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT workflow_email_tracking_pk PRIMARY KEY (workflow_id, unique_id)
```

### `proc.supplier_response`
```
id serial PRIMARY KEY,
    workflow_id text NOT NULL,
    unique_id text NOT NULL,
    supplier_id text,
    supplier_email text,
    response_message_id text,
    response_subject text,
    response_body text,
    response_from text,
    response_date timestamp with time zone,
    original_message_id text,
    original_subject text,
    match_confidence numeric(3,2),
    response_time numeric(18,6),
    created_at timestamp with time zone DEFAULT now(),
    processed boolean DEFAULT false,
    CONSTRAINT supplier_response_unique UNIQUE (workflow_id, unique_id)
```

### `proc.rfq_targets`
```
rfq_id text NOT NULL,
    supplier_id text,
    target_price numeric(18,2),
    negotiation_round integer NOT NULL DEFAULT 1,
    notes text,
    created_on timestamp with time zone NOT NULL DEFAULT now(),
    updated_on timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT rfq_targets_pk PRIMARY KEY (rfq_id, negotiation_round)
```

### `proc.negotiation_sessions`
```
rfq_id text NOT NULL,
    supplier_id text NOT NULL,
    round integer NOT NULL,
    counter_offer numeric(18,2),
    created_on timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT negotiation_sessions_pk PRIMARY KEY (rfq_id, supplier_id, round)
```

### `proc.negotiation_session_state`
```
rfq_id text NOT NULL,
    supplier_id text NOT NULL,
    supplier_reply_count integer NOT NULL DEFAULT 0,
    current_round integer NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'ACTIVE',
    awaiting_response boolean NOT NULL DEFAULT false,
    last_supplier_msg_id text,
    last_agent_msg_id text,
    last_email_sent_at timestamp with time zone,
    base_subject text,
    initial_body text,
    updated_on timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT negotiation_session_state_pk PRIMARY KEY (rfq_id, supplier_id)
```
Tracks supplier reply counts and awaiting-response status for each negotiation session. The
reply counter advances only after an outbound counter email when the agent is awaiting a
response, ensuring each supplier response is monitored until the negotiation finalises.

### `proc.email_dispatch_chains`
```
id serial PRIMARY KEY,
    rfq_id text NOT NULL,
    message_id text NOT NULL UNIQUE,
    thread_index integer NOT NULL DEFAULT 1,
    supplier_id text,
    workflow_ref text,
    recipients jsonb,
    subject text NOT NULL,
    body text NOT NULL,
    dispatch_metadata jsonb,
    awaiting_response boolean NOT NULL DEFAULT true,
    responded_at timestamptz,
    response_message_id text,
    response_metadata jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
```
Records each dispatched email along with its message identifier so replies can be
matched to the originating workflow. When an inbound message references one of the
stored message IDs the row is updated with the response metadata, allowing workflows
to confirm that every dispatched email has received a reply before closing.

### `proc.email_thread_map`
```
message_id text PRIMARY KEY,
    rfq_id text NOT NULL,
    supplier_id text,
    recipients jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
```
Stores the outbound message to RFQ mapping so inbound replies can be
associated without DynamoDB.

### `proc.supplier_replies`
```
rfq_id text NOT NULL,
    message_id text NOT NULL,
    mailbox text,
    subject text,
    from_address text,
    reply_body text,
    s3_key text,
    received_at timestamp with time zone,
    headers jsonb,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    updated_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT supplier_replies_pk PRIMARY KEY (rfq_id, message_id)
```
Stores normalised inbound RFQ replies captured by the SES ingestion Lambda.
