# Procurement Table Reference

This document captures the canonical structure of key procurement tables
used by agents in the ProcWise framework. It also outlines the
end-to-end data flow combining the vector database and PostgreSQL
sources.

## Data Flow Overview
1. **Contracts ➜ Suppliers**: `proc.contracts.supplier_id` joins to
   `proc.supplier.supplier_id` to fetch each `supplier_name`.
2. **Suppliers ➜ Purchase Orders**: The retrieved `supplier_name` is
   matched against `proc.purchase_order_agent.supplier_id` to gather
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
supplier_id text,
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

### `proc.invoice_agent`
```
invoice_id text NOT NULL DEFAULT ('IN'::text || lpad((nextval('proc.inv_sequence_new'::regclass))::text, 6, '0')),
po_id text,
supplier_id text,
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
