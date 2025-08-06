# ProcWise/agents/data_extraction_agent.py

import json
import re
import logging
import uuid
from io import BytesIO
from datetime import datetime
from dateutil.parser import parse as parse_date
import pdfplumber
import ollama
import psycopg2
from qdrant_client import models

# --- Imports for the advanced pipeline ---
from unstructured.partition.pdf import partition_pdf
from agents.schemas import SCHEMA_TO_PROMPT_MAP, ExtractedDocument, LineItem
from agents.base_agent import BaseAgent
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _normalize_point_id(raw_id: str) -> int | str:
    """Ensures a given string ID is compliant with Qdrant's requirements."""
    if not isinstance(raw_id, str): raw_id = str(raw_id)
    if raw_id.isdigit(): return int(raw_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


class DataExtractionAgent(BaseAgent):
    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.extraction_model = self.settings.extraction_model
        self.table_schemas = self._fetch_all_table_schemas()

    def run(self, s3_prefix: str = None, s3_object_key: str = None) -> Dict:
        logger.info("--- High-Accuracy Data Extraction Run Starting ---")
        s3_prefixes = [s3_prefix] if s3_prefix else self.settings.s3_prefixes
        results = []

        for prefix in s3_prefixes:
            files_to_process = []
            if s3_object_key and s3_object_key.startswith(prefix):
                files_to_process.append({'Key': s3_object_key})
            else:
                response = self.agent_nick.s3_client.list_objects_v2(Bucket=self.settings.s3_bucket_name,
                                                                     Prefix=prefix)
                files_to_process = response.get('Contents', [])

            for file_info in files_to_process:
                object_key = file_info['Key']
                if not object_key.lower().endswith('.pdf'): continue

                logger.info(f"\n[Processing]: {object_key}")

                try:
                    s3_object = self.agent_nick.s3_client.get_object(Bucket=self.settings.s3_bucket_name,
                                                                     Key=object_key)
                    pdf_bytes = s3_object['Body'].read()
                except Exception as e:
                    logger.error(f"Failed to download {object_key} from S3: {e}");
                    continue

                document_representation = self._get_enhanced_document_representation(pdf_bytes, object_key)
                header_data = self._extract_header(document_representation)
                if not header_data:
                    results.append(
                        {"object_key": object_key, "status": "failed", "reason": "Header extraction failed."});
                    continue

                line_items = self._extract_line_items(document_representation)
                final_data = self._reconcile_and_validate(header_data, line_items)

                try:
                    validated_doc = ExtractedDocument.model_validate(final_data)
                    if not validated_doc.validation.is_valid:
                        logger.warning(f"Validation Failed by LLM Auditor: {validated_doc.validation.notes}");
                        continue
                except Exception as e:
                    logger.error(f"Pydantic Schema Validation Failed: {e}");
                    continue

                logger.info(f"Extraction Successful! Confidence: {validated_doc.validation.confidence_score:.2f}")

                # This is where the error occurred. validated_doc.line_items is a List[LineItem]
                product_type = self._classify_product_type(validated_doc.line_items)

                doc_type_from_llm = self._classify_document_type(document_representation)
                doc_type = doc_type_from_llm.replace(" ", "_") if doc_type_from_llm else (
                    'Invoice' if 'invoice' in prefix.lower() else 'Purchase_Order')

                if doc_type not in self.table_schemas:
                    logger.error(f"Invalid doc_type '{doc_type}' generated. Skipping.");
                    continue

                pk_value = validated_doc.header_data.invoice_id or validated_doc.header_data.po_number

                if not pk_value:
                    logger.error("Persistence Failed: No primary key found.");
                    continue

                db_data = validated_doc.model_dump(by_alias=True)

                self._upsert_to_qdrant(db_data, pk_value, doc_type, product_type, object_key)
                self._insert_data_to_postgres(db_data, self.table_schemas[doc_type])
                results.append({"object_key": object_key, "id": pk_value, "status": "success"})

        return {"status": "completed", "details": results}

    def _get_enhanced_document_representation(self, pdf_bytes: bytes, filename: str) -> str:
        logger.info("  -> Stage 1: Performing layout-aware document analysis...")
        try:
            elements = partition_pdf(file=BytesIO(pdf_bytes), strategy="hi_res", infer_table_structure=True)
            return "\n\n".join(
                [f"### TABLE:\n{el.metadata.text_as_html}\n" if "Table" in str(type(el)) else el.text for el in
                 elements])
        except Exception as e:
            logger.warning(f"Advanced PDF processing failed: {e}. Falling back to simple text extraction.")
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                return "".join(page.extract_text() for page in pdf.pages if page.extract_text())

    def _extract_header(self, text: str) -> Dict | None:
        logger.info("  -> Stage 2a: Extracting Header Data...")
        prompt = "Extract header info (invoiceId, poNumber, vendorName, totalAmount, subtotal, taxAmount, invoiceDate, dueDate) from the text. Respond with a single JSON object."
        try:
            response = ollama.generate(model=self.extraction_model,
                                       prompt=f"{prompt}\n\nDocument Content:\n---\n{text}\n---\nJSON:", format='json')
            return json.loads(response.get('response'))
        except Exception as e:
            logger.error(f"Header extraction failed: {e}");
            return None

    def _extract_line_items(self, text: str) -> List[Dict]:
        logger.info("  -> Stage 2b: Extracting Line Items...")
        prompt = "Extract line items (description, quantity, unitPrice, lineTotal) from the text. Respond with a JSON object with a single \"lineItems\" key."
        try:
            response = ollama.generate(model=self.extraction_model,
                                       prompt=f"{prompt}\n\nDocument Content:\n---\n{text}\n---\nJSON:", format='json')
            return json.loads(response.get('response', '{"lineItems": []}')).get('lineItems')
        except Exception as e:
            logger.error(f"Line item extraction failed: {e}");
            return []

    # --- THIS IS THE DEFINITIVE, ROBUST CLEANING AND COERCION FUNCTION ---

    def _coerce_and_clean_data(self, data: Any) -> Any:
        """
        Recursively cleans and coerces data to prevent Pydantic validation errors.
        """
        if isinstance(data, dict):
            cleaned_dict = {}
            STRING_FIELDS = {'invoiceId', 'poNumber', 'vendorName', 'customerName', 'invoiceDate', 'dueDate',
                             'description'}
            NUMERIC_FIELDS = {'quantity', 'unitPrice', 'lineTotal', 'totalAmount', 'subtotal', 'taxAmount'}

            for key, value in data.items():
                if value is None:
                    cleaned_dict[key] = None
                    continue

                if key in STRING_FIELDS:
                    cleaned_dict[key] = str(value)
                elif key in NUMERIC_FIELDS:
                    if isinstance(value, str) and value.strip() == '':
                        cleaned_dict[key] = None  # Set empty string to None
                    else:
                        try:
                            cleaned_dict[key] = float(value)
                        except (ValueError, TypeError):
                            cleaned_dict[key] = None
                elif isinstance(value, (dict, list)):
                    cleaned_dict[key] = self._coerce_and_clean_data(value)
                elif isinstance(value, str) and re.search(r'[£$€,]', value):
                    cleaned_string = re.sub(r'[^\d.]', '', value)
                    try:
                        cleaned_dict[key] = float(cleaned_string) if cleaned_string else None
                    except (ValueError, TypeError):
                        cleaned_dict[key] = None
                else:
                    cleaned_dict[key] = value

            return cleaned_dict

        if isinstance(data, list):
            return [self._coerce_and_clean_data(item) for item in data]

        return data

    def _reconcile_and_validate(self, header: Dict, lines: List[Dict]) -> Dict:
        logger.info("  -> Stage 3: Cleaning, Reconciling, and Validating...")

        # Apply the comprehensive cleaning function before validation
        cleaned_header = self._coerce_and_clean_data(header)
        cleaned_lines = self._coerce_and_clean_data(lines)

        prompt = f"""You are a data validation expert. Review the cleaned data below.
1. Sum all `lineTotal` values.
2. If the sum differs from `totalAmount`, correct the header's `totalAmount`.
3. Assess if data is valid and provide a `confidenceScore` (0.0 to 1.0).
4. Your response MUST be a single JSON object with `headerData`, `lineItems`, and `validation` keys. The `validation` object MUST contain `isValid` (boolean), `confidenceScore` (float), and `notes` (string).

Cleaned Data:\n- Header: {json.dumps(cleaned_header)}\n- Line Items: {json.dumps(cleaned_lines)}\nFinal Validated JSON Output:"""
        try:
            response = ollama.generate(model=self.extraction_model, prompt=prompt, format='json',
                                       options={'temperature': 0.0})
            return json.loads(response.get('response'))
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            return {"headerData": cleaned_header, "lineItems": cleaned_lines,
                    "validation": {"isValid": False, "confidenceScore": 0.0,
                                   "notes": "Catastrophic failure in validation LLM call."}}

    def _classify_document_type(self, text: str) -> str | None:
        prompt = f"""Is the text an "Invoice" or a "Purchase Order"? Respond with JSON: {{"document_type": "value"}}"""
        try:
            response = ollama.generate(model=self.extraction_model, prompt=f"{prompt}\n\nText:\n{text[:1000]}",
                                       format='json')
            return json.loads(response.get('response', '{}')).get('document_type')
        except Exception:
            return None

    def _classify_product_type(self, line_items: List[LineItem]) -> str:
        """
        Generically classifies the product type based on a list of Pydantic LineItem objects.
        """
        if not line_items: return "Uncategorized"

        # Use dot notation (.description) to access attributes on Pydantic objects
        descriptions = "\n- ".join([item.description for item in line_items if item.description])

        if not descriptions: return "Uncategorized"

        prompt = f"""What is the single best high-level category for these items?
Examples: "IT Hardware", "Office Supplies", "Software Licenses", "Marketing Services".
Items:
- {descriptions}
Respond with a single JSON object: {{"product_category": "Your Category"}}"""

        try:
            response = ollama.generate(model=self.extraction_model, prompt=prompt, format='json')
            return json.loads(response.get('response', '{}')).get('product_category', 'General Goods')
        except Exception as e:
            logger.error(f"Generic product classification failed: {e}")
            return "General Goods"

    def _upsert_to_qdrant(self, data: Dict, pk_value: str, doc_type: str, product_type: str, object_key: str):
        point_id = _normalize_point_id(pk_value)
        summary = self._generate_document_summary(data, pk_value, doc_type)
        vector = self.agent_nick.embedding_model.encode(summary).tolist()
        payload = {"document_type": doc_type, "product_type": product_type, "s3_object_key": object_key,
                   "summary": summary, "confidence": data.get('validation', {}).get('confidenceScore')}
        self.agent_nick.qdrant_client.upsert(collection_name=self.settings.qdrant_collection_name,
                                             points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
                                             wait=True)

    def _generate_document_summary(self, data: Dict, pk_value: str, doc_type: str) -> str:
        header = data.get('headerData', {})
        return f"{doc_type} {pk_value} from {header.get('vendorName')} for ${header.get('totalAmount')} on {header.get('invoiceDate')}."

    def _fetch_all_table_schemas(self):
        logger.info("Fetching database schemas...")
        schemas = {'Invoice': {'pk_col': 'invoice_id'}, 'Purchase_Order': {'pk_col': 'po_id'}}
        with self.agent_nick.get_db_connection() as conn:
            schemas['Invoice']['header_table'], schemas['Invoice'][
                'line_table'] = "proc.invoice_agent", "proc.invoice_line_items_agent"
            schemas['Invoice']['header_cols'] = self._get_table_columns(conn, 'proc', 'invoice_agent')
            schemas['Invoice']['line_cols'] = self._get_table_columns(conn, 'proc', 'invoice_line_items_agent')
            schemas['Purchase_Order']['header_table'], schemas['Purchase_Order'][
                'line_table'] = "proc.purchase_order_agent", "proc.po_line_items_agent"
            schemas['Purchase_Order']['header_cols'] = self._get_table_columns(conn, 'proc', 'purchase_order_agent')
            schemas['Purchase_Order']['line_cols'] = self._get_table_columns(conn, 'proc', 'po_line_items_agent')
        return schemas

    def _get_table_columns(self, db_conn, table_schema, table_name):
        with db_conn.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;",
                    (table_schema, table_name))
                return [row[0] for row in cursor.fetchall()]
            except psycopg2.Error:
                return []

    def _clean_and_standardize_data_for_db(self, data_dict):
        cleaned = data_dict.copy()
        integer_fields = ['quantity', 'line_no']
        float_fields = ['amount', 'price', 'total', 'subtotal', 'tax_amount', 'line_total', 'unit_price']
        for key, value in cleaned.items():
            if value is None or value == '': continue
            if "date" in key and isinstance(value, str):
                try:
                    cleaned[key] = parse_date(value).strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    cleaned[key] = None
            elif any(field_key in key for field_key in integer_fields):
                try:
                    cleaned[key] = int(float(re.sub(r'[^\d.]', '', str(value))))
                except (ValueError, TypeError):
                    cleaned[key] = None
            elif any(field_key in key for field_key in float_fields):
                try:
                    cleaned[key] = float(re.sub(r'[^\d.]', '', str(value)))
                except (ValueError, TypeError):
                    cleaned[key] = None
        return cleaned

    def _insert_data_to_postgres(self, data: Dict, config: Dict):
        header_pydantic = data.get('headerData', {})
        header_data = {'invoice_id': header_pydantic.get('invoiceId'), 'po_id': header_pydantic.get('poNumber'),
                       'vendor_name': header_pydantic.get('vendorName'),
                       'total_amount': header_pydantic.get('totalAmount'), 'subtotal': header_pydantic.get('subtotal'),
                       'tax_amount': header_pydantic.get('taxAmount'),
                       'invoice_date': header_pydantic.get('invoiceDate'), 'due_date': header_pydantic.get('dueDate'),
                       'order_date': header_pydantic.get('invoiceDate')}
        line_items = [{'description': str(li.get('description')), 'quantity': li.get('quantity'),
                       'unit_price': li.get('unitPrice'), 'line_total': li.get('lineTotal')} for li in
                      data.get('lineItems', [])]

        pk_value = header_data.get(config['pk_col'])
        if not pk_value: logger.error("No primary key found for PostgreSQL insertion."); return

        cleaned_header = self._clean_and_standardize_data_for_db(header_data)
        current_time = datetime.now()
        cleaned_header.update({'last_modified_date': current_time, 'last_modified_by': self.settings.script_user,
                               'created_date': current_time, 'created_by': self.settings.script_user})

        with self.agent_nick.get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cols_to_insert = {k: v for k, v in cleaned_header.items() if
                                  v is not None and k in config['header_cols']}
                update_parts = [f"{col} = EXCLUDED.{col}" for col in cols_to_insert if
                                col not in self.settings.audit_columns and col != config['pk_col']]
                update_clause = ", ".join(
                    update_parts + [f"last_modified_date = NOW()", f"last_modified_by = '{self.settings.script_user}'"])
                cols_str, vals_ph = ', '.join(cols_to_insert.keys()), ', '.join(['%s'] * len(cols_to_insert))
                sql_upsert = f"INSERT INTO {config['header_table']} ({cols_str}) VALUES ({vals_ph}) ON CONFLICT ({config['pk_col']}) DO UPDATE SET {update_clause};"
                cursor.execute(sql_upsert, tuple(cols_to_insert.values()))

                cursor.execute(f"DELETE FROM {config['line_table']} WHERE {config['pk_col']} = %s;", (pk_value,))
                for i, item in enumerate(line_items, 1):
                    cleaned_item = self._clean_and_standardize_data_for_db(item)
                    line_pk_col = 'invoice_line_id' if 'invoice' in config['line_table'] else 'po_line_id'
                    cleaned_item.update({config['pk_col']: pk_value, 'line_no': i, line_pk_col: f"{pk_value}-{i}",
                                         'created_date': current_time, 'created_by': self.settings.script_user,
                                         'last_modified_date': current_time,
                                         'last_modified_by': self.settings.script_user})
                    item_to_insert = {k: v for k, v in cleaned_item.items() if
                                      v is not None and k in config['line_cols']}
                    if not item_to_insert.get(line_pk_col): continue

                    line_cols, line_vals = ', '.join(item_to_insert.keys()), ', '.join(['%s'] * len(item_to_insert))
                    cursor.execute(f"INSERT INTO {config['line_table']} ({line_cols}) VALUES ({line_vals});",
                                   tuple(item_to_insert.values()))

                conn.commit()
                logger.info(f"SUCCESS (PostgreSQL): Data for {config['pk_col']} {pk_value} saved.")
            except psycopg2.Error as e:
                logger.error(f"ERROR (PostgreSQL): Database transaction failed for {pk_value}. {e}")
                conn.rollback()


if __name__ == "__main__":

    from agents.base_agent import AgentNick
    agent_nick = AgentNick()
    de = DataExtractionAgent(agent_nick)
    de.run('Purchase_Order/')

