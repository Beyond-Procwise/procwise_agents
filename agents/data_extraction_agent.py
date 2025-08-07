# ProcWise/agents/data_extraction_agent.py
import json
import re
import pandas as pd
import logging
import uuid
from io import BytesIO
from datetime import datetime
from dateutil.parser import parse as parse_date

import ollama
import psycopg2
from qdrant_client import models

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image

from agents.schemas import ExtractedDocument
from agents.base_agent import BaseAgent
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

HITL_CONFIDENCE_THRESHOLD = 0.85


def _normalize_point_id(raw_id: str) -> int | str:
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
            # ... File fetching logic ... (same as before)
            files_to_process = []
            if s3_object_key and s3_object_key.startswith(prefix):
                files_to_process.append({'Key': s3_object_key})
            else:
                response = self.agent_nick.s3_client.list_objects_v2(Bucket=self.settings.s3_bucket_name, Prefix=prefix)
                files_to_process = response.get('Contents', [])

            for file_info in files_to_process:
                object_key = file_info['Key']
                if not any(object_key.lower().endswith(ext) for ext in
                           ['.pdf', '.png', '.jpg', '.jpeg', '.csv', '.xlsx']): continue
                logger.info(f"\n[Processing]: {object_key}")

                try:
                    s3_object = self.agent_nick.s3_client.get_object(Bucket=self.settings.s3_bucket_name,
                                                                     Key=object_key)
                    file_bytes = s3_object['Body'].read()
                except Exception as e:
                    logger.error(f"Failed to download {object_key} from S3: {e}");
                    continue

                document_representation = self._get_enhanced_document_representation(file_bytes, object_key)
                if not document_representation: continue

                header_data = self._extract_header(document_representation)
                if not header_data: continue

                line_items = self._extract_line_items(document_representation)

                final_data = self._reconcile_and_validate(header_data, line_items)
                if not final_data: continue

                try:
                    # With the new schema, this step is now extremely robust.
                    validated_doc = ExtractedDocument.model_validate(final_data)
                except Exception as e:
                    logger.error(
                        f"Pydantic Schema Validation Failed for {object_key}: {e}\nRaw Data was: {final_data}");
                    continue

                pk_value = validated_doc.header_data.invoice_id or validated_doc.header_data.po_number
                if not pk_value:
                    logger.error("Persistence Failed: No primary key found.")
                    continue

                logger.info(f"Successfully identified PK '{pk_value}' for {object_key}.")
                db_data = validated_doc.model_dump()

                # --- ENTERPRISE PIPELINE REORDERING ---
                # 1. Create Vector First
                doc_type = self._determine_doc_type(prefix, document_representation)
                if doc_type not in self.table_schemas: continue

                product_type = self._classify_product_type(db_data.get('line_items', []))
                self._upsert_to_qdrant(db_data, pk_value, doc_type, product_type, object_key)

                # 2. Attempt to save to PostgreSQL
                status = "success"
                if validated_doc.validation.confidence_score < HITL_CONFIDENCE_THRESHOLD: status = "needs_review"
                if not validated_doc.validation.is_valid: status = "failed"

                try:
                    self._insert_data_to_postgres(db_data, pk_value, self.table_schemas[doc_type], status)
                except ValueError as e:
                    logger.error(str(e))
                    status = "db_failed"
                except Exception as e:
                    logger.error(f"DB insertion failed for {object_key}, but vector was saved. Error: {e}")
                    status = "db_failed"

                results.append({"object_key": object_key, "id": pk_value, "status": status})

        return {"status": "completed", "details": results}

    def _determine_doc_type(self, s3_prefix, text_content):
        llm_type = self._classify_document_type(text_content)
        return llm_type.replace(" ", "_") if llm_type else (
            'Invoice' if 'invoice' in s3_prefix.lower() else 'Purchase_Order')

    def _get_enhanced_document_representation(self, file_bytes: bytes, filename: str) -> str:
        # ... this method remains the same ...
        logger.info(f"-> Stage 1: Performing document analysis for '{filename}'...")
        file_ext = filename.lower().split('.')[-1]
        try:
            if file_ext == 'pdf':
                elements = partition_pdf(file=BytesIO(file_bytes), strategy="hi_res", infer_table_structure=True)
            elif file_ext in ['png', 'jpg', 'jpeg']:
                elements = partition_image(file=BytesIO(file_bytes), strategy="hi_res", infer_table_structure=True)
            elif file_ext == 'csv':
                return f"Content from CSV file '{filename}':\n{pd.read_csv(BytesIO(file_bytes)).to_string()}"
            elif file_ext == 'xlsx':
                xls = pd.ExcelFile(BytesIO(file_bytes))
                return "\n\n".join([
                                       f"Content from Excel sheet '{sheet_name}':\n{pd.read_excel(xls, sheet_name=sheet_name).to_string()}"
                                       for sheet_name in xls.sheet_names])
            else:
                return ""
            return "\n\n".join(
                [f"### TABLE:\n{el.metadata.text_as_html}\n" if "Table" in str(type(el)) else el.text for el in
                 elements])
        except Exception as e:
            logger.error(f"Advanced processing failed for {filename}: {e}.")
            return ""

    def _extract_header(self, text: str) -> Dict | None:
        # ... this method remains the same ...
        logger.info(" -> Stage 2a: Extracting Header Data...")
        prompt = "Extract header info (invoiceId, poNumber, vendorName, totalAmount, etc.) from the text. Respond with a single JSON object."
        try:
            response = ollama.generate(model=self.extraction_model,
                                       prompt=f"{prompt}\n\nDocument Content:\n---\n{text}\n---\nJSON:", format='json')
            return json.loads(response.get('response'))
        except Exception as e:
            logger.error(f"Header extraction failed: {e}");
            return None

    def _extract_line_items(self, text: str) -> List[Dict]:
        # ... this method remains the same ...
        logger.info(" -> Stage 2b: Extracting Line Items...")
        prompt = "Extract line items (description, quantity, etc.) from the text. Respond with a JSON object with a single \"lineItems\" key."
        try:
            response = ollama.generate(model=self.extraction_model,
                                       prompt=f"{prompt}\n\nDocument Content:\n---\n{text}\n---\nJSON:", format='json')
            return json.loads(response.get('response', '{"lineItems": []}')).get('lineItems')
        except Exception as e:
            logger.error(f"Line item extraction failed: {e}");
            return []

    def _reconcile_and_validate(self, header: Dict, lines: List[Dict]) -> Dict | None:
        """
        --- THIS IS THE FIX for the validation.isValid error ---
        This method now asks the LLM for simpler outputs and constructs the final, valid JSON in Python.
        """
        logger.info(" -> Stage 3: Cleaning, Reconciling, and Validating...")
        prompt = f"""You are a data validation expert. Review the data below.
1. Sum all line item totals. If the sum differs from the header's totalAmount, correct the header's totalAmount.
2. Provide a `confidenceScore` (float from 0.0 to 1.0) for the extraction.
3. Provide brief `validation_notes` (string) about the data quality or any corrections made.
4. Your response MUST be a single JSON object containing these keys: `corrected_header`, `corrected_line_items`, `confidence_score`, `validation_notes`.

**Data to Validate:**
- Header: {json.dumps(header)}
- Line Items: {json.dumps(lines)}

**JSON OUTPUT:**"""
        try:
            response = ollama.generate(model=self.extraction_model, prompt=prompt, format='json',
                                       options={'temperature': 0.0})
            llm_output = json.loads(response.get('response'))

            # Deterministically build the final, valid object in Python
            notes = llm_output.get('validation_notes', '')
            is_valid_flag = "fail" not in notes.lower() and "error" not in notes.lower()

            final_data = {
                "headerData": llm_output.get('corrected_header', header),
                "lineItems": llm_output.get('corrected_line_items', lines),
                "validation": {
                    "isValid": is_valid_flag,
                    "confidenceScore": llm_output.get('confidence_score', 0.0),
                    "notes": notes
                }
            }
            return final_data
        except Exception as e:
            logger.error(f"Reconciliation and validation step failed: {e}");
            return None

    def _classify_document_type(self, text: str) -> str | None:
        # ... this method remains the same ...
        prompt = f"""Is the text an "Invoice" or a "Purchase Order"? Respond with JSON: {{"document_type": "value"}}"""
        try:
            response = ollama.generate(model=self.extraction_model, prompt=f"{prompt}\n\nText:\n{text[:1000]}",
                                       format='json')
            return json.loads(response.get('response', '{}')).get('document_type')
        except Exception:
            return None

    def _classify_product_type(self, line_items: list) -> str:
        # ... this method remains the same ...
        if not line_items: return "Uncategorized"
        descriptions = "\n- ".join([str(item.get('description', '')) for item in line_items if item.get('description')])
        if not descriptions: return "Uncategorized"
        prompt = f"""What is the best category for these items? (e.g., "IT Hardware"). Items:\n- {descriptions}\nRespond with JSON: {{"product_category": "Your Category"}}"""
        try:
            response = ollama.generate(model=self.extraction_model, prompt=prompt, format='json')
            return json.loads(response.get('response', '{}')).get('product_category', 'General Goods')
        except Exception as e:
            logger.error(f"Product classification failed: {e}");
            return "General Goods"

    def _upsert_to_qdrant(self, data: Dict, pk_value: str, doc_type: str, product_type: str, object_key: str):
        # ... this method remains the same ...
        try:
            point_id = _normalize_point_id(pk_value)
            summary = self._generate_document_summary(data, pk_value, doc_type)
            logger.info(f"Generating vector for summary: '{summary}'")
            vector = self.agent_nick.embedding_model.encode(summary).tolist()
            payload = {"document_type": doc_type, "product_type": product_type, "s3_object_key": object_key,
                       "summary": summary, "confidence": data.get('validation', {}).get('confidence_score')}
            self.agent_nick.qdrant_client.upsert(collection_name=self.settings.qdrant_collection_name, points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)], wait=True)
            logger.info(f"SUCCESS (Qdrant): Vector for {pk_value} (ID: {point_id}) was created/updated.")
        except Exception as e:
            logger.error(f"ERROR (Qdrant): Failed to upsert vector for {pk_value}. Reason: {e}")

    def _generate_document_summary(self, data: Dict, pk_value: str, doc_type: str) -> str:
        # ... this method remains the same ...
        header = data.get('header_data', {})
        return f"{doc_type} {pk_value} from {header.get('vendor_name')} for ${header.get('total_amount')} on {header.get('invoice_date')}."

    # All database-related methods (_fetch_all_table_schemas, _get_table_columns,
    # _clean_and_standardize_data_for_db, _insert_data_to_postgres) from the previous answer
    # are still valid and can be copied here without change.

    def _fetch_all_table_schemas(self):
        # This method is unchanged
        logger.info("Fetching database schemas...")
        schemas = {'Invoice': {'pk_col': 'invoice_id'}, 'Purchase_Order': {'pk_col': 'po_id'}}
        with self.agent_nick.get_db_connection() as conn:
            for doc_type, config in schemas.items():
                table_prefix = doc_type.lower()
                header_table_name = f"{table_prefix}_agent"
                line_table_name = f"{table_prefix}_line_items_agent" if doc_type == "Invoice" else "po_line_items_agent"
                config['header_table'] = f"proc.{header_table_name}"
                config['line_table'] = f"proc.{line_table_name}"
                config['header_cols'] = self._get_table_columns(conn, 'proc', header_table_name)
                config['line_cols'] = self._get_table_columns(conn, 'proc', line_table_name)
                status_col = next((c for c in config['header_cols'] if c.endswith('_status')), 'status')
                config['status_col'] = status_col
        return schemas

    def _get_table_columns(self, db_conn, table_schema, table_name):
        # This method is unchanged
        with db_conn.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;",
                    (table_schema, table_name))
                return [row[0] for row in cursor.fetchall()]
            except psycopg2.Error:
                return []

    def _clean_and_standardize_data_for_db(self, data_dict: dict, db_cols: list) -> dict:
        # This method is unchanged
        cleaned_data = {}
        PYDANTIC_TO_DB_MAP = {
            'invoice_id': 'invoice_id', 'po_number': 'po_id', 'vendor_name': 'vendor_name',
            'total_amount': 'total_amount', 'subtotal': 'subtotal', 'tax_amount': 'tax_amount',
            'invoice_date': 'invoice_date', 'due_date': 'due_date', 'order_date': 'order_date',
            'payment_terms': 'payment_terms', 'description': 'description', 'quantity': 'quantity',
            'unit_price': 'unit_price', 'line_total': 'line_total'
        }
        for db_col in db_cols:
            pydantic_field = next((k for k, v in PYDANTIC_TO_DB_MAP.items() if v == db_col), None)
            if not pydantic_field: continue
            value = data_dict.get(pydantic_field)
            if value is None or (isinstance(value, str) and not value.strip()): cleaned_data[db_col] = None; continue
            if 'date' in db_col and isinstance(value, str):
                try:
                    cleaned_data[db_col] = parse_date(value).strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    cleaned_data[db_col] = None
                continue
            if any(t in db_col for t in ['amount', 'price', 'quantity', 'total', 'subtotal', 'tax']):
                if isinstance(value, (int, float)):
                    cleaned_data[db_col] = value
                else:
                    cleaned_data[db_col] = None
                continue
            cleaned_data[db_col] = str(value)
        return cleaned_data

    def _insert_data_to_postgres(self, data: Dict, pk_value: str, config: Dict, status: str):
        # This method is unchanged
        pk_col_db = config.get('pk_col')
        if not pk_col_db or not pk_value:
            raise ValueError("No primary key found for PostgreSQL insertion.")
        header_pydantic_data = data.get('header_data', {})
        cleaned_header = self._clean_and_standardize_data_for_db(header_pydantic_data, config['header_cols'])
        cleaned_header[pk_col_db] = pk_value
        status_column_name = config.get('status_col', 'status')
        current_time = datetime.now()
        cleaned_header.update({'last_modified_date': current_time, 'last_modified_by': self.settings.script_user,
                               'created_date': current_time, 'created_by': self.settings.script_user,
                               status_column_name: status})
        with self.agent_nick.get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cols_to_insert = {k: v for k, v in cleaned_header.items() if k in config['header_cols']}
                update_parts = [f"{col} = EXCLUDED.{col}" for col in cols_to_insert if
                                col not in self.settings.audit_columns and col != pk_col_db]
                update_clause = ", ".join(
                    update_parts + [f"last_modified_date = NOW()", f"last_modified_by = '{self.settings.script_user}'"])
                cols_str, vals_ph = ', '.join(cols_to_insert.keys()), ', '.join(['%s'] * len(cols_to_insert))
                sql_upsert = f"INSERT INTO {config['header_table']} ({cols_str}) VALUES ({vals_ph}) ON CONFLICT ({pk_col_db}) DO UPDATE SET {update_clause};"
                cursor.execute(sql_upsert, tuple(cols_to_insert.values()))
                cursor.execute(f"DELETE FROM {config['line_table']} WHERE {pk_col_db} = %s;", (pk_value,))
                for i, item in enumerate(data.get('line_items', []), 1):
                    cleaned_item = self._clean_and_standardize_data_for_db(item, config['line_cols'])
                    line_pk_col = 'invoice_line_id' if 'invoice' in config['line_table'] else 'po_line_id'
                    cleaned_item.update({pk_col_db: pk_value, 'line_no': i, line_pk_col: f"{pk_value}-{i}"})
                    item_to_insert = {k: v for k, v in cleaned_item.items() if k in config['line_cols']}
                    if not item_to_insert.get(line_pk_col): continue
                    line_cols, line_vals = ', '.join(item_to_insert.keys()), ', '.join(['%s'] * len(item_to_insert))
                    cursor.execute(f"INSERT INTO {config['line_table']} ({line_cols}) VALUES ({line_vals});",
                                   tuple(item_to_insert.values()))
                conn.commit()
                logger.info(f"SUCCESS (PostgreSQL): Data for {pk_col_db} {pk_value} saved.")
            except psycopg2.Error as e:
                logger.error(
                    f"ERROR (PostgreSQL): Database transaction failed for {pk_value}. Rolling back. Error: {e}")
                conn.rollback()
                raise


if __name__ == "__main__":

    from agents.base_agent import AgentNick
    agent_nick = AgentNick()
    de = DataExtractionAgent(agent_nick)
    de.run('Purchase_Order/')

