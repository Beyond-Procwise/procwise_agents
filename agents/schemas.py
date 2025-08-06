import json
import re
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any


SCHEMA_TO_PROMPT_MAP = {
    'invoice_id': 'the unique invoice identifier (often labeled as Invoice No, Invoice #, or Invoice ID)',
    'invoice_number': 'the unique invoice identifier (often labeled as Invoice No, Invoice #, or Invoice ID)',
    'invoice_date': 'the date the invoice was issued (e.g., Invoice Date)',
    'due_date': 'the date the payment is due (e.g., Due Date, Payment Due)',
    'total_amount': 'the final total amount due, including all taxes and fees',
    'subtotal': 'the total amount before taxes are applied',
    'tax_amount': 'the total amount of tax (e.g., VAT, GST, Sales Tax)',
    'vendor_name': 'the name of the company or person who sent the invoice/PO (e.g., From, Vendor)',
    'customer_name': 'the name of the company or person receiving the invoice/PO (e.g., To, Bill To, Customer)',
    'po_id': 'the unique purchase order identifier (often labeled as PO No, PO #, or Purchase Order)',
    'po_number': 'the unique purchase order identifier (often labeled as PO No, PO #, or Purchase Order)',
    'order_date': 'the date the purchase order was created (e.g., Order Date)',
    'description': 'the description of the product or service',
    'item_description': 'the description of the product or service', 'quantity': 'the quantity of the item',
    'unit_price': 'the price for a single unit of the item',
    'line_total': 'the total price for the line item (quantity * unit price)',
    'total_price': 'the total price for the line item (quantity * unit price)',
}


def clean_and_convert_to_float(value: Any) -> Optional[float]:
    """Utility to safely convert a value to float, handling currency symbols and commas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if not value.strip():
            return None
        # Remove currency symbols, commas, and other non-numeric characters except the decimal point
        cleaned_string = re.sub(r'[^\d.]', '', value)
        if cleaned_string:
            try:
                return float(cleaned_string)
            except (ValueError, TypeError):
                return None
    return None


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = Field(None, alias='unitPrice')
    line_total: Optional[float] = Field(None, alias='lineTotal')

    @model_validator(mode='before')
    @classmethod
    def coerce_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Ensure description is always a string
            if 'description' in data:
                data['description'] = str(data.get('description', ''))
            # Clean numeric fields
            for field in ['quantity', 'unitPrice', 'line_total']:
                if field in data:
                    data[field] = clean_and_convert_to_float(data[field])
        return data

    class Config:
        populate_by_name = True


class HeaderData(BaseModel):
    # Canonical fields
    invoice_id: Optional[str] = None
    po_number: Optional[str] = None
    vendor_name: Optional[str] = Field(None, alias='vendorName')
    total_amount: Optional[float] = Field(None, alias='totalAmount')
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = Field(None, alias='taxAmount')
    invoice_date: Optional[str] = Field(None, alias='invoiceDate')
    due_date: Optional[str] = Field(None, alias='dueDate')
    order_date: Optional[str] = Field(None, alias='orderDate')
    payment_terms: Optional[str] = Field(None, alias='paymentTerms')

    @field_validator('total_amount', 'subtotal', 'tax_amount', mode='before')
    @classmethod
    def clean_numeric_fields(cls, v: Any) -> Optional[float]:
        return clean_and_convert_to_float(v)

    @field_validator('payment_terms', mode='before')
    @classmethod
    def flatten_payment_terms(cls, v: Any) -> Optional[str]:
        """If the LLM hallucinates a dict for payment_terms, flatten it to a string."""
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v) if v is not None else None

    @model_validator(mode='before')
    @classmethod
    def consolidate_primary_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict): return data
        # Consolidate invoice ID
        invoice_id_val = data.get('invoice_id') or data.get('invoiceId') or data.get('invoice_number')
        if invoice_id_val: data['invoice_id'] = str(invoice_id_val)
        # Consolidate PO Number
        po_number_val = data.get('po_number') or data.get('poNumber') or data.get('po_id') or data.get('poId')
        if po_number_val: data['po_number'] = str(po_number_val)
        return data

    class Config:
        populate_by_name = True


class ValidationResult(BaseModel):
    is_valid: bool = Field(..., alias='isValid')
    confidence_score: float = Field(..., alias='confidenceScore', ge=0.0, le=1.0)
    notes: Optional[str] = None


class ExtractedDocument(BaseModel):
    header_data: HeaderData = Field(..., alias='headerData')
    line_items: List[LineItem] = Field([], alias='lineItems')
    validation: ValidationResult

    class Config:
        populate_by_name = True