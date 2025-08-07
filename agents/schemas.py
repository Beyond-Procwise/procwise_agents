from pydantic import BaseModel, Field
from typing import List, Optional, Dict

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
    'quote_id': 'the unique quote identifier (often labeled as Quote No, Quote #, or Quote ID)',
    'quote_number': 'the unique quote identifier (often labeled as Quote No, Quote #, or Quote ID)',
    'quote_date': 'the date the quote was issued (e.g., Quote Date)',
    'valid_until': 'the expiration date of the quote (e.g., Valid Until, Expiration Date)',
    'order_date': 'the date the purchase order was created (e.g., Order Date)',
    'description': 'the description of the product or service',
    'item_description': 'the description of the product or service', 'quantity': 'the quantity of the item',
    'unit_price': 'the price for a single unit of the item',
    'line_total': 'the total price for the line item (quantity * unit price)',
    'total_price': 'the total price for the line item (quantity * unit price)',
}

class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = Field(None, alias='unitPrice')
    line_total: Optional[float] = Field(None, alias='lineTotal')

class HeaderData(BaseModel):
    invoice_id: Optional[str] = Field(None, alias='invoiceId')
    po_number: Optional[str] = Field(None, alias='poNumber')
    quote_id: Optional[str] = Field(None, alias='quoteId')
    quote_number: Optional[str] = Field(None, alias='quoteNumber')
    vendor_name: Optional[str] = Field(None, alias='vendorName')
    total_amount: Optional[float] = Field(None, alias='totalAmount')
    invoice_date: Optional[str] = Field(None, alias='invoiceDate')
    quote_date: Optional[str] = Field(None, alias='quoteDate')
    due_date: Optional[str] = Field(None, alias='dueDate')
    valid_until: Optional[str] = Field(None, alias='validUntil')
    order_date: Optional[str] = Field(None, alias='orderDate')
    customer_name: Optional[str] = Field(None, alias='customerName')
    po_id: Optional[str] = Field(None, alias='poId')
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = Field(None, alias='taxAmount')
    tax_percent: Optional[float] = Field(None, alias='taxPercent')
    currency: Optional[str] = None
    expected_delivery_date: Optional[str] = Field(None, alias='expectedDeliveryDate')
    payment_terms: Optional[str] = Field(None, alias='paymentTerms')

class ValidationResult(BaseModel):
    is_valid: bool = Field(..., alias='isValid')
    confidence_score: float = Field(..., alias='confidenceScore', ge=0.0, le=1.0)
    notes: Optional[str] = None

class ExtractedDocument(BaseModel):
    header_data: HeaderData = Field(..., alias='headerData')
    line_items: List[LineItem] = Field([], alias='lineItems')
    validation: ValidationResult
