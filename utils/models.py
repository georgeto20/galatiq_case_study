from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class LineItem(BaseModel):
    name: str = Field(description="Item or product name")
    quantity: Optional[float] = Field(description="Quantity ordered; null if missing")
    unit_price: Optional[float] = Field(description="Price per unit; null if missing")


class InvoiceData(BaseModel):
    vendor: Optional[str] = Field(description="Vendor or supplier name")
    total_amount: Optional[float] = Field(description="Total invoice amount as a plain number")
    due_date: Optional[str] = Field(description="Due date in YYYY-MM-DD if possible, else as written")
    items: list[LineItem] = Field(description="All line items on the invoice")
    flags: list[str] = Field(description="Anomalies: missing fields, negative qty, suspicious language, etc.")
    invoice_number: Optional[str] = Field(
        description="Invoice number or ID as stated in the document itself (e.g. 'INV-1004'); null if absent"
    )
    replaces_invoice_id: Optional[str] = Field(
        description="Invoice ID that this document amends, revises, or supersedes, if explicitly stated; null otherwise"
    )


class ApprovalDecision(BaseModel):
    approved: bool = Field(description="True if the invoice is approved for payment, False if rejected")
    requires_human_review: bool = Field(description="True if uncertain or if the invoice is high-value (>$10K) and needs a human to confirm")
    reasoning: str = Field(description="Step-by-step reasoning behind the decision")


class InvoiceState(TypedDict):
    thread_id: str
    file_path: str
    invoice_text: str
    extracted: Optional[InvoiceData]
    inventory_flags: list[str]
    suspicious_items: list[str]
    amount_due: Optional[float]
    approval: Optional[ApprovalDecision]
    retry_count: int
    retry_feedback: str
    saved: bool
