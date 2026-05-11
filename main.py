"""
Invoice Extraction Agent — CLI entry point.

Usage:
    python main.py --invoice_path data/invoices/invoice_1001.txt
    python main.py --invoice_path inv1.txt inv2.txt
"""

import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from utils.models import InvoiceData, ApprovalDecision
from agents.graph import agent


def print_result(
    file_path: str,
    data: InvoiceData,
    inventory_flags: list[str],
    approval: Optional[ApprovalDecision],
) -> None:
    name = Path(file_path).name
    amount = data.total_amount

    print(f"\n{'='*52}")
    print(f"File:    {name}")
    print(f"Vendor:  {data.vendor or 'UNKNOWN'}")
    print(f"Amount:  ${amount:,.2f}" if amount is not None else "Amount:  NOT FOUND")
    print(f"Due:     {data.due_date or 'NOT SPECIFIED'}")
    print(f"\nItems ({len(data.items)}):")
    for item in data.items:
        qty_str = str(item.quantity) if item.quantity is not None else "?"
        price_str = f"${item.unit_price:,.2f}" if item.unit_price is not None else "?"
        print(f"  - {item.name:<22} qty: {qty_str:>6}  @ {price_str}")
    if data.flags:
        print(f"\n⚠  Extraction Flags:")
        for flag in data.flags:
            print(f"   • {flag}")
    if inventory_flags:
        print(f"\n🚫 Inventory Flags:")
        for flag in inventory_flags:
            print(f"   • {flag}")
    if approval:
        status = "✅ APPROVED" if approval.approved else "❌ REJECTED"
        print(f"\n{status}")
        print(f"   Reasoning: {approval.reasoning}")
    print(f"{'='*52}")


if __name__ == "__main__":
    if not os.environ.get("XAI_API_KEY"):
        sys.exit("Error: XAI_API_KEY environment variable is not set.")

    parser = argparse.ArgumentParser(description="Extract and process invoice files.")
    parser.add_argument("--invoice_path", nargs="+", required=True, help="Path(s) to invoice file(s) to process")
    args = parser.parse_args()

    for invoice_path in args.invoice_path:
        try:
            stem = Path(invoice_path).stem
            thread_id = stem
            config = {"configurable": {"thread_id": thread_id}}

            final_state = agent.invoke(
                {
                    "thread_id": thread_id,
                    "file_path": invoice_path,
                    "invoice_text": "",
                    "extracted": None,
                    "inventory_flags": [],
                    "suspicious_items": [],
                    "amount_due": None,
                    "approval": None,
                    "retry_count": 0,
                    "retry_feedback": "",
                    "saved": False,
                },
                config=config,
            )
            if final_state:
                print_result(invoice_path, final_state["extracted"], final_state["inventory_flags"], final_state.get("approval"))
            else:
                print(f"  ⏸  {Path(invoice_path).name} is paused — awaiting human review in the UI.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error processing {invoice_path}: {e}")
