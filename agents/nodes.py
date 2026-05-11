import difflib
import os
import uuid
from pathlib import Path
from typing import Optional

def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value:
            os.environ.setdefault(key.strip(), value)

_load_env()

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from utils.models import LineItem, InvoiceData, ApprovalDecision, InvoiceState
from utils.db import query_inventory, get_inventory_items, lookup_vendor, get_paid_amount, was_human_reviewed, save_pending_review, record_review_decision, save_invoice, check_duplicate_invoice

# ---------------------------------------------------------------------------
# LLM — xAI Grok via OpenAI-compatible endpoint (lazy init)
# ---------------------------------------------------------------------------

_extractor_llm = None
_approval_llm = None


def _get_llm():
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY environment variable is not set")
    return ChatOpenAI(model="grok-3", api_key=api_key, base_url="https://api.x.ai/v1")


def extractor_llm():
    global _extractor_llm
    if _extractor_llm is None:
        _extractor_llm = _get_llm().with_structured_output(InvoiceData)
    return _extractor_llm


def approval_llm():
    global _approval_llm
    if _approval_llm is None:
        _approval_llm = _get_llm().with_structured_output(ApprovalDecision)
    return _approval_llm


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    import pymupdf
    doc = pymupdf.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def _load_json(path: Path) -> str:
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2)


def _load_csv(path: Path) -> str:
    import csv
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    col_widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(max(len(r) for r in rows))]
    lines = ["  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join(lines)


def _load_xml(path: Path) -> str:
    from xml.etree import ElementTree as ET

    def _element_to_text(el: ET.Element, indent: int = 0) -> str:
        prefix = "  " * indent
        text = (el.text or "").strip()
        lines = [f"{prefix}{el.tag}: {text}" if text else f"{prefix}{el.tag}:"]
        for child in el:
            lines.append(_element_to_text(child, indent + 1))
        return "\n".join(lines)

    tree = ET.parse(str(path))
    return _element_to_text(tree.getroot())


_LOADERS = {
    ".txt": (_load_txt, "plain text"),
    ".pdf": (_load_pdf, "PDF"),
    ".json": (_load_json, "JSON"),
    ".csv": (_load_csv, "CSV"),
    ".xml": (_load_xml, "XML"),
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an invoice data extraction agent. Extract structured information from the invoice content below.

Rules:
- Extract vendor name, total amount, due date, and all line items with quantities and unit prices.
- Normalize amounts to plain numbers (strip $, commas, currency symbols, etc.).
- Normalize dates to YYYY-MM-DD where possible; preserve original text if ambiguous.
- Set fields to null if missing or unreadable — do not invent data.
- Flag anomalies in the `flags` list: missing vendor, vague/invalid due dates, negative quantities, urgent payment pressure, or anything suspicious.
- Extract the invoice number or ID as written in the document into `invoice_number` (e.g. "INV-1004"); set to null if absent.
- If the invoice explicitly states it amends, revises, supersedes, or replaces a prior invoice, extract that original invoice ID into `replaces_invoice_id` (e.g. "Amendment to INV-1001" → "INV-1001"). Set to null if no such reference exists.

{feedback}

Invoice ({format}):
{invoice_text}"""


APPROVAL_PROMPT = """\
You are a senior accounts payable reviewer. Decide whether to approve or reject this invoice for payment.

Apply the following rules:
- Invoices over $10,000 MUST be flagged for human review (set requires_human_review=true).
- Flag for human review if you are uncertain about the vendor's legitimacy or the invoice's authenticity.
- Reject outright (no human review needed) if the vendor is clearly fraudulent, the due date is nonsensical, or there are obvious data integrity issues.
- Approve outright if the invoice is clean, vendor is credible, amount is under $10K, and due date is valid.

Reason through each criterion step by step before giving your final decision.

Invoice summary:
  Vendor:       {vendor}
  Total amount: {amount}
  Due date:     {due_date}
  Items:        {items}
  Flags:        {flags}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoice_id_from_path(file_path: str) -> str:
    stem = Path(file_path).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"INV-{parts[-1].upper()}"
    return stem


def _coerce_invoice_data(value) -> InvoiceData:
    """Deserialize extracted field — may come back as a dict from the checkpoint."""
    if isinstance(value, InvoiceData):
        return value
    items = [
        LineItem(**item) if isinstance(item, dict) else item
        for item in value.get("items", [])
    ]
    return InvoiceData(
        vendor=value.get("vendor"),
        total_amount=value.get("total_amount"),
        due_date=value.get("due_date"),
        items=items,
        flags=value.get("flags", []),
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_and_extract(state: InvoiceState) -> dict:
    path = Path(state["file_path"])
    if not path.exists():
        raise FileNotFoundError(f"Invoice not found: {state['file_path']}")

    suffix = path.suffix.lower()
    loader, fmt_label = _LOADERS.get(suffix, (None, None))
    if loader is None:
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported: {', '.join(_LOADERS)}")

    is_retry = bool(state.get("retry_feedback"))
    print(f"\n[Agent 1 - load_and_extract] {path.name}" + (" (retry)" if is_retry else ""))

    invoice_text = loader(path)

    feedback = ""
    if is_retry:
        for line in state["retry_feedback"].splitlines():
            print(f"        {line}")
        feedback = f"Previous attempt had issues — please fix them:\n{state['retry_feedback']}\n"

    prompt = EXTRACTION_PROMPT.format(feedback=feedback, format=fmt_label, invoice_text=invoice_text)
    result = extractor_llm().invoke(prompt)

    amount_str = f"${result.total_amount:,.2f}" if result.total_amount is not None else "N/A"
    items_str = ", ".join(f"{i.name} x{i.quantity}" for i in result.items) or "none"
    print(f"        vendor: {result.vendor or 'N/A'}  |  amount: {amount_str}  |  due: {result.due_date or 'N/A'}")
    print(f"        items:  {items_str}")
    if result.flags:
        for f in result.flags:
            print(f"        flag:   {f}")

    return {"invoice_text": invoice_text, "extracted": result}


def validation(state: InvoiceState) -> dict:
    data: InvoiceData = state["extracted"]
    print(f"\n[Agent 2 - validation] {_invoice_id_from_path(state['file_path'])}")

    extraction_issues = []
    if not data.vendor:
        extraction_issues.append("vendor is null — re-read the invoice for any company or sender name")
    if data.total_amount is None:
        extraction_issues.append("total_amount is null — look for Total, Amount Due, or Amt fields")
    if not data.items:
        extraction_issues.append("no line items extracted — look for product names with quantities")
    for item in data.items:
        if item.quantity is not None and item.quantity < 0:
            extraction_issues.append(f"item '{item.name}' has negative quantity ({item.quantity})")

    if extraction_issues and state["retry_count"] == 0:
        print(f"        extraction incomplete — will retry:")
        for issue in extraction_issues:
            print(f"          - {issue}")
        return {
            "retry_feedback": "\n".join(f"- {i}" for i in extraction_issues),
            "retry_count": 1,
            "inventory_flags": [],
        }

    inventory_flags = []
    suspicious_items = []
    amount_due = None

    invoice_id = _invoice_id_from_path(state["file_path"])
    source_file = Path(state["file_path"]).name

    def _apply_delta(prior_id: str, label: str) -> bool:
        """Look up prior paid amount and set amount_due. Returns True if handled."""
        nonlocal amount_due
        if was_human_reviewed(prior_id):
            inventory_flags.append(
                f"REJECTED: '{prior_id}' previously required human review; "
                f"new version ('{source_file}') is automatically rejected"
            )
            return True
        prev_paid = get_paid_amount(prior_id)
        if prev_paid is None:
            return False  # prior invoice not paid — treat as new
        current_amount = data.total_amount or 0
        delta = round(current_amount - prev_paid, 2)
        if delta > 0:
            amount_due = delta
            print(f"        {label}: prev paid ${prev_paid:,.2f}, delta ${delta:,.2f}")
        else:
            inventory_flags.append(
                f"NO PAYMENT DUE: '{prior_id}' previously paid ${prev_paid:,.2f}; "
                f"current amount ${current_amount:,.2f} requires no additional payment"
            )
        return True

    # Try explicit amendment reference first, then document invoice number, then filename.
    # Each path calls _apply_delta which sets amount_due or adds to inventory_flags.
    # We fall through to the next path only if the current one finds no prior record.
    handled = False

    # 1. Explicit amendment reference in the invoice content
    if data.replaces_invoice_id:
        handled = _apply_delta(data.replaces_invoice_id, f"Amendment to {data.replaces_invoice_id}")

    # 2. Document states an invoice number different from the filename-derived ID
    if not handled and data.invoice_number and data.invoice_number != invoice_id:
        handled = _apply_delta(data.invoice_number, f"Revision of {data.invoice_number}")

    # 3. Filename-based duplicate detection — always runs if nothing above handled it
    if not handled:
        existing = check_duplicate_invoice(invoice_id, source_file)
        if existing:
            if existing == source_file:
                inventory_flags.append(
                    f"DUPLICATE: '{invoice_id}' already saved from '{existing}' (exact duplicate)"
                )
            else:
                handled = _apply_delta(invoice_id, f"Revised invoice {invoice_id}")
                if not handled:
                    inventory_flags.append(
                        f"DUPLICATE ID: '{invoice_id}' already exists from '{existing}' (previously rejected); "
                        f"this version ('{source_file}') will be stored as rejected"
                    )
        elif was_human_reviewed(invoice_id):
            # Prior version is still awaiting human decision (in pending_reviews, not yet in invoices)
            inventory_flags.append(
                f"REJECTED: '{invoice_id}' is currently pending human review; "
                f"new version ('{source_file}') is automatically rejected"
            )

    if data.vendor and not lookup_vendor(data.vendor):
        suspicious_items.append(f"UNKNOWN VENDOR: '{data.vendor}' is not in the approved vendor list")

    known_items = get_inventory_items()

    for item in data.items:
        # Parenthetical additions are suspicious (e.g. "WidgetA (rush fee)")
        if "(" in item.name:
            suspicious_items.append(f"PARENTHETICAL: '{item.name}' contains unexpected qualifier")
            continue

        # Fuzzy match — flag names that look like typos of known items
        close = difflib.get_close_matches(item.name, known_items, n=1, cutoff=0.6)
        if close and close[0] != item.name:
            suspicious_items.append(
                f"POSSIBLE TYPO: '{item.name}' looks like '{close[0]}'"
            )
            continue

        stock = query_inventory(item.name)
        if stock is None:
            inventory_flags.append(f"UNKNOWN ITEM: '{item.name}' not found in inventory")
        elif stock == 0:
            inventory_flags.append(f"OUT OF STOCK: '{item.name}' has 0 units available")
        elif item.quantity is not None and item.quantity > stock:
            inventory_flags.append(
                f"STOCK MISMATCH: '{item.name}' requests {int(item.quantity)} "
                f"but only {stock} in stock"
            )
        elif item.quantity is not None and item.quantity < 0:
            inventory_flags.append(f"INVALID: '{item.name}' has negative quantity ({item.quantity}) after retry")

    if inventory_flags:
        for flag in inventory_flags:
            print(f"        rejected: {flag}")
    if suspicious_items:
        for s in suspicious_items:
            print(f"        suspicious: {s}")
        print(f"        → routing to specialist review")
    if not inventory_flags and not suspicious_items:
        print(f"        passed → routing to approve")

    return {"inventory_flags": inventory_flags, "suspicious_items": suspicious_items, "amount_due": amount_due, "retry_feedback": ""}


def approve(state: InvoiceState) -> dict:
    data: InvoiceData = _coerce_invoice_data(state["extracted"])
    items_summary = ", ".join(f"{item.name} x{item.quantity}" for item in data.items)
    flags_summary = "; ".join(data.flags) if data.flags else "none"

    prompt = APPROVAL_PROMPT.format(
        vendor=data.vendor or "UNKNOWN",
        amount=f"${data.total_amount:,.2f}" if data.total_amount is not None else "UNKNOWN",
        due_date=data.due_date or "NOT SPECIFIED",
        items=items_summary,
        flags=flags_summary,
    )

    print(f"\n[Agent 3 - approve] {data.vendor or 'unknown vendor'}")
    decision = approval_llm().invoke(prompt)
    if decision.approved:
        outcome = "approved"
    elif decision.requires_human_review:
        outcome = "needs VP review"
    else:
        outcome = "rejected"
    print(f"        {outcome}: {decision.reasoning[:120]}")
    return {"approval": decision}


def _do_human_review(state: InvoiceState, reasoning: str, reviewer_role: str) -> dict:
    data: InvoiceData = _coerce_invoice_data(state["extracted"])
    thread_id = state.get("thread_id", str(uuid.uuid4()))
    invoice_id = _invoice_id_from_path(state["file_path"])

    node_name = "Specialist - specialist_review" if reviewer_role == "specialist" else "VP - vp_approval"
    print(f"\n[{node_name}] {invoice_id} — waiting for {reviewer_role} decision")
    print(f"        reason: {reasoning[:120]}")
    save_pending_review(thread_id, invoice_id, data, reasoning, reviewer_role)

    human_response = interrupt({
        "invoice_id": invoice_id,
        "reasoning": reasoning,
        "thread_id": thread_id,
    })

    decision_word = "approved" if human_response["approved"] else "rejected"
    note = human_response.get("note", "")
    print(f"\n[{reviewer_role}] {invoice_id} — {decision_word}" + (f": {note}" if note else ""))
    record_review_decision(thread_id, decision_word, note)
    return {"approval": ApprovalDecision(
        approved=human_response["approved"],
        requires_human_review=False,
        reasoning=note or "Decision made by human reviewer.",
    )}


def validation_review(state: InvoiceState) -> dict:
    suspicious = state.get("suspicious_items", [])
    reasoning = "Suspicious item names flagged for specialist review: " + "; ".join(suspicious)
    return _do_human_review(state, reasoning, "specialist")


def vp_review(state: InvoiceState) -> dict:
    approval = state.get("approval")
    reasoning = (approval.reasoning if isinstance(approval, ApprovalDecision)
                 else approval.get("reasoning", "") if isinstance(approval, dict) else "")
    return _do_human_review(state, reasoning, "VP")


def mock_payment(vendor: str, amount: float) -> dict:
    print(f"        payment: ${amount:,.2f} to {vendor}")
    return {"status": "success"}


def save(state: InvoiceState) -> dict:
    data: InvoiceData = _coerce_invoice_data(state["extracted"])
    source_file = Path(state["file_path"]).name

    invoice_id = _invoice_id_from_path(state["file_path"])

    inventory_flags = state["inventory_flags"]
    approval: Optional[ApprovalDecision] = state.get("approval")

    if inventory_flags:
        review_status = "rejected"
        approval_status = None
        rejection_reason = "; ".join(inventory_flags)
    elif approval and not approval.approved:
        review_status = "passed"
        approval_status = "rejected"
        rejection_reason = approval.reasoning
    else:
        review_status = "passed"
        approval_status = "approved"
        rejection_reason = None

    print(f"\n[Agent 4 - save] {invoice_id}  review={review_status}  approval={approval_status or 'n/a'}")
    if rejection_reason:
        print(f"        reason: {rejection_reason[:120]}")

    result = save_invoice(
        invoice_id=invoice_id,
        vendor_name=data.vendor,
        total_amount=data.total_amount,
        due_date=data.due_date,
        source_file=source_file,
        review_status=review_status,
        approval_status=approval_status,
        rejection_reason=rejection_reason,
        items=[
            {"name": item.name, "quantity": item.quantity, "unit_price": item.unit_price}
            for item in data.items
        ],
    )
    if approval_status == "approved" and data.vendor and data.total_amount is not None:
        payment_amount = state.get("amount_due") or data.total_amount
        payment_result = mock_payment(data.vendor, payment_amount)
        if payment_result["status"] != "success":
            print(f"        payment failed: {payment_result}")
        else:
            if state.get("amount_due") is not None:
                print(f"        (delta only — full invoice ${data.total_amount:,.2f})")

    return {"saved": True}
