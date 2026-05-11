"""
Pipeline test — runs every invoice in data/invoices/ through the agent and prints a summary.
Clears the databases on each run so results are always fresh.

Usage:
    python test_pipeline.py
    python test_pipeline.py --invoices_dir path/to/invoices
"""

import warnings
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# DB paths derived here (without importing utils.db) so we can delete them
# before the DB-initialising imports run.
_DATA_DIR = Path(__file__).parent / "data"
_DB_PATH = _DATA_DIR / "inventory.db"
_CHECKPOINT_PATH = _DATA_DIR / "checkpoints.db"

POLL_INTERVAL = 3  # seconds between DB checks while waiting for human decision
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".json", ".csv", ".xml"}


def _reset_databases() -> None:
    for path in [_DB_PATH, _CHECKPOINT_PATH]:
        for suffix in ["", "-wal", "-shm"]:
            p = Path(str(path) + suffix)
            if p.exists():
                p.unlink()
    print("  Databases cleared.")


def _wait_for_decision(thread_id: str) -> tuple[str, str]:
    """Block until a human decision is recorded in pending_reviews."""
    while True:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT decision, decision_note FROM pending_reviews "
                "WHERE thread_id = ? AND decision IS NOT NULL",
                (thread_id,),
            ).fetchone()
        if row:
            return row[0], row[1] or ""
        time.sleep(POLL_INTERVAL)


def run_all(invoices_dir: Path) -> None:
    # Import after DB files are deleted so init_*_tables() starts from scratch
    from langgraph.types import Command
    from agents.graph import make_agent
    from agents.nodes import _invoice_id_from_path

    if not os.environ.get("XAI_API_KEY"):
        sys.exit("Error: XAI_API_KEY environment variable is not set.")

    invoice_files = sorted(
        f for f in invoices_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not invoice_files:
        print(f"No supported invoice files found in {invoices_dir}")
        return

    agent = make_agent()

    results = []
    print(f"\nRunning pipeline on {len(invoice_files)} invoice(s) in {invoices_dir}\n")
    sep = "─" * 60

    for invoice_path in invoice_files:
        thread_id = invoice_path.stem
        config = {"configurable": {"thread_id": thread_id}}
        invoice_id = _invoice_id_from_path(str(invoice_path))

        print(f"{sep}")
        print(f"  {invoice_path.name}  ({invoice_id})")

        start = time.time()
        status = "unknown"
        note = ""

        try:
            final_state = agent.invoke(
                {
                    "thread_id": thread_id,
                    "file_path": str(invoice_path),
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

            if agent.get_state(config).next:
                print(f"  ⏸   Waiting for human review — open the UI and decide.")
                decision, decision_note = _wait_for_decision(thread_id)
                final_state = agent.invoke(
                    Command(resume={"approved": decision == "approved", "note": decision_note}),
                    config=config,
                )

            flags = final_state.get("inventory_flags", [])
            approval = final_state.get("approval")

            if flags:
                status = "rejected"
                note = flags[0]
            elif approval and not approval.approved:
                status = "rejected"
                note = approval.reasoning[:80]
            else:
                status = "accepted"

        except Exception as e:
            status = "error"
            note = str(e)

        elapsed = time.time() - start
        icon = {"accepted": "✅", "rejected": "❌", "paused": "⏸ ", "error": "💥"}.get(status, "?")
        print(f"  {icon}  {status.upper():<10}  ({elapsed:.1f}s)")
        if note:
            print(f"     {note}")

        results.append({"file": invoice_path.name, "status": status, "note": note})

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY — {len(results)} invoices")
    print(f"{'═' * 60}")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for s, count in sorted(counts.items()):
        icon = {"accepted": "✅", "rejected": "❌", "paused": "⏸ ", "error": "💥"}.get(s, "?")
        print(f"  {icon}  {s:<12} {count}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the invoice pipeline on all invoices in a directory.")
    parser.add_argument(
        "--invoices_dir",
        type=Path,
        default=Path("data/invoices"),
        help="Directory containing invoice files (default: data/invoices)",
    )
    args = parser.parse_args()

    if not args.invoices_dir.exists():
        sys.exit(f"Error: directory not found: {args.invoices_dir}")

    _reset_databases()
    run_all(args.invoices_dir)
