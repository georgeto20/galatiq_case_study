import difflib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from utils.models import InvoiceData

DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"
CHECKPOINT_DB_PATH = Path(__file__).parent.parent / "data" / "checkpoints.db"

DB_PATH.parent.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

_checkpoint_conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(_checkpoint_conn)
checkpointer.setup()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def init_invoice_tables() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id       TEXT NOT NULL,
                vendor_name      TEXT,
                total_amount     REAL,
                due_date         TEXT,
                source_file      TEXT,
                review_status    TEXT CHECK(review_status IN ('passed', 'rejected')),
                approval_status  TEXT CHECK(approval_status IN ('approved', 'rejected')),
                rejection_reason TEXT,
                UNIQUE(invoice_id, source_file)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invoice_items (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id   TEXT,
                item_name    TEXT,
                quantity     INTEGER,
                unit_price   REAL
            )
        """)

        cols = [row[1] for row in conn.execute("PRAGMA table_info(invoices)")]

        # Migrate: invoice_id was once the primary key
        pk_col = next(
            (row[1] for row in conn.execute("PRAGMA table_info(invoices)") if row[5] == 1),
            None,
        )
        if pk_col == "invoice_id":
            conn.execute("""
                CREATE TABLE invoices_new (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id       TEXT NOT NULL,
                    vendor_name      TEXT,
                    total_amount     REAL,
                    due_date         TEXT,
                    source_file      TEXT,
                    review_status    TEXT CHECK(review_status IN ('passed', 'rejected')),
                    approval_status  TEXT CHECK(approval_status IN ('approved', 'rejected')),
                    rejection_reason TEXT,
                    UNIQUE(invoice_id, source_file)
                )
            """)
            conn.execute("""
                INSERT INTO invoices_new (invoice_id, vendor_name, total_amount, due_date,
                                         source_file, review_status, approval_status, rejection_reason)
                SELECT invoice_id, vendor_name, total_amount, due_date, source_file,
                       CASE WHEN status = 'accepted' THEN 'passed' ELSE 'rejected' END,
                       CASE WHEN status = 'accepted' THEN 'approved' ELSE NULL END,
                       rejection_reason FROM invoices
            """)
            conn.execute("DROP TABLE invoices")
            conn.execute("ALTER TABLE invoices_new RENAME TO invoices")
            cols = [row[1] for row in conn.execute("PRAGMA table_info(invoices)")]

        # Migrate: single 'status' column → review_status + approval_status
        if "status" in cols and "review_status" not in cols:
            conn.execute("""
                CREATE TABLE invoices_new (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id       TEXT NOT NULL,
                    vendor_name      TEXT,
                    total_amount     REAL,
                    due_date         TEXT,
                    source_file      TEXT,
                    review_status    TEXT CHECK(review_status IN ('passed', 'rejected')),
                    approval_status  TEXT CHECK(approval_status IN ('approved', 'rejected')),
                    rejection_reason TEXT,
                    UNIQUE(invoice_id, source_file)
                )
            """)
            conn.execute("""
                INSERT INTO invoices_new (id, invoice_id, vendor_name, total_amount, due_date,
                                         source_file, review_status, approval_status, rejection_reason)
                SELECT id, invoice_id, vendor_name, total_amount, due_date, source_file,
                       CASE WHEN status = 'accepted' THEN 'passed' ELSE 'rejected' END,
                       CASE WHEN status = 'accepted' THEN 'approved' ELSE NULL END,
                       rejection_reason FROM invoices
            """)
            conn.execute("DROP TABLE invoices")
            conn.execute("ALTER TABLE invoices_new RENAME TO invoices")


def init_pending_reviews_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_reviews (
                thread_id      TEXT PRIMARY KEY,
                invoice_id     TEXT,
                vendor_name    TEXT,
                total_amount   REAL,
                due_date       TEXT,
                items_summary  TEXT,
                reasoning      TEXT,
                created_at     TEXT,
                decision       TEXT,
                decision_note  TEXT,
                reviewer_role  TEXT DEFAULT 'VP'
            )
        """)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(pending_reviews)")]
        if "reviewer_role" not in cols:
            conn.execute("ALTER TABLE pending_reviews ADD COLUMN reviewer_role TEXT DEFAULT 'VP'")


def init_inventory_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)")
        cursor.executemany(
            "INSERT OR IGNORE INTO inventory (item, stock) VALUES (?, ?)",
            [("WidgetA", 15), ("WidgetB", 10), ("GadgetX", 5), ("FakeItem", 0)],
        )


APPROVED_VENDORS = [
    "Widgets Inc.",
    "Gadgets Co.",
    "Fraudster LLC",
    "Precision Parts Ltd.",
    "Global Supply Chain Partners",
    "Acme Industrial Supplies",
    "MegaWidgets Corp",
    "NoProd Industries",
    "Consolidated Materials Group",
    "Summit Manufacturing Co.",
    "QuickShip Distributers",
    "Atlas Industrial Supply",
    "TechParts International",
    "Reliable Components Inc.",
]


def init_vendors_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS vendors (name TEXT PRIMARY KEY)")

        # Migrate: old schema had status + notes columns — drop them
        cols = [row[1] for row in conn.execute("PRAGMA table_info(vendors)")]
        if "status" in cols:
            conn.execute("CREATE TABLE vendors_new (name TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO vendors_new (name) SELECT name FROM vendors")
            conn.execute("DROP TABLE vendors")
            conn.execute("ALTER TABLE vendors_new RENAME TO vendors")

        conn.executemany(
            "INSERT OR IGNORE INTO vendors (name) VALUES (?)",
            [(v,) for v in APPROVED_VENDORS],
        )


init_invoice_tables()
init_pending_reviews_table()
init_inventory_table()
init_vendors_table()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def save_pending_review(
    thread_id: str, invoice_id: str, data: InvoiceData, reasoning: str, reviewer_role: str = "VP"
) -> None:
    items_summary = "; ".join(
        f"{item.name} x{int(item.quantity or 0)}" for item in data.items
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pending_reviews "
            "(thread_id, invoice_id, vendor_name, total_amount, due_date, items_summary, reasoning, created_at, reviewer_role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, invoice_id, data.vendor, data.total_amount,
             data.due_date, items_summary, reasoning, datetime.utcnow().isoformat(), reviewer_role),
        )


def record_review_decision(thread_id: str, decision: str, note: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE pending_reviews SET decision = ?, decision_note = ? WHERE thread_id = ?",
            (decision, note, thread_id),
        )


def query_inventory(item_name: str) -> Optional[int]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT stock FROM inventory WHERE item = ?", (item_name,)
        ).fetchone()
    return row[0] if row else None


def lookup_vendor(vendor_name: str) -> bool:
    """Return True if the vendor is in the known vendor list (exact or fuzzy match)."""
    with sqlite3.connect(DB_PATH) as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM vendors").fetchall()]

    if vendor_name in names:
        return True

    return bool(difflib.get_close_matches(vendor_name, names, n=1, cutoff=0.8))


def get_inventory_items() -> list[str]:
    """Return all item names in the inventory table."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT item FROM inventory").fetchall()
    return [row[0] for row in rows]


def was_human_reviewed(invoice_id: str) -> bool:
    """Return True if any version of this invoice_id has an entry in pending_reviews."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM pending_reviews WHERE invoice_id = ? LIMIT 1",
            (invoice_id,),
        ).fetchone()
    return row is not None


def get_paid_amount(invoice_id: str) -> Optional[float]:
    """Return total amount already paid for invoice_id (sum of all accepted versions), or None if none accepted."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT SUM(total_amount) FROM invoices WHERE invoice_id = ? AND review_status = 'passed' AND approval_status = 'approved'",
            (invoice_id,),
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def check_duplicate_invoice(invoice_id: str, source_file: str) -> Optional[str]:
    """Return the first existing source_file for invoice_id, regardless of extension.

    Same file → exact duplicate (skip re-save via INSERT OR IGNORE).
    Different file → variant duplicate (saved as rejected with its own row).
    Returns None if this invoice_id has never been seen before.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT source_file FROM invoices WHERE invoice_id = ? ORDER BY id LIMIT 1",
            (invoice_id,),
        ).fetchone()
    return row[0] if row else None


def save_invoice(
    invoice_id: str,
    vendor_name: Optional[str],
    total_amount: Optional[float],
    due_date: Optional[str],
    source_file: str,
    review_status: str,
    approval_status: Optional[str],
    rejection_reason: Optional[str],
    items: list[dict],
) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        inserted = conn.execute(
            "INSERT OR IGNORE INTO invoices "
            "(invoice_id, vendor_name, total_amount, due_date, source_file, review_status, approval_status, rejection_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (invoice_id, vendor_name, total_amount, due_date, source_file, review_status, approval_status, rejection_reason),
        ).rowcount
        if inserted:
            conn.executemany(
                "INSERT INTO invoice_items (invoice_id, item_name, quantity, unit_price) VALUES (?, ?, ?, ?)",
                [(invoice_id, item.get("name"), item.get("quantity"), item.get("unit_price")) for item in items],
            )
    outcome = f"review={review_status}, approval={approval_status or 'n/a'}"
    return f"Saved invoice {invoice_id} — {outcome}."
