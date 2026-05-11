"""
Invoice Monitor — queries the database and returns/prints pipeline statistics.
Can be run directly (every 10 min) or imported by the UI.

Usage:
    python utils/monitor.py
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"


_EMPTY_STATS = {
    "as_of": "",
    "pipeline": {"total": 0, "accepted": 0, "rejected": 0, "accepted_value": 0.0, "rejected_value": 0.0},
    "invoice_status": {"total": 0, "auto_approved": 0, "auto_rejected": 0, "routed_to_humans": 0},
    "pending_human": {"count": 0, "value": 0.0},
    "human_decisions": {},
    "top_vendors": [],
    "recent": [],
}


def fetch_stats() -> dict:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if not DB_PATH.exists():
        return {**_EMPTY_STATS, "as_of": now}

    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row

            totals = conn.execute("""
                SELECT
                    COUNT(*)  AS total,
                    SUM(review_status = 'passed' AND approval_status = 'approved')  AS accepted,
                    SUM(review_status = 'rejected' OR approval_status = 'rejected') AS rejected,
                    SUM(CASE WHEN review_status='passed' AND approval_status='approved' THEN total_amount ELSE 0 END) AS accepted_value,
                    SUM(CASE WHEN review_status='rejected' OR approval_status='rejected'  THEN total_amount ELSE 0 END) AS rejected_value
                FROM invoices
            """).fetchone()

            pending = conn.execute("""
                SELECT COUNT(*) AS cnt, SUM(total_amount) AS value
                FROM pending_reviews
                WHERE decision IS NULL
            """).fetchone()

            human = conn.execute("""
                SELECT reviewer_role, decision, COUNT(*) AS cnt, SUM(total_amount) AS value
                FROM pending_reviews
                WHERE decision IS NOT NULL
                GROUP BY reviewer_role, decision
            """).fetchall()

            top_vendors = conn.execute("""
                SELECT vendor_name, COUNT(*) AS cnt, SUM(total_amount) AS value
                FROM invoices
                WHERE review_status = 'passed' AND approval_status = 'approved' AND vendor_name IS NOT NULL
                GROUP BY vendor_name
                ORDER BY value DESC
                LIMIT 5
            """).fetchall()

            recent = conn.execute("""
                SELECT invoice_id, vendor_name, total_amount, decision, created_at
                FROM pending_reviews
                WHERE decision IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 5
            """).fetchall()

            invoice_status = conn.execute("""
                SELECT
                    SUM(CASE WHEN review_status = 'passed' AND approval_status = 'approved'
                                  AND invoice_id NOT IN (SELECT invoice_id FROM pending_reviews)
                             THEN 1 ELSE 0 END) AS auto_approved,
                    SUM(CASE WHEN review_status = 'rejected'
                              OR (review_status = 'passed' AND approval_status = 'rejected'
                                  AND invoice_id NOT IN (SELECT invoice_id FROM pending_reviews))
                             THEN 1 ELSE 0 END) AS auto_rejected
                FROM invoices
            """).fetchone()

            routed_count = conn.execute("""
                SELECT COUNT(DISTINCT invoice_id) FROM pending_reviews
            """).fetchone()[0]

            pending_count = conn.execute("""
                SELECT COUNT(*) FROM pending_reviews WHERE decision IS NULL
            """).fetchone()[0]

    except sqlite3.OperationalError:
        return {**_EMPTY_STATS, "as_of": now}

    human_decisions = [
        {"role": row["reviewer_role"], "decision": row["decision"], "cnt": row["cnt"], "value": row["value"] or 0}
        for row in human
    ]

    auto_approved = invoice_status["auto_approved"] or 0
    auto_rejected = invoice_status["auto_rejected"] or 0
    routed = routed_count
    is_total = auto_approved + auto_rejected + routed

    return {
        "as_of": now,
        "invoice_status": {
            "total": is_total,
            "auto_approved": auto_approved,
            "auto_rejected": auto_rejected,
            "routed_to_humans": routed,
        },
        "pipeline": {
            "total":          totals["total"] or 0,
            "accepted":       totals["accepted"] or 0,
            "rejected":       totals["rejected"] or 0,
            "accepted_value": totals["accepted_value"] or 0.0,
            "rejected_value": totals["rejected_value"] or 0.0,
        },
        "pending_human": {
            "count": pending["cnt"] or 0,
            "value": pending["value"] or 0.0,
        },
        "human_decisions": human_decisions,
        "top_vendors": [dict(r) for r in top_vendors],
        "recent": [dict(r) for r in recent],
    }


def print_report(stats: dict) -> None:
    p = stats["pipeline"]
    ph = stats["pending_human"]
    hd = stats["human_decisions"]

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  INVOICE MONITOR  —  {stats['as_of']}")
    print(sep)

    print(f"\n  Pipeline totals")
    print(f"    Processed :  {p['total']:>5}")
    print(f"    Accepted  :  {p['accepted']:>5}   ${p['accepted_value']:>12,.2f}")
    print(f"    Rejected  :  {p['rejected']:>5}   ${p['rejected_value']:>12,.2f}")

    print(f"\n  Human review queue")
    print(f"    Awaiting  :  {ph['count']:>5}   ${ph['value']:>12,.2f}")

    if hd:
        print(f"\n  Human decisions")
        for row in hd:
            icon = "✅" if row["decision"] == "approved" else "❌"
            print(f"    {icon} {row['role']:<12} {row['decision']:<10}:  {row['cnt']:>5}   ${row['value']:>12,.2f}")

    if stats["top_vendors"]:
        print(f"\n  Top vendors (by accepted value)")
        for v in stats["top_vendors"]:
            print(f"    {v['vendor_name']:<28}  {v['cnt']:>2} inv  ${v['value']:>10,.2f}")

    if stats["recent"]:
        print(f"\n  Recent decisions")
        for r in stats["recent"]:
            icon = "✅" if r["decision"] == "approved" else "❌"
            amt = f"${r['total_amount']:,.2f}" if r["total_amount"] else "N/A"
            print(f"    {icon}  {r['invoice_id']:<14}  {(r['vendor_name'] or 'Unknown'):<20}  {amt}")

    print(f"\n{sep}\n")


def check_alerts(stats: dict) -> list[str]:
    alerts = []
    if stats["pending_human"]["count"] >= 5:
        alerts.append(f"⚠️  {stats['pending_human']['count']} invoices awaiting human review — queue is growing")
    if stats["pending_human"]["value"] >= 50_000:
        alerts.append(f"⚠️  ${stats['pending_human']['value']:,.2f} held in human review queue")
    return alerts


def run() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    stats = fetch_stats()
    print_report(stats)

    alerts = check_alerts(stats)
    if alerts:
        print("ALERTS:")
        for a in alerts:
            print(f"  {a}")
        print()


if __name__ == "__main__":
    run()
