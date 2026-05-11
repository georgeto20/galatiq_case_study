"""
Invoice Review UI — Streamlit
Shows invoices paused for human review and resumes the agent with the decision.
Includes a Command Center tab with live pipeline statistics.
"""

import warnings
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

import sqlite3

import streamlit as st
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.graph import make_agent
from utils.db import CHECKPOINT_DB_PATH, DB_PATH
from utils.monitor import fetch_stats

st.set_page_config(page_title="Invoice Review", page_icon="🧾", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources — created once per Streamlit session, not on every rerun
# ---------------------------------------------------------------------------

def _db_mtime() -> float:
    """Return checkpoint DB mtime, or 0 if it doesn't exist yet."""
    return CHECKPOINT_DB_PATH.stat().st_mtime if CHECKPOINT_DB_PATH.exists() else 0


@st.cache_resource
def get_agent(mtime: float):
    """Create a single agent+checkpointer. Re-created whenever the DB mtime changes."""
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    cp = SqliteSaver(conn)
    cp.setup()
    return make_agent(cp)


agent = get_agent(_db_mtime())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def row_to_dict(row) -> dict:
    return {key: row[key] for key in row.keys()}


def load_pending(reviewer_role: str) -> list[dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM pending_reviews WHERE decision IS NULL AND reviewer_role = ? ORDER BY created_at DESC",
            (reviewer_role,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def load_decided() -> list[dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM invoices ORDER BY id DESC"
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def submit_decision(thread_id: str, approved: bool, note: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke(
        Command(resume={"approved": approved, "note": note}),
        config=config,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

try:
    _test_pending = load_pending("specialist")
except Exception as _e:
    st.error(f"DB error: {_e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

st.title("🧾 Invoice Review")
st.caption("Invoices flagged for human approval appear here. Review the agent's reasoning and make a decision.")

col_title, col_refresh = st.columns([6, 1])
with col_refresh:
    if st.button("🔄 Refresh", width='stretch'):
        st.rerun()

tab_dashboard, tab_specialist, tab_vp, tab_history = st.tabs(["📊 Command Center", "🔬 Specialist Validation", "👔 VP Review", "📋 History"])

# ── Command Center ───────────────────────────────────────────────────────────
with tab_dashboard:
    @st.fragment(run_every=5)
    def _dashboard():
        import pandas as pd

        stats = fetch_stats()
        p = stats["pipeline"]
        ph = stats["pending_human"]
        hd = stats["human_decisions"]

        st.caption(f"Last updated: {stats['as_of']}  ·  auto-refreshes every 5 s")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Processed", p["total"])
        k2.metric("✅ Accepted", p["accepted"], delta=f"${p['accepted_value']:,.2f}")
        k3.metric("❌ Rejected", p["rejected"])
        k4.metric("⏳ Awaiting Review", ph["count"], delta=f"${ph['value']:,.2f} held",
                  delta_color="inverse")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Invoice status")
            ist = stats["invoice_status"]
            if ist["total"] > 0:
                def _pct(n): return f"{n / ist['total'] * 100:.0f}%"
                status_df = pd.DataFrame({
                    "Category": ["Approved automatically", "Rejected automatically", "Routed to humans"],
                    "Count": [ist["auto_approved"], ist["auto_rejected"], ist["routed_to_humans"]],
                    "Share": [_pct(ist["auto_approved"]), _pct(ist["auto_rejected"]), _pct(ist["routed_to_humans"])],
                })
                st.dataframe(status_df, width='stretch', hide_index=True)
                st.bar_chart(status_df.set_index("Category")["Count"])
            else:
                st.caption("No invoices processed yet.")

        with col_r:
            st.subheader("Human review outcomes")
            if hd:
                hd_df = pd.DataFrame([
                    {
                        "Reviewer": row["role"],
                        "Decision": row["decision"].capitalize(),
                        "Count": row["cnt"],
                        "Value ($)": f"${row['value']:,.2f}",
                    }
                    for row in hd
                ])
                st.dataframe(hd_df, width='stretch', hide_index=True)
            else:
                st.caption("No human decisions recorded yet.")

        alerts = []
        if ph["count"] >= 5:
            alerts.append(f"⚠️  **{ph['count']} invoices** in the human review queue — action needed.")
        if ph["value"] >= 50_000:
            alerts.append(f"⚠️  **${ph['value']:,.2f}** held in human review queue.")
        if alerts:
            st.divider()
            for a in alerts:
                st.warning(a)

        if stats["top_vendors"]:
            st.divider()
            st.subheader("Top vendors by accepted value")
            vdf = pd.DataFrame(stats["top_vendors"]).rename(columns={
                "vendor_name": "Vendor", "cnt": "Invoices", "value": "Total Accepted ($)"
            })
            vdf["Total Accepted ($)"] = vdf["Total Accepted ($)"].apply(lambda v: f"${v:,.2f}")
            st.dataframe(vdf, width='stretch', hide_index=True)

        if stats["recent"]:
            st.divider()
            st.subheader("Recent decisions")
            for r in stats["recent"]:
                icon = "✅" if r["decision"] == "approved" else "❌"
                amt = f"${r['total_amount']:,.2f}" if r["total_amount"] else "N/A"
                st.markdown(
                    f"{icon} **{r['invoice_id']}** &nbsp;·&nbsp; "
                    f"{r['vendor_name'] or 'Unknown'} &nbsp;·&nbsp; {amt}"
                )

    _dashboard()

def _render_pending(rows: list[dict]) -> None:
    for row in rows:
        with st.container(border=True):
            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.subheader(f"🗂 {row['invoice_id']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Vendor", row["vendor_name"] or "Unknown")
                m2.metric("Amount", f"${row['total_amount']:,.2f}" if row["total_amount"] else "N/A")
                m3.metric("Due Date", row["due_date"] or "Not specified")

                st.markdown("**Items**")
                st.caption(row["items_summary"] or "—")

                st.markdown("**Agent reasoning**")
                st.info(row["reasoning"])

            with col_right:
                st.markdown("**Your decision**")
                note = st.text_area(
                    "Note (optional)",
                    placeholder="Add a comment or override reason...",
                    key=f"note_{row['thread_id']}",
                    height=120,
                )

                approve_btn = st.button(
                    "✅ Approve",
                    key=f"approve_{row['thread_id']}",
                    type="primary",
                    width='stretch',
                )
                reject_btn = st.button(
                    "❌ Reject",
                    key=f"reject_{row['thread_id']}",
                    width='stretch',
                )

                tid = row["thread_id"]
                pending_key = f"pending_decision_{tid}"

                if approve_btn:
                    st.session_state[pending_key] = {"approved": True, "note": note}
                    st.rerun()
                if reject_btn:
                    st.session_state[pending_key] = {"approved": False, "note": note}
                    st.rerun()

                if pending_key in st.session_state:
                    decision = st.session_state.pop(pending_key)
                    with st.spinner("Resuming agent..."):
                        submit_decision(tid, approved=decision["approved"], note=decision["note"])
                    if decision["approved"]:
                        st.success("Invoice approved — agent resumed.")
                    else:
                        st.warning("Invoice rejected — agent resumed.")
                    st.rerun()


# ── Specialist Validation ─────────────────────────────────────────────────────
with tab_specialist:
    specialist_rows = load_pending("specialist")
    if not specialist_rows:
        st.success("No invoices pending specialist validation.")
    else:
        st.info(f"{len(specialist_rows)} invoice(s) awaiting specialist review.")
    _render_pending(specialist_rows)

# ── VP Review ─────────────────────────────────────────────────────────────────
with tab_vp:
    vp_rows = load_pending("VP")
    if not vp_rows:
        st.success("No invoices pending VP review.")
    else:
        st.info(f"{len(vp_rows)} invoice(s) awaiting VP decision.")
    _render_pending(vp_rows)

# ── History ──────────────────────────────────────────────────────────────────
with tab_history:
    decided = load_decided()

    if not decided:
        st.caption("No decisions made yet.")
    else:
        for row in decided:
            accepted = row["review_status"] == "passed" and row["approval_status"] == "approved"
            icon = "✅" if accepted else "❌"
            amount_str = f"${row['total_amount']:,.2f}" if row["total_amount"] else "N/A"
            label = f"{icon} {row['invoice_id']} — {row['vendor_name'] or 'Unknown'} — {amount_str}"
            with st.expander(label):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Review", row["review_status"] or "—")
                m2.metric("Approval", row["approval_status"] or "—")
                m3.metric("Amount", amount_str)
                m4.metric("Due Date", row["due_date"] or "Not specified")
                if row["source_file"]:
                    st.caption(f"Source: {row['source_file']}")
                if row["rejection_reason"]:
                    st.markdown("**Rejection reason:**")
                    st.caption(row["rejection_reason"])
