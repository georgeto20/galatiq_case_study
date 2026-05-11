import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", module="urllib3")

from langgraph.graph import StateGraph, END

from utils.models import InvoiceState, ApprovalDecision
from agents.nodes import load_and_extract, validation, approve, validation_review, vp_review, save
from utils.db import checkpointer


def route_validation(state: InvoiceState) -> str:
    if state.get("retry_feedback") and state["retry_count"] == 1:
        return "retry"
    if state["inventory_flags"]:
        return "rejected"
    if state.get("suspicious_items"):
        return "suspicious"
    return "passed"


def route_validation_review(state: InvoiceState) -> str:
    approval = state.get("approval")
    approved = approval.approved if isinstance(approval, ApprovalDecision) else (
        approval.get("approved", False) if isinstance(approval, dict) else False
    )
    return "passed" if approved else "rejected"


def route_approve(state: InvoiceState) -> str:
    approval = state.get("approval")
    requires_human = (
        approval.requires_human_review if isinstance(approval, ApprovalDecision)
        else approval.get("requires_human_review", False) if isinstance(approval, dict)
        else False
    )
    return "needs vp approval" if requires_human else "doesn't need vp approval"


def build_graph_base() -> StateGraph:
    graph = StateGraph(InvoiceState)

    graph.add_node("Agent 1 - load_and_extract", load_and_extract)
    graph.add_node("Agent 2 - validation", validation)
    graph.add_node("Agent 3 - approve", approve)
    graph.add_node("Specialist - specialist_review", validation_review)
    graph.add_node("VP - vp_approval", vp_review)
    graph.add_node("Agent 4 - save", save)

    graph.set_entry_point("Agent 1 - load_and_extract")
    graph.add_edge("Agent 1 - load_and_extract", "Agent 2 - validation")
    graph.add_conditional_edges(
        "Agent 2 - validation",
        route_validation,
        {"retry": "Agent 1 - load_and_extract", "passed": "Agent 3 - approve", "rejected": "Agent 4 - save", "suspicious": "Specialist - specialist_review"},
    )
    graph.add_conditional_edges(
        "Agent 3 - approve",
        route_approve,
        {"needs vp approval": "VP - vp_approval", "doesn't need vp approval": "Agent 4 - save"},
    )
    graph.add_conditional_edges(
        "Specialist - specialist_review",
        route_validation_review,
        {"passed": "Agent 3 - approve", "rejected": "Agent 4 - save"},
    )
    graph.add_edge("VP - vp_approval", "Agent 4 - save")
    graph.add_edge("Agent 4 - save", END)

    return graph


def make_agent(cp=None):
    """Compile the graph with a checkpointer. Uses the module-level one by default."""
    return build_graph_base().compile(checkpointer=cp or checkpointer)


agent = make_agent()
