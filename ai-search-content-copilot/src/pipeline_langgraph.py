"""
LangGraph pipeline — two-layer agentic GEO workflow with iterative refinement.

Layer 1 (sequential, runs once):
  extract_webpage → research_product → write_markdown → verify_claims → analyze_gaps

Layer 2 (iterative loop):
  simulate_geo → [should_continue?]
                      ├─ "revise"  → analyze_visibility → select_strategies
                      │                                         → revise_content
                      │                                               ↓
                      │                                        simulate_geo  (loop)
                      └─ "end"    → END

Stopping conditions (any one triggers "end"):
  - iteration >= max_iterations
  - coverage_history[-1] >= coverage_threshold
  - improvement from last cycle < min_improvement (plateau)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.pipeline_state import GEOState
from src.webpage_extractor import build_evidence_bundle
from src.schemas import (
    ProductFacts, ClaimAudit, ContentGaps,
    InventoryRow, EvaluationQuestions,
    GEOSimResult, VisibilityGap, RevisionStrategy,
)
from src.agents import (
    product_research_agent,
    markdown_writer_agent,
    claim_verifier_agent,
    gap_analysis_agent,
    geo_simulator_agent,
    visibility_analyzer_agent,
    revision_strategy_agent,
    content_reviser_agent,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _model(state: GEOState) -> str:
    return state.get("generator_model", "qwen2.5:latest")

def _reviewer(state: GEOState) -> str:
    return state.get("reviewer_model", "llama3.1:8b-instruct-q4_K_M")

def _token(state: GEOState) -> str:
    return state.get("hf_token", "")


# ---------------------------------------------------------------------------
# Layer 1 nodes
# ---------------------------------------------------------------------------

def extract_webpage(state: GEOState) -> dict:
    bundle = build_evidence_bundle(state["url"])
    return {"bundle": bundle}


def research_product(state: GEOState) -> dict:
    facts = product_research_agent.run(
        state["bundle"],
        product_line_hint=state.get("product_line_hint", ""),
        model=_model(state),
        token=_token(state),
    )
    return {"facts_dict": facts.model_dump(), "product_name": facts.product_name or "Unknown"}


def write_markdown(state: GEOState) -> dict:
    facts = ProductFacts.model_validate(state["facts_dict"])
    md = markdown_writer_agent.run(facts, model=_model(state), token=_token(state))
    return {"markdown": md}


def verify_claims(state: GEOState) -> dict:
    audit = claim_verifier_agent.run(
        state["bundle"],
        state["markdown"],
        model=_reviewer(state),
        token=_token(state),
    )
    return {"audit_dict": audit.model_dump()}


def analyze_gaps(state: GEOState) -> dict:
    facts = ProductFacts.model_validate(state["facts_dict"])
    audit = ClaimAudit.model_validate(state["audit_dict"])
    gaps  = gap_analysis_agent.run(facts, model=_model(state), token=_token(state))

    pname = state["product_name"]
    inv = InventoryRow(
        product_name=pname,
        product_line=facts.product_line,
        source_url=state["url"],
        missing_information_count=len(facts.missing_information),
        unsupported_claim_count=audit.summary.unsupported_claims,
    )
    eq = EvaluationQuestions(
        product_name=pname,
        questions=[
            f"What is {pname}?",
            f"Who is {pname} designed for?",
            f"What are the key ingredients in {pname}?",
            f"What makes {pname} different from similar products?",
            f"What formulation philosophy is described for {pname}?",
            f"What are the main use cases for {pname}?",
            f"What claims about {pname} require caution or internal review?",
            f"How would you summarize {pname} for an AI search query?",
        ],
    )
    return {
        "gaps_dict": gaps.model_dump(),
        "inventory_dict": inv.model_dump(),
        "eval_questions": eq.questions,
    }


# ---------------------------------------------------------------------------
# Layer 2 nodes
# ---------------------------------------------------------------------------

def simulate_geo(state: GEOState) -> dict:
    facts    = ProductFacts.model_validate(state["facts_dict"])
    markdown = state.get("revised_markdown") or state["markdown"]
    questions = state.get("eval_questions") or [f"What is {state['product_name']}?"]

    results = geo_simulator_agent.run(
        markdown=markdown,
        facts=facts,
        questions=questions,
        model=_model(state),
        token=_token(state),
    )
    avg_score   = sum(r.coverage_score for r in results) / len(results) if results else 0.0
    results_dicts = [r.model_dump() for r in results]

    update: dict = {
        "sim_results_current": results_dicts,
        "coverage_history": [round(avg_score, 2)],  # operator.add appends this
    }
    # Store the very first simulation as "before" for comparison
    if not state.get("sim_results_before"):
        update["sim_results_before"] = results_dicts

    return update


def analyze_visibility(state: GEOState) -> dict:
    facts       = ProductFacts.model_validate(state["facts_dict"])
    sim_results = [GEOSimResult.model_validate(r) for r in state["sim_results_current"]]
    gaps = visibility_analyzer_agent.run(
        facts=facts,
        sim_results=sim_results,
        model=_model(state),
        token=_token(state),
    )
    return {"visibility_gaps_list": [g.model_dump() for g in gaps]}


def select_strategies(state: GEOState) -> dict:
    gaps = [VisibilityGap.model_validate(g) for g in state["visibility_gaps_list"]]
    strategies = revision_strategy_agent.run(gaps=gaps, model=_model(state), token=_token(state))
    return {"strategies_list": [s.model_dump() for s in strategies]}


def revise_content(state: GEOState) -> dict:
    facts      = ProductFacts.model_validate(state["facts_dict"])
    strategies = [RevisionStrategy.model_validate(s) for s in state["strategies_list"]]
    current_md = state.get("revised_markdown") or state["markdown"]

    revised = content_reviser_agent.run(
        original_markdown=current_md,
        facts=facts,
        strategies=strategies,
        model=_model(state),
        token=_token(state),
    )
    return {
        "revised_markdown": revised,
        "iteration": state.get("iteration", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge — decides whether to loop or stop
# ---------------------------------------------------------------------------

def should_continue(state: GEOState) -> str:
    iteration = state.get("iteration", 0)
    history   = state.get("coverage_history", [])
    max_iter  = state.get("max_iterations", 3)
    threshold = state.get("coverage_threshold", 7.5)
    min_imp   = state.get("min_improvement", 0.5)

    # First simulation (no revision yet) — always enter the loop
    if iteration == 0:
        return "revise"

    # Hard stop
    if iteration >= max_iter:
        return "end"

    # Coverage target reached
    if history and history[-1] >= threshold:
        return "end"

    # Plateau — last revision barely helped
    if len(history) >= 2 and (history[-1] - history[-2]) < min_imp:
        return "end"

    return "revise"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(GEOState)

    # Register nodes
    builder.add_node("extract_webpage",   extract_webpage)
    builder.add_node("research_product",  research_product)
    builder.add_node("write_markdown",    write_markdown)
    builder.add_node("verify_claims",     verify_claims)
    builder.add_node("analyze_gaps",      analyze_gaps)
    builder.add_node("simulate_geo",      simulate_geo)
    builder.add_node("analyze_visibility", analyze_visibility)
    builder.add_node("select_strategies", select_strategies)
    builder.add_node("revise_content",    revise_content)

    # Layer 1 — sequential
    builder.set_entry_point("extract_webpage")
    builder.add_edge("extract_webpage",  "research_product")
    builder.add_edge("research_product", "write_markdown")
    builder.add_edge("write_markdown",   "verify_claims")
    builder.add_edge("verify_claims",    "analyze_gaps")
    builder.add_edge("analyze_gaps",     "simulate_geo")

    # Layer 2 — iterative loop
    builder.add_conditional_edges(
        "simulate_geo",
        should_continue,
        {"revise": "analyze_visibility", "end": END},
    )
    builder.add_edge("analyze_visibility", "select_strategies")
    builder.add_edge("select_strategies",  "revise_content")
    builder.add_edge("revise_content",     "simulate_geo")   # ← the cycle

    return builder.compile()


# Singleton — imported and reused by the Streamlit app
geo_graph = build_graph()


# ---------------------------------------------------------------------------
# Human-readable labels for UI progress display
# ---------------------------------------------------------------------------

NODE_LABELS: dict[str, str] = {
    "extract_webpage":    "Extracting webpage evidence",
    "research_product":   "Product Research Agent",
    "write_markdown":     "Markdown Writer Agent",
    "verify_claims":      "Claim Verification Agent",
    "analyze_gaps":       "Gap Analysis Agent",
    "simulate_geo":       "GEO Simulator",
    "analyze_visibility": "Visibility Analyzer",
    "select_strategies":  "Revision Strategy Selector",
    "revise_content":     "Content Reviser",
}
