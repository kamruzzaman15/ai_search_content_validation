import json
import os
import sys
from pathlib import Path
import pandas as pd

import streamlit as st
from dotenv import load_dotenv

# Make src importable when running from the project root
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

# Explicit path + override=True so .env always wins over stale shell vars
load_dotenv(dotenv_path=_root / ".env", override=True)

from src.webpage_extractor import build_evidence_bundle
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
from src.schemas import InventoryRow, EvaluationQuestions, GEOFeedbackReport
from src.pipeline_langgraph import geo_graph, NODE_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Search Content Copilot",
    page_icon="🔍",
    layout="wide",
)

st.title("AI Search Content Copilot")
st.caption("Agentic GEO prototype — product knowledge structuring with human-in-the-loop review")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    pipeline_mode = st.radio(
        "Pipeline Mode",
        ["Sequential", "LangGraph"],
        help=(
            "**Sequential** — two separate buttons: run Layer 1, then optionally Layer 2.\n\n"
            "**LangGraph** — one button runs both layers end-to-end with an iterative "
            "refinement loop that automatically stops when coverage plateaus."
        ),
    )
    using_langgraph = pipeline_mode == "LangGraph"

    if using_langgraph:
        st.caption("LangGraph settings")
        max_iterations = st.slider("Max revision iterations", 1, 5, 3,
                                   help="Hard stop on the Layer 2 refinement loop.")
        coverage_threshold = st.slider("Coverage threshold (stop early)", 5.0, 10.0, 7.5, 0.5,
                                       help="Stop iterating if avg coverage score reaches this.")
        min_improvement = st.slider("Min improvement per iteration", 0.1, 2.0, 0.5, 0.1,
                                    help="Stop iterating if last cycle gained less than this.")
        st.divider()

    backend = st.radio(
        "Model Backend",
        ["Ollama (local)", "Hugging Face"],
        index=0 if "/" not in os.environ.get("GENERATOR_MODEL", "gemma:latest") else 1,
        help="Ollama runs models locally. Hugging Face uses the Inference API.",
    )
    using_ollama = backend == "Ollama (local)"

    if using_ollama:
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        generator_model = st.text_input(
            "Generator Model",
            value=os.environ.get("GENERATOR_MODEL", "gemma:latest"),
            help="Ollama model name, e.g. gemma3:latest, llama3.1:8b, qwen3:8b",
        )
        reviewer_model = st.text_input(
            "Reviewer Model",
            value=os.environ.get("REVIEWER_MODEL", "gemma:latest"),
            help="Can be the same model or a different one for independent review",
        )
        hf_token = ""
    else:
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
            help="Your HF token with Inference API access",
        )
        generator_model = st.text_input(
            "Generator Model",
            value=os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B"),
        )
        reviewer_model = st.text_input(
            "Reviewer Model",
            value=os.environ.get("REVIEWER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        )
        ollama_base_url = ""

    st.divider()
    st.header("Product Input")

    product_url = st.text_input(
        "Product Page URL",
        placeholder="https://example.com/products/my-product",
    )
    product_line_hint = st.text_input(
        "Product Line (optional hint)",
        placeholder="e.g. Pet Supplements, Skincare, Nutrition...",
    )
    internal_notes = st.text_area(
        "Internal Notes (optional)",
        placeholder="Paste any internal context or clarifications here...",
        height=100,
    )

    run_label = "Run LangGraph Pipeline" if using_langgraph else "Run Agentic GEO Workflow"
    run_button = st.button(run_label, type="primary", use_container_width=True)

    if not using_langgraph:
        st.divider()
        st.header("GEO Feedback Loop")
        st.caption("Layer 2: simulate visibility, identify gaps, apply targeted revisions.")
        geo_button = st.button(
            "Run GEO Feedback Loop",
            type="secondary",
            use_container_width=True,
            help="Run after the agentic workflow completes.",
        )
    else:
        geo_button = False  # integrated into LangGraph run

    st.divider()
    st.header("Baseline Comparison")
    st.caption("Compare agentic pipeline vs simple one-shot LLM on the same URL.")
    baseline_button = st.button(
        "Run Baseline Comparison",
        type="secondary",
        use_container_width=True,
        help="Run after the agentic workflow completes.",
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key in ("bundle", "facts", "markdown", "audit", "gaps", "inventory", "eval_questions", "error",
            "geo_report", "geo_error", "baseline", "baseline_error"):
    if key not in st.session_state:
        st.session_state[key] = None

if "run_history" not in st.session_state:
    st.session_state.run_history = []  # list of dicts, one per completed product run

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LangGraph pipeline execution
# ---------------------------------------------------------------------------

if run_button and using_langgraph:
    if not product_url.strip():
        st.error("Please enter a product page URL.")
    elif not using_ollama and not hf_token.strip():
        st.error("Please enter your Hugging Face token in the sidebar.")
    else:
        if using_ollama:
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        else:
            os.environ["HF_TOKEN"] = hf_token
        os.environ["GENERATOR_MODEL"] = generator_model
        os.environ["REVIEWER_MODEL"]  = reviewer_model

        st.session_state.error = None
        st.session_state.geo_error = None

        initial_state: dict = {
            "url":                product_url.strip(),
            "product_line_hint":  product_line_hint,
            "generator_model":    generator_model,
            "reviewer_model":     reviewer_model,
            "hf_token":           hf_token,
            "max_iterations":     max_iterations,
            "coverage_threshold": coverage_threshold,
            "min_improvement":    min_improvement,
            # layer 1 placeholders
            "bundle": {}, "facts_dict": {}, "product_name": "",
            "markdown": "", "audit_dict": {}, "gaps_dict": {},
            "inventory_dict": {}, "eval_questions": [],
            # layer 2 placeholders
            "sim_results_before": [], "sim_results_current": [],
            "visibility_gaps_list": [], "strategies_list": [],
            "revised_markdown": "",
            "coverage_history": [],
            "iteration": 0,
        }

        with st.status("Running LangGraph pipeline...", expanded=True) as lg_status:
            try:
                # Stream node-by-node so we can show live progress
                final: dict = dict(initial_state)
                for chunk in geo_graph.stream(
                    initial_state,
                    config={"recursion_limit": 100},
                ):
                    for node_name, updates in chunk.items():
                        if node_name == "__end__":
                            continue
                        label = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())

                        # Accumulate coverage_history manually (operator.add field)
                        if "coverage_history" in updates:
                            existing = final.get("coverage_history", [])
                            final["coverage_history"] = existing + updates["coverage_history"]
                            updates = {k: v for k, v in updates.items() if k != "coverage_history"}

                        final.update(updates)

                        # Build status message
                        extra = ""
                        if node_name == "simulate_geo":
                            h = final.get("coverage_history", [])
                            if h:
                                extra = f" — coverage: {h[-1]}/10"
                                if len(h) > 1:
                                    extra += f" (was {h[-2]}/10)"
                        if node_name == "revise_content":
                            extra = f" — iteration {final.get('iteration', '?')}"
                        st.write(f"✓ {label}{extra}")

                # ── Map LangGraph final state → session_state ──────────────
                from src.schemas import (
                    ProductFacts, ClaimAudit, ContentGaps,
                    InventoryRow, EvaluationQuestions, GEOFeedbackReport,
                    GEOSimResult, VisibilityGap, RevisionStrategy,
                )

                facts   = ProductFacts.model_validate(final["facts_dict"])
                audit   = ClaimAudit.model_validate(final["audit_dict"])
                gaps    = ContentGaps.model_validate(final["gaps_dict"])
                inv     = InventoryRow.model_validate(final["inventory_dict"])
                eq      = EvaluationQuestions(product_name=facts.product_name,
                                              questions=final["eval_questions"])

                h = final.get("coverage_history", [])
                score_before = h[0]  if h else 0.0
                score_after  = h[-1] if h else 0.0

                geo_report = GEOFeedbackReport(
                    product_name=facts.product_name,
                    sim_results=[GEOSimResult.model_validate(r)
                                 for r in final.get("sim_results_before", [])],
                    sim_results_after=[GEOSimResult.model_validate(r)
                                       for r in final.get("sim_results_current", [])],
                    visibility_gaps=[VisibilityGap.model_validate(g)
                                     for g in final.get("visibility_gaps_list", [])],
                    selected_strategies=[RevisionStrategy.model_validate(s)
                                         for s in final.get("strategies_list", [])],
                    revised_markdown=final.get("revised_markdown", ""),
                    overall_coverage_score_before=score_before,
                    overall_coverage_score_after=score_after,
                )

                st.session_state.bundle        = final["bundle"]
                st.session_state.facts         = facts
                st.session_state.markdown      = final["markdown"]
                st.session_state.audit         = audit
                st.session_state.gaps          = gaps
                st.session_state.inventory     = inv
                st.session_state.eval_questions = eq
                st.session_state.geo_report    = geo_report

                # Auto-add to run history
                SECTIONS = ["Product Overview","Formulation Philosophy","Key Ingredients",
                            "Differentiators","Use Cases","Competitive Positioning",
                            "Claims Requiring Review","Open Questions for Internal Experts"]
                md = final["markdown"]
                s  = audit.summary
                history_row = {
                    "Product":            facts.product_name or "Unknown",
                    "URL":                product_url.strip(),
                    "Sections":           f"{sum(1 for sec in SECTIONS if sec.lower() in md.lower())}/8",
                    "Total Claims":       s.total_claims,
                    "Supported %":        round(s.supported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                    "Unsupported %":      round(s.unsupported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                    "Missing Info":       len(facts.missing_information),
                    "GEO Coverage Before": round(score_before, 1),
                    "GEO Coverage After":  round(score_after, 1),
                    "GEO Delta":           f"{score_after - score_before:+.1f}",
                }
                st.session_state.run_history = [
                    r for r in st.session_state.run_history
                    if r["URL"] != product_url.strip()
                ]
                st.session_state.run_history.append(history_row)

                total_iter = final.get("iteration", 0)
                lg_status.update(
                    label=f"LangGraph complete — {total_iter} revision cycle(s), "
                          f"coverage {score_before:.1f} → {score_after:.1f}/10",
                    state="complete",
                )

            except Exception as exc:
                import traceback
                st.session_state.error = str(exc)
                lg_status.update(label=f"LangGraph error: {exc}", state="error")
                st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Sequential pipeline execution (original path — unchanged)
# ---------------------------------------------------------------------------

if run_button and not using_langgraph:
    if not product_url.strip():
        st.error("Please enter a product page URL.")
    elif not using_ollama and not hf_token.strip():
        st.error("Please enter your Hugging Face token in the sidebar.")
    else:
        if using_ollama:
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        else:
            os.environ["HF_TOKEN"] = hf_token
        os.environ["GENERATOR_MODEL"] = generator_model
        os.environ["REVIEWER_MODEL"] = reviewer_model

        st.session_state.error = None

        with st.status("Running agentic workflow...", expanded=True) as status:
            try:
                st.write("Extracting webpage evidence...")
                bundle = build_evidence_bundle(product_url.strip())
                if internal_notes.strip():
                    bundle["internal_notes"] = internal_notes.strip()
                st.session_state.bundle = bundle
                st.write(f"Extracted {len(bundle.get('paragraphs', []))} paragraphs, "
                         f"{len(bundle.get('bullets', []))} bullets.")

                st.write("Running Product Research Agent...")
                facts = product_research_agent.run(
                    bundle,
                    product_line_hint=product_line_hint,
                    model=generator_model,
                    token=hf_token,
                )
                st.session_state.facts = facts
                st.write(f"Extracted facts for: {facts.product_name or 'Unknown product'}")

                st.write("Running Markdown Writer Agent...")
                markdown = markdown_writer_agent.run(facts, model=generator_model, token=hf_token)
                st.session_state.markdown = markdown
                st.write("Markdown draft generated.")

                st.write("Running Claim Verification Agent...")
                audit = claim_verifier_agent.run(bundle, markdown, model=reviewer_model, token=hf_token)
                st.session_state.audit = audit
                st.write(f"Audited {audit.summary.total_claims} claims — "
                         f"{audit.summary.supported_claims} supported, "
                         f"{audit.summary.unsupported_claims} unsupported.")

                st.write("Running Gap Analysis Agent...")
                gaps = gap_analysis_agent.run(facts, model=generator_model, token=hf_token)
                st.session_state.gaps = gaps
                st.write(f"Identified {len(gaps.recommended_internal_questions)} expert questions.")

                # Build inventory row
                inv = InventoryRow(
                    product_name=facts.product_name,
                    product_line=facts.product_line or product_line_hint,
                    source_url=product_url.strip(),
                    missing_information_count=len(facts.missing_information),
                    unsupported_claim_count=audit.summary.unsupported_claims,
                )
                st.session_state.inventory = inv

                # Accumulate into run history for multi-product comparison
                s = audit.summary
                history_row = {
                    "Product": facts.product_name or "Unknown",
                    "URL": product_url.strip(),
                    "Sections": f"{sum(1 for sec in ['Product Overview','Formulation Philosophy','Key Ingredients','Differentiators','Use Cases','Competitive Positioning','Claims Requiring Review','Open Questions for Internal Experts'] if sec.lower() in markdown.lower())}/8",
                    "Total Claims": s.total_claims,
                    "Supported %": round(s.supported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                    "Unsupported %": round(s.unsupported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                    "Missing Info": len(facts.missing_information),
                    "GEO Coverage Before": "—",
                    "GEO Coverage After": "—",
                    "GEO Delta": "—",
                }
                # Avoid duplicate entries for same URL
                st.session_state.run_history = [
                    r for r in st.session_state.run_history
                    if r["URL"] != product_url.strip()
                ]
                st.session_state.run_history.append(history_row)

                # Build evaluation questions
                eq = EvaluationQuestions(
                    product_name=facts.product_name,
                    questions=[
                        f"What is {facts.product_name}?",
                        f"Who is {facts.product_name} designed for?",
                        f"What are the key ingredients in {facts.product_name}?",
                        f"What makes {facts.product_name} different from similar products?",
                        f"What formulation philosophy is described for {facts.product_name}?",
                        f"What are the main use cases for {facts.product_name}?",
                        f"What claims about {facts.product_name} require caution or internal review?",
                        f"How would you summarize {facts.product_name} for an AI search query?",
                    ],
                )
                st.session_state.eval_questions = eq

                status.update(label="Workflow complete!", state="complete")

            except Exception as exc:
                st.session_state.error = str(exc)
                status.update(label=f"Error: {exc}", state="error")

if st.session_state.error:
    st.error(f"Pipeline error: {st.session_state.error}")

# ---------------------------------------------------------------------------
# GEO Feedback Loop execution (Layer 2)
# ---------------------------------------------------------------------------

if geo_button:
    if not st.session_state.markdown or not st.session_state.facts:
        st.warning("Run the Agentic GEO Workflow first to generate the initial content.")
    else:
        st.session_state.geo_error = None
        facts = st.session_state.facts
        markdown = st.session_state.markdown
        eval_questions = st.session_state.eval_questions

        with st.status("Running GEO Feedback Loop...", expanded=True) as geo_status:
            try:
                questions = eval_questions.questions if eval_questions else [
                    f"What is {facts.product_name}?",
                    f"What makes {facts.product_name} different?",
                    f"What are the key ingredients in {facts.product_name}?",
                    f"Who is {facts.product_name} designed for?",
                ]

                st.write("Simulating generative search answers...")
                sim_results = geo_simulator_agent.run(
                    markdown=markdown,
                    facts=facts,
                    questions=questions,
                    model=generator_model,
                    token=hf_token,
                )
                avg_score_before = (
                    sum(r.coverage_score for r in sim_results) / len(sim_results)
                    if sim_results else 0.0
                )
                st.write(f"Simulated {len(sim_results)} questions. Avg coverage: {avg_score_before:.1f}/10")

                st.write("Analyzing visibility gaps...")
                vis_gaps = visibility_analyzer_agent.run(
                    facts=facts,
                    sim_results=sim_results,
                    model=generator_model,
                    token=hf_token,
                )
                high_gaps = sum(1 for g in vis_gaps if g.priority == "high")
                st.write(f"Found {len(vis_gaps)} gaps ({high_gaps} high priority).")

                st.write("Selecting revision strategies...")
                strategies = revision_strategy_agent.run(
                    gaps=vis_gaps,
                    model=generator_model,
                    token=hf_token,
                )
                st.write(f"Selected {len(strategies)} targeted revision strategies.")

                st.write("Applying revisions...")
                revised_markdown = content_reviser_agent.run(
                    original_markdown=markdown,
                    facts=facts,
                    strategies=strategies,
                    model=generator_model,
                    token=hf_token,
                )

                # Re-simulate with revised content to measure improvement
                st.write("Re-simulating to measure improvement...")
                sim_results_after = geo_simulator_agent.run(
                    markdown=revised_markdown,
                    facts=facts,
                    questions=questions,
                    model=generator_model,
                    token=hf_token,
                )
                avg_score_after = (
                    sum(r.coverage_score for r in sim_results_after) / len(sim_results_after)
                    if sim_results_after else 0.0
                )

                report = GEOFeedbackReport(
                    product_name=facts.product_name,
                    sim_results=sim_results,
                    sim_results_after=sim_results_after,
                    visibility_gaps=vis_gaps,
                    selected_strategies=strategies,
                    revised_markdown=revised_markdown,
                    overall_coverage_score_before=avg_score_before,
                    overall_coverage_score_after=avg_score_after,
                )
                st.session_state.geo_report = report

                # Patch run_history with GEO scores
                current_url = product_url.strip() if product_url else ""
                for row in st.session_state.run_history:
                    if row["URL"] == current_url:
                        row["GEO Coverage Before"] = round(avg_score_before, 1)
                        row["GEO Coverage After"]  = round(avg_score_after, 1)
                        row["GEO Delta"]           = f"{avg_score_after - avg_score_before:+.1f}"

                geo_status.update(
                    label=f"GEO Feedback Loop complete! Coverage: {avg_score_before:.1f} → {avg_score_after:.1f}/10",
                    state="complete",
                )

            except Exception as exc:
                st.session_state.geo_error = str(exc)
                geo_status.update(label=f"GEO loop error: {exc}", state="error")

if st.session_state.geo_error:
    st.error(f"GEO Feedback Loop error: {st.session_state.geo_error}")

# ---------------------------------------------------------------------------
# Baseline Comparison execution
# ---------------------------------------------------------------------------

ONE_SHOT_PROMPT = """\
You are a product content writer. Read the following product webpage text and write a
clean markdown content file. Include ALL of these sections:
- Product Overview
- Formulation Philosophy
- Key Ingredients
- Differentiators
- Use Cases
- Competitive Positioning
- Claims Requiring Review
- Open Questions for Internal Experts

Write only the markdown. No preamble.

PRODUCT PAGE TEXT:
{text}

MARKDOWN:
"""

REQUIRED_SECTIONS = [
    "Product Overview", "Formulation Philosophy", "Key Ingredients",
    "Differentiators", "Use Cases", "Competitive Positioning",
    "Claims Requiring Review", "Open Questions for Internal Experts",
]

if baseline_button:
    if not st.session_state.bundle or not st.session_state.markdown:
        st.warning("Run the Agentic GEO Workflow first.")
    else:
        from src.llm_client import call_llm
        st.session_state.baseline_error = None

        with st.status("Running Baseline Comparison...", expanded=True) as bl_status:
            try:
                bundle   = st.session_state.bundle
                agentic_md = st.session_state.markdown

                st.write("Generating one-shot baseline markdown...")
                raw_text = bundle.get("raw_extracted_text", "")[:4000]
                oneshot_md = call_llm(
                    ONE_SHOT_PROMPT.format(text=raw_text),
                    model=generator_model,
                    token=hf_token,
                    max_new_tokens=2048,
                    temperature=0.3,
                )

                st.write("Auditing one-shot claims...")
                audit_oneshot = claim_verifier_agent.run(bundle, oneshot_md, model=reviewer_model, token=hf_token)

                st.write("Auditing agentic claims...")
                audit_agentic = st.session_state.audit  # already computed — reuse it

                def _score_md(label, md, audit):
                    s = audit.summary
                    return {
                        "System": label,
                        "Sections Found": f"{sum(1 for sec in REQUIRED_SECTIONS if sec.lower() in md.lower())}/8",
                        "Word Count": len(md.split()),
                        "Total Claims": s.total_claims,
                        "Supported %": round(s.supported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                        "Unsupported %": round(s.unsupported_claims / s.total_claims * 100, 1) if s.total_claims else 0,
                        "Needs Review": s.needs_internal_confirmation,
                    }

                st.session_state.baseline = {
                    "oneshot_md":    oneshot_md,
                    "agentic_md":    agentic_md,
                    "audit_oneshot": audit_oneshot,
                    "audit_agentic": audit_agentic,
                    "scores": [
                        _score_md("One-shot LLM", oneshot_md, audit_oneshot),
                        _score_md("Agentic Pipeline", agentic_md, audit_agentic),
                    ],
                }
                bl_status.update(label="Baseline comparison complete!", state="complete")
            except Exception as exc:
                st.session_state.baseline_error = str(exc)
                bl_status.update(label=f"Error: {exc}", state="error")

if st.session_state.baseline_error:
    st.error(f"Baseline comparison error: {st.session_state.baseline_error}")

# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------

if st.session_state.bundle:
    bundle = st.session_state.bundle
    facts = st.session_state.facts
    markdown = st.session_state.markdown
    audit = st.session_state.audit
    gaps = st.session_state.gaps
    inventory = st.session_state.inventory
    eval_questions = st.session_state.eval_questions

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "Extracted Evidence",
        "Product Facts",
        "Markdown Draft",
        "Claim Audit",
        "Content Gaps",
        "Inventory Row",
        "Evaluation Questions",
        "GEO Simulation",
        "Visibility Gaps",
        "Revision Strategies",
        "Revised Content",
        "Baseline Comparison",
        "Product History",
    ])

    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Extracted Evidence Bundle")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Paragraphs", len(bundle.get("paragraphs", [])))
            st.metric("Bullets", len(bundle.get("bullets", [])))
        with col2:
            st.metric("Headings", len(bundle.get("headings", [])))
            st.metric("Tables", len(bundle.get("tables", [])))

        with st.expander("Page Title & Meta"):
            st.write(f"**Title:** {bundle.get('page_title', '')}")
            st.write(f"**Meta Description:** {bundle.get('meta_description', '')}")

        with st.expander("Headings"):
            for h in bundle.get("headings", []):
                st.write(f"- {h}")

        with st.expander("Paragraphs (first 10)"):
            for p in bundle.get("paragraphs", [])[:10]:
                st.write(p)
                st.divider()

        with st.expander("Bullets (first 30)"):
            for b in bundle.get("bullets", [])[:30]:
                st.write(f"- {b}")

        with st.expander("Full Evidence JSON"):
            display_bundle = {k: v for k, v in bundle.items() if k != "raw_extracted_text"}
            st.json(display_bundle)

    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Structured Product Facts")
        if facts:
            st.write(f"**Product:** {facts.product_name}")
            st.write(f"**Line:** {facts.product_line}")
            st.write(f"**Category:** {facts.product_category}")
            st.write(f"**Summary:** {facts.short_summary}")

            with st.expander("Formulation Philosophy"):
                st.write(facts.formulation_philosophy.summary or "Not found.")
                for e in facts.formulation_philosophy.evidence:
                    st.caption(f"> {e}")

            with st.expander(f"Key Ingredients ({len(facts.key_ingredients)})"):
                for ing in facts.key_ingredients:
                    st.markdown(f"**{ing.ingredient}** — {ing.stated_role}")
                    for e in ing.evidence:
                        st.caption(f"> {e}")

            with st.expander(f"Differentiators ({len(facts.differentiators)})"):
                for d in facts.differentiators:
                    st.markdown(f"- {d.point}")
                    for e in d.evidence:
                        st.caption(f"> {e}")

            with st.expander(f"Use Cases ({len(facts.use_cases)})"):
                for u in facts.use_cases:
                    st.markdown(f"- {u.point}")

            if facts.claims_needing_review:
                with st.expander(f"Claims Needing Review ({len(facts.claims_needing_review)})"):
                    for c in facts.claims_needing_review:
                        st.warning(c)

            if facts.missing_information:
                with st.expander(f"Missing Information ({len(facts.missing_information)})"):
                    for m in facts.missing_information:
                        st.info(m)

            with st.expander("Full JSON"):
                st.json(facts.model_dump())

    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Generated Markdown Draft")
        if markdown:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Preview**")
                st.markdown(markdown)
            with col_b:
                st.markdown("**Raw Markdown**")
                st.code(markdown, language="markdown")
            st.download_button(
                "Download Markdown",
                data=markdown,
                file_name=f"{(facts.product_name or 'product').replace(' ', '_').lower()}_context.md",
                mime="text/markdown",
            )

    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Claim Audit Report")
        if audit:
            s = audit.summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims", s.total_claims)
            col2.metric("Supported", s.supported_claims, delta=None)
            col3.metric("Unsupported", s.unsupported_claims)
            col4.metric("Needs Review", s.needs_internal_confirmation)

            if s.total_claims > 0:
                supported_pct = round(s.supported_claims / s.total_claims * 100)
                st.progress(supported_pct / 100, text=f"{supported_pct}% supported")

            STATUS_COLORS = {
                "supported": "normal",
                "partially_supported": "off",
                "unsupported": "inverse",
                "needs_internal_confirmation": "off",
            }

            for claim in audit.claims:
                icon = {"supported": "✅", "partially_supported": "⚠️", "unsupported": "❌",
                        "needs_internal_confirmation": "🔍"}.get(claim.status, "•")
                risk_badge = f"[{claim.risk_level.upper()}]" if claim.risk_level != "low" else ""
                with st.expander(f"{icon} {risk_badge} {claim.claim[:100]}"):
                    st.write(f"**Status:** {claim.status}")
                    st.write(f"**Risk Level:** {claim.risk_level}")
                    st.write(f"**Action:** {claim.recommended_action}")
                    if claim.evidence:
                        st.caption(f"Evidence: {claim.evidence}")

            with st.expander("Full Audit JSON"):
                st.json(audit.model_dump())

            st.download_button(
                "Download Claim Audit JSON",
                data=json.dumps(audit.model_dump(), indent=2),
                file_name="claim_audit.json",
                mime="application/json",
            )

    # ------------------------------------------------------------------
    with tab5:
        st.subheader("Content Gaps & Internal Questions")
        if gaps:
            if gaps.missing_sections:
                st.markdown("**Missing Sections**")
                for m in gaps.missing_sections:
                    st.warning(m)

            if gaps.unclear_claims:
                st.markdown("**Unclear Claims**")
                for c in gaps.unclear_claims:
                    st.info(c)

            if gaps.recommended_internal_questions:
                st.markdown("**Questions for Internal Experts**")
                for i, q in enumerate(gaps.recommended_internal_questions, 1):
                    st.write(f"{i}. {q}")

            if gaps.recommended_page_improvements:
                st.markdown("**Recommended Page Improvements**")
                for p in gaps.recommended_page_improvements:
                    st.write(f"- {p}")

            st.download_button(
                "Download Content Gaps JSON",
                data=json.dumps(gaps.model_dump(), indent=2),
                file_name="content_gaps.json",
                mime="application/json",
            )

    # ------------------------------------------------------------------
    with tab6:
        st.subheader("Content Inventory Row")
        if inventory:
            st.json(inventory.model_dump())
            st.download_button(
                "Download Inventory Row JSON",
                data=json.dumps(inventory.model_dump(), indent=2),
                file_name="inventory_row.json",
                mime="application/json",
            )

    # ------------------------------------------------------------------
    with tab7:
        st.subheader("Evaluation Questions")
        if eval_questions:
            st.write(f"Product: **{eval_questions.product_name}**")
            for i, q in enumerate(eval_questions.questions, 1):
                st.write(f"{i}. {q}")
            st.download_button(
                "Download Evaluation Questions JSON",
                data=json.dumps(eval_questions.model_dump(), indent=2),
                file_name="evaluation_questions.json",
                mime="application/json",
            )

    # ------------------------------------------------------------------
    with tab8:
        st.subheader("GEO Simulation — Before vs After Comparison")
        geo_report = st.session_state.geo_report
        if geo_report:
            # ── Summary metrics ──────────────────────────────────────────
            score_before = geo_report.overall_coverage_score_before
            score_after  = geo_report.overall_coverage_score_after
            delta        = score_after - score_before

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Coverage Before", f"{score_before:.1f} / 10")
            col2.metric("Avg Coverage After",  f"{score_after:.1f} / 10",
                        delta=f"{delta:+.1f}", delta_color="normal")
            col3.metric("Improvement", f"{delta:+.1f} pts",
                        delta_color="normal" if delta >= 0 else "inverse")

            st.caption(
                "**Coverage score (0–10):** how completely the markdown enabled a generative "
                "system to answer a realistic customer question. "
                "Higher = more facts surfaced, fewer facts missed. "
                "_This is an internal proxy — not a live search engine test._"
            )
            st.divider()

            # ── Per-question comparison table ────────────────────────────
            after_by_q = {r.question: r for r in geo_report.sim_results_after}

            st.markdown("#### Per-question breakdown")
            st.caption("The most important signal: which specific questions improved, stayed the same, or got worse.")

            import pandas as pd
            rows = []
            for r_before in geo_report.sim_results:
                r_after = after_by_q.get(r_before.question)
                score_a = r_after.coverage_score if r_after else None
                missed_before = len(r_before.key_facts_missed)
                missed_after  = len(r_after.key_facts_missed) if r_after else None
                rows.append({
                    "Question": r_before.question,
                    "Score Before": r_before.coverage_score,
                    "Score After":  score_a if score_a is not None else "—",
                    "Delta": (score_a - r_before.coverage_score) if score_a is not None else "—",
                    "Facts Missed Before": missed_before,
                    "Facts Missed After":  missed_after if missed_after is not None else "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Per-question deep dive ───────────────────────────────────
            st.markdown("#### Deep dive — expand any question")
            for r_before in geo_report.sim_results:
                r_after = after_by_q.get(r_before.question)
                score_a = r_after.coverage_score if r_after else None
                d = (score_a - r_before.coverage_score) if score_a is not None else 0
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
                label = (
                    f"{arrow} [{r_before.coverage_score}→{score_a}/10]  {r_before.question}"
                    if score_a is not None
                    else f"[{r_before.coverage_score}/10]  {r_before.question}"
                )
                with st.expander(label):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Before revision**")
                        st.write(r_before.simulated_answer)
                        if r_before.key_facts_missed:
                            st.markdown("*Facts missed:*")
                            for f in r_before.key_facts_missed:
                                st.error(f"✗ {f}")
                    with c2:
                        if r_after:
                            st.markdown("**After revision**")
                            st.write(r_after.simulated_answer)
                            if r_after.key_facts_missed:
                                st.markdown("*Facts still missed:*")
                                for f in r_after.key_facts_missed:
                                    st.warning(f"⚠ {f}")
                            else:
                                st.success("No facts missed after revision.")
                        else:
                            st.info("After-revision simulation not available for this question.")
        else:
            st.info("Run the **GEO Feedback Loop** in the sidebar to see the comparison.")

    # ------------------------------------------------------------------
    with tab9:
        st.subheader("Visibility Gaps")
        geo_report = st.session_state.geo_report
        if geo_report:
            high = [g for g in geo_report.visibility_gaps if g.priority == "high"]
            medium = [g for g in geo_report.visibility_gaps if g.priority == "medium"]
            low = [g for g in geo_report.visibility_gaps if g.priority == "low"]

            col1, col2, col3 = st.columns(3)
            col1.metric("High Priority", len(high))
            col2.metric("Medium Priority", len(medium))
            col3.metric("Low Priority", len(low))

            PRIORITY_ICONS = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for g in geo_report.visibility_gaps:
                icon = PRIORITY_ICONS.get(g.priority, "•")
                with st.expander(f"{icon} [{g.gap_type}] {g.description[:80]}"):
                    st.write(f"**Type:** {g.gap_type}")
                    st.write(f"**Target Section:** {g.target_section}")
                    st.write(f"**Priority:** {g.priority}")
                    st.write(f"**Description:** {g.description}")
                    if g.triggered_by_question:
                        st.caption(f"Exposed by: {g.triggered_by_question}")
        else:
            st.info("Run the **GEO Feedback Loop** in the sidebar to see visibility gaps.")

    # ------------------------------------------------------------------
    with tab10:
        st.subheader("Revision Strategies")
        geo_report = st.session_state.geo_report
        if geo_report:
            st.write(f"{len(geo_report.selected_strategies)} strategies selected.")
            STRATEGY_ICONS = {
                "add_specificity": "🎯",
                "surface_differentiators": "⭐",
                "add_faq": "❓",
                "clarify_ambiguous": "🔍",
                "improve_use_case_depth": "📖",
                "consolidate_evidence": "🔗",
                "add_structured_summary": "📋",
            }
            for s in geo_report.selected_strategies:
                icon = STRATEGY_ICONS.get(s.strategy_name, "•")
                with st.expander(f"{icon} [{s.strategy_name}] → {s.target_section}"):
                    st.write(f"**Strategy:** {s.strategy_name}")
                    st.write(f"**Section:** {s.target_section}")
                    st.write(f"**Rationale:** {s.rationale}")
                    st.info(f"**Instruction:** {s.specific_instruction}")
        else:
            st.info("Run the **GEO Feedback Loop** in the sidebar to see revision strategies.")

    # ------------------------------------------------------------------
    with tab11:
        st.subheader("Revised Content")
        geo_report = st.session_state.geo_report
        if geo_report and geo_report.revised_markdown:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Revised Markdown Preview**")
                st.markdown(geo_report.revised_markdown)
            with col2:
                st.markdown("**Raw Revised Markdown**")
                st.code(geo_report.revised_markdown, language="markdown")

            product_slug = (facts.product_name or "product").replace(" ", "_").lower()
            st.download_button(
                "Download Revised Markdown",
                data=geo_report.revised_markdown,
                file_name=f"{product_slug}_revised.md",
                mime="text/markdown",
            )
            st.download_button(
                "Download Full GEO Report JSON",
                data=geo_report.model_dump_json(indent=2),
                file_name=f"{product_slug}_geo_report.json",
                mime="application/json",
            )
        else:
            st.info("Run the **GEO Feedback Loop** in the sidebar to generate revised content.")

    # ------------------------------------------------------------------
    with tab12:
        st.subheader("Baseline Comparison: One-shot LLM vs Agentic Pipeline")
        bl = st.session_state.baseline
        if bl:
            st.markdown("#### Summary table")
            st.caption(
                "Key question: does the multi-step agentic pipeline produce more accurate, "
                "complete, and trustworthy content than simply asking the LLM to write from "
                "the raw page text in one shot?"
            )
            df_scores = pd.DataFrame(bl["scores"])
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

            # Highlight the winner on each metric
            s_oneshot = bl["scores"][0]
            s_agentic = bl["scores"][1]
            st.divider()
            st.markdown("#### Winner by metric")
            winners = []
            for metric, higher_better, label in [
                ("Supported %",    True,  "Best supported claim rate"),
                ("Unsupported %",  False, "Lowest unsupported rate (lower = better)"),
                ("Sections Found", True,  "Most sections covered"),
                ("Word Count",     True,  "Most content"),
            ]:
                v0 = s_oneshot[metric]
                v1 = s_agentic[metric]
                # parse fractions like "8/8"
                if isinstance(v0, str) and "/" in v0:
                    v0 = int(v0.split("/")[0])
                    v1 = int(v1.split("/")[0])
                if higher_better:
                    winner = "Agentic Pipeline" if v1 >= v0 else "One-shot LLM"
                else:
                    winner = "Agentic Pipeline" if v1 <= v0 else "One-shot LLM"
                winners.append({"Metric": label, "Winner": winner,
                                 "One-shot": s_oneshot[metric], "Agentic": s_agentic[metric]})
            st.dataframe(pd.DataFrame(winners), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### Side-by-side content")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**One-shot LLM**")
                with st.expander("Preview"):
                    st.markdown(bl["oneshot_md"])
                st.download_button("Download one-shot .md", bl["oneshot_md"],
                                   file_name="oneshot_baseline.md", mime="text/markdown")
            with col_b:
                st.markdown("**Agentic Pipeline**")
                with st.expander("Preview"):
                    st.markdown(bl["agentic_md"])
                st.download_button("Download agentic .md", bl["agentic_md"],
                                   file_name="agentic_output.md", mime="text/markdown")

            st.divider()
            st.markdown("#### Claim-level detail")
            col_a, col_b = st.columns(2)
            ICONS = {"supported": "✅", "partially_supported": "⚠️",
                     "unsupported": "❌", "needs_internal_confirmation": "🔍"}
            with col_a:
                st.markdown("**One-shot claims**")
                for c in bl["audit_oneshot"].claims:
                    st.write(f"{ICONS.get(c.status,'•')} {c.claim[:90]}")
            with col_b:
                st.markdown("**Agentic claims**")
                for c in bl["audit_agentic"].claims:
                    st.write(f"{ICONS.get(c.status,'•')} {c.claim[:90]}")
        else:
            st.info("Click **Run Baseline Comparison** in the sidebar after running the workflow.")

    # ------------------------------------------------------------------
    with tab13:
        st.subheader("Product History — Multi-product Comparison")
        history = st.session_state.run_history
        if history:
            st.caption(
                "Each row is one completed product run in this session. "
                "GEO columns populate after running the GEO Feedback Loop for that product."
            )
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            # Highlight best/worst unsupported rate
            if len(history) > 1:
                best = min(history, key=lambda r: r["Unsupported %"])
                worst = max(history, key=lambda r: r["Unsupported %"])
                st.success(f"Best unsupported claim rate: **{best['Product']}** ({best['Unsupported %']}%)")
                st.error(f"Highest unsupported claim rate: **{worst['Product']}** ({worst['Unsupported %']}%)")

            st.download_button(
                "Download Product History CSV",
                data=pd.DataFrame(history).to_csv(index=False),
                file_name="product_history.csv",
                mime="text/csv",
            )
        else:
            st.info("Run the workflow on at least one product URL to start building history. "
                    "Run on multiple products to compare them here.")

else:
    st.info("Enter a product URL in the sidebar and click **Run Agentic GEO Workflow** to begin.")
