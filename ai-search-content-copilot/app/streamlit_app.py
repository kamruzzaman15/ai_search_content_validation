import json
import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Make src importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from src.webpage_extractor import build_evidence_bundle
from src.agents import (
    product_research_agent,
    markdown_writer_agent,
    claim_verifier_agent,
    gap_analysis_agent,
)
from src.schemas import InventoryRow, EvaluationQuestions

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Search Content Copilot",
    page_icon="🔍",
    layout="wide",
)

st.title("AI Search Content Copilot")
st.caption("Paste any product webpage URL — the agentic workflow extracts facts, drafts content, audits claims, and flags gaps.")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

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

    st.divider()
    st.header("Product Input")

    product_url = st.text_input(
        "Product Page URL",
        placeholder="https://example.com/product-page",
    )
    product_line_hint = st.selectbox(
        "Product Line (optional hint)",
        ["", "Health & Wellness", "Food & Beverage", "Beauty & Skincare", "Pet Products", "Household & Cleaning", "Electronics", "Apparel", "Other"],
    )
    internal_notes = st.text_area(
        "Internal Notes (optional)",
        placeholder="Paste any internal context or clarifications here...",
        height=100,
    )

    run_button = st.button("Run Agentic GEO Workflow", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key in ("bundle", "facts", "markdown", "audit", "gaps", "inventory", "eval_questions", "error"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button:
    if not product_url.strip():
        st.error("Please enter a product page URL.")
    elif not hf_token.strip():
        st.error("Please enter your Hugging Face token in the sidebar.")
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Extracted Evidence",
        "Product Facts",
        "Markdown Draft",
        "Claim Audit",
        "Content Gaps",
        "Inventory Row",
        "Evaluation Questions",
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

else:
    st.info("Paste any public product page URL in the sidebar and click **Run Agentic GEO Workflow** to begin.")
