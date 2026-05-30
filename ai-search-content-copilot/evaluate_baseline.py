"""
Baseline comparison script — generates a one-shot markdown from the raw page
text and compares it against the agentic pipeline output side-by-side.

Usage:
    python3 evaluate_baseline.py <product_url>
"""
from __future__ import annotations
import sys
import os
import json
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(override=True)

from src.webpage_extractor import build_evidence_bundle
from src.llm_client import call_llm
from src.agents import product_research_agent, markdown_writer_agent, claim_verifier_agent

REQUIRED_SECTIONS = [
    "Product Overview",
    "Formulation Philosophy",
    "Key Ingredients",
    "Differentiators",
    "Use Cases",
    "Competitive Positioning",
    "Claims Requiring Review",
    "Open Questions for Internal Experts",
]

ONE_SHOT_PROMPT = """\
You are a product content writer. Read the following product webpage text and write a
clean markdown content file about the product. Include these sections:
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


def score_markdown(label: str, markdown: str, bundle: dict) -> dict:
    """Score a markdown file on 5 simple dimensions."""
    sections_found = sum(1 for s in REQUIRED_SECTIONS if s.lower() in markdown.lower())
    word_count = len(markdown.split())

    # Check evidence grounding: does the markdown quote or closely echo source text?
    source_words = set(bundle.get("raw_extracted_text", "").lower().split())
    md_words = set(markdown.lower().split())
    overlap = len(md_words & source_words) / max(len(md_words), 1)

    # Run claim audit via the reviewer model
    model = os.environ.get("REVIEWER_MODEL", "llama3.1:8b")
    audit = claim_verifier_agent.run(bundle, markdown, model=model)
    s = audit.summary

    unsupported_rate = (s.unsupported_claims / s.total_claims * 100) if s.total_claims else 100
    supported_rate   = (s.supported_claims  / s.total_claims * 100) if s.total_claims else 0

    result = {
        "system": label,
        "sections_found":     f"{sections_found}/{len(REQUIRED_SECTIONS)}",
        "word_count":          word_count,
        "vocab_overlap_%":    round(overlap * 100, 1),
        "total_claims":        s.total_claims,
        "supported_%":        round(supported_rate, 1),
        "unsupported_%":      round(unsupported_rate, 1),
        "needs_review":        s.needs_internal_confirmation,
    }
    return result, audit, markdown


def print_comparison(results: list[dict]):
    print(f"\n{'='*70}")
    print("EVALUATION COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("sections_found",  "Sections found"),
        ("word_count",      "Word count"),
        ("supported_%",     "Supported claims %"),
        ("unsupported_%",   "Unsupported claims %  (lower=better)"),
        ("needs_review",    "Needs internal review"),
    ]

    col_w = 26
    header = f"{'Metric':<32}" + "".join(f"{r['system']:<{col_w}}" for r in results)
    print(header)
    print("-" * (32 + col_w * len(results)))

    for key, label in metrics:
        row = f"{label:<32}" + "".join(f"{str(r[key]):<{col_w}}" for r in results)
        print(row)

    print(f"\n{'='*70}")
    print("WINNER BY METRIC")
    print(f"{'='*70}")

    # Supported % — higher is better
    best_sup = max(results, key=lambda r: r["supported_%"])
    print(f"  Best supported claim rate : {best_sup['system']} ({best_sup['supported_%']}%)")

    # Unsupported % — lower is better
    best_unsup = min(results, key=lambda r: r["unsupported_%"])
    print(f"  Lowest unsupported rate   : {best_unsup['system']} ({best_unsup['unsupported_%']}%)")

    # Sections — higher is better
    best_sec = max(results, key=lambda r: int(r["sections_found"].split("/")[0]))
    print(f"  Most sections covered     : {best_sec['system']} ({best_sec['sections_found']})")


def run(url: str):
    model = os.environ.get("GENERATOR_MODEL", "qwen2.5:7b-instruct")
    reviewer = os.environ.get("REVIEWER_MODEL", "llama3.1:8b")

    print(f"\n{'='*70}")
    print("AI Search Content Copilot — Baseline Comparison")
    print(f"URL     : {url}")
    print(f"Models  : generator={model}  reviewer={reviewer}")
    print(f"{'='*70}")

    # --- Step 1: Extract page ---
    print("\n[1/4] Extracting webpage...")
    bundle = build_evidence_bundle(url)
    raw_text = bundle.get("raw_extracted_text", "")[:4000]
    print(f"      → {len(bundle['paragraphs'])} paragraphs, {len(bundle['bullets'])} bullets")

    # --- Step 2: Baseline B — one-shot ---
    print("\n[2/4] Generating one-shot baseline markdown...")
    one_shot_md = call_llm(
        ONE_SHOT_PROMPT.format(text=raw_text),
        model=model,
        max_new_tokens=2048,
        temperature=0.3,
    )
    print(f"      → {len(one_shot_md.split())} words")

    # --- Step 3: Agentic pipeline ---
    print("\n[3/4] Running full agentic pipeline...")
    facts = product_research_agent.run(bundle, model=model)
    agentic_md = markdown_writer_agent.run(facts, model=model)
    print(f"      → Product: {facts.product_name} | {len(agentic_md.split())} words")

    # --- Step 4: Score both ---
    print("\n[4/4] Scoring both outputs with reviewer model...")
    print("      Scoring one-shot...")
    r_oneshot, audit_oneshot, _ = score_markdown("One-shot", one_shot_md, bundle)
    print("      Scoring agentic...")
    r_agentic, audit_agentic, _ = score_markdown("Agentic", agentic_md, bundle)

    # --- Print comparison ---
    print_comparison([r_oneshot, r_agentic])

    # --- Save outputs ---
    os.makedirs("outputs/eval", exist_ok=True)
    slug = url.rstrip("/").split("/")[-1] or "product"

    with open(f"outputs/eval/{slug}_oneshot.md", "w") as f:
        f.write(one_shot_md)
    with open(f"outputs/eval/{slug}_agentic.md", "w") as f:
        f.write(agentic_md)
    with open(f"outputs/eval/{slug}_oneshot_audit.json", "w") as f:
        json.dump(audit_oneshot.model_dump(), f, indent=2)
    with open(f"outputs/eval/{slug}_agentic_audit.json", "w") as f:
        json.dump(audit_agentic.model_dump(), f, indent=2)

    print(f"\nOutputs saved to outputs/eval/{slug}_*.md and *_audit.json")
    print("Open both .md files side-by-side to compare readability manually.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1])
