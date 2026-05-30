"""
Quick evaluation script — runs the full pipeline on a product URL and
scores it across 4 checks from the README evaluation framework.

Usage:
    python3 evaluate.py <product_url>

Example:
    python3 evaluate.py https://lifesabundance.com/products/pet-supplements/hip-joint-support-chewable-tablets/
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from src.webpage_extractor import build_evidence_bundle
from src.agents import product_research_agent, markdown_writer_agent, claim_verifier_agent, gap_analysis_agent

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _score(label: str, result: bool, detail: str = ""):
    tag = PASS if result else FAIL
    line = f"  [{tag}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return result


def check_1_schema_validity(facts) -> bool:
    """All required top-level fields must be non-empty."""
    print("\n=== Check 1: Schema Validity ===")
    required = ["product_name", "product_category", "short_summary"]
    results = []
    for field in required:
        val = getattr(facts, field, None)
        results.append(_score(f"field '{field}' populated", bool(val), val[:80] if val else "EMPTY"))
    return all(results)


def check_2_field_coverage(facts) -> bool:
    """Key structured fields should have at least 1 entry."""
    print("\n=== Check 2: Field Coverage ===")
    checks = [
        ("key_ingredients", facts.key_ingredients),
        ("differentiators", facts.differentiators),
        ("use_cases", facts.use_cases),
        ("formulation_philosophy", facts.formulation_philosophy.summary),
    ]
    results = []
    for label, val in checks:
        if isinstance(val, list):
            ok = len(val) > 0
            detail = f"{len(val)} items found"
        else:
            ok = bool(val)
            detail = (val[:80] if val else "EMPTY")
        results.append(_score(label, ok, detail))
    return all(results)


def check_3_evidence_grounding(facts) -> bool:
    """At least 70% of ingredients and differentiators should have evidence snippets."""
    print("\n=== Check 3: Evidence Grounding ===")
    items = facts.key_ingredients + facts.differentiators
    if not items:
        _score("evidence grounding", False, "no items to check")
        return False
    grounded = sum(1 for i in items if getattr(i, "evidence", []))
    pct = grounded / len(items) * 100
    ok = pct >= 70
    return _score(f"{grounded}/{len(items)} items have evidence snippets ({pct:.0f}%)", ok)


def check_4_claim_audit(audit) -> bool:
    """Unsupported claim rate should be below 20%."""
    print("\n=== Check 4: Claim Audit Sanity ===")
    s = audit.summary
    _score("claims were audited", s.total_claims > 0, f"{s.total_claims} total claims")
    if s.total_claims == 0:
        return False
    unsupported_rate = s.unsupported_claims / s.total_claims * 100
    supported_rate = s.supported_claims / s.total_claims * 100
    _score(f"supported rate ≥ 60%  ({supported_rate:.0f}%)", supported_rate >= 60)
    ok = unsupported_rate <= 20
    return _score(f"unsupported rate ≤ 20%  ({unsupported_rate:.0f}%)", ok)


def run_evaluation(url: str):
    model = os.environ.get("GENERATOR_MODEL", "gemma:latest")
    reviewer = os.environ.get("REVIEWER_MODEL", "gemma:latest")

    print(f"\n{'='*60}")
    print(f"AI Search Content Copilot — Evaluation Run")
    print(f"URL:   {url}")
    print(f"Model: {model}  |  Reviewer: {reviewer}")
    print(f"{'='*60}")

    print("\n[1/4] Fetching and extracting webpage...")
    bundle = build_evidence_bundle(url)
    print(f"      → {len(bundle['paragraphs'])} paragraphs, {len(bundle['bullets'])} bullets, "
          f"{len(bundle['headings'])} headings")

    print("\n[2/4] Running Product Research Agent...")
    facts = product_research_agent.run(bundle, model=model)
    print(f"      → Product: {facts.product_name}")

    print("\n[3/4] Running Markdown Writer Agent...")
    markdown = markdown_writer_agent.run(facts, model=model)
    print(f"      → Draft: {len(markdown)} chars")

    print("\n[4/4] Running Claim Verification Agent...")
    audit = claim_verifier_agent.run(bundle, markdown, model=reviewer)
    print(f"      → Audited {audit.summary.total_claims} claims")

    # --- Scores ---
    r1 = check_1_schema_validity(facts)
    r2 = check_2_field_coverage(facts)
    r3 = check_3_evidence_grounding(facts)
    r4 = check_4_claim_audit(audit)

    passed = sum([r1, r2, r3, r4])
    total = 4
    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} checks passed")
    print(f"{'='*60}")

    # Save outputs for manual review
    os.makedirs("outputs/eval", exist_ok=True)
    slug = url.rstrip("/").split("/")[-1] or "product"
    with open(f"outputs/eval/{slug}_facts.json", "w") as f:
        json.dump(facts.model_dump(), f, indent=2)
    with open(f"outputs/eval/{slug}_draft.md", "w") as f:
        f.write(markdown)
    with open(f"outputs/eval/{slug}_audit.json", "w") as f:
        json.dump(audit.model_dump(), f, indent=2)
    print(f"\nOutputs saved to outputs/eval/{slug}_*.json/md for manual review.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_evaluation(sys.argv[1])
