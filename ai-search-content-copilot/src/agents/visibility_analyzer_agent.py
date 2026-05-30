"""
Visibility Analyzer Agent

Compares GEO simulation results against the full product facts to surface
specific, prioritized content gaps. Each gap points to a section that needs
improvement and the question that exposed the weakness.
"""
from __future__ import annotations
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import GEOSimResult, ProductFacts, VisibilityGap


SYSTEM_PROMPT = """\
You are a content analyst evaluating how well a product markdown file supports AI-search retrieval.

You will receive:
1. The full structured product facts (ground truth of what is known about this product).
2. A set of simulated AI-search answers — each shows what a generative system actually surfaced from the content, and what it missed.

Your job is to identify specific, actionable visibility gaps:
- Facts that are known but not surfaced in any simulated answer.
- Sections where the simulated answer was vague, incomplete, or had a low coverage score.
- Specific claims that are ambiguous or poorly positioned for retrieval.
- Missing structure that would help a generative system find and use the right information.

Gap types (use exactly these strings):
  missing_fact        — a known fact that never appeared in any simulated answer
  weak_coverage       — a section produced a low-quality or incomplete simulated answer
  ambiguous_claim     — a claim that is present but phrased so vaguely it was ignored or misused
  no_differentiator   — the content fails to make clear what makes this product distinctive
  poor_structure      — a structural issue (e.g. key info is buried, no clear summary) hurting retrieval

Priority:
  high   — directly blocks a core customer question from being answered
  medium — weakens answer quality for an important question
  low    — minor improvement opportunity

Rules:
- Be specific. Name the section and the fact, not generic advice like "improve the content."
- Only flag gaps that are real problems, not hypothetical improvements.
- Return ONLY valid JSON. No text outside the JSON.
"""

SCHEMA_EXAMPLE = """[
  {
    "gap_type": "missing_fact | weak_coverage | ambiguous_claim | no_differentiator | poor_structure",
    "description": "specific description of the gap",
    "target_section": "the markdown section that should be improved",
    "priority": "high | medium | low",
    "triggered_by_question": "the customer question that exposed this gap"
  }
]"""


def _build_prompt(facts: ProductFacts, sim_results: list[GEOSimResult]) -> str:
    facts_json = facts.model_dump_json(indent=2)
    sim_block = "\n\n".join(
        f"Q: {r.question}\n"
        f"Simulated answer: {r.simulated_answer}\n"
        f"Facts surfaced: {r.key_facts_surfaced}\n"
        f"Facts missed: {r.key_facts_missed}\n"
        f"Coverage score: {r.coverage_score}/10"
        for r in sim_results
    )

    return f"""{SYSTEM_PROMPT}

--- PRODUCT FACTS (ground truth) ---
{facts_json}
--- END FACTS ---

--- SIMULATION RESULTS ---
{sim_block}
--- END SIMULATIONS ---

Identify all significant visibility gaps. Return a JSON array:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(
    facts: ProductFacts,
    sim_results: list[GEOSimResult],
    model: str | None = None,
    token: str | None = None,
) -> list[VisibilityGap]:
    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(facts, sim_results)
    raw = call_llm(prompt, model=model, token=token, max_new_tokens=1536, temperature=0.2)

    try:
        data = parse_json_response(raw)
        if isinstance(data, dict) and "gaps" in data:
            data = data["gaps"]
        return [VisibilityGap.model_validate(item) for item in data]
    except Exception:
        # Fallback: derive basic gaps from simulation scores
        gaps: list[VisibilityGap] = []
        for r in sim_results:
            if r.coverage_score < 5:
                gaps.append(VisibilityGap(
                    gap_type="weak_coverage",
                    description=f"Low coverage score ({r.coverage_score}/10) for: {r.question}",
                    target_section="Product Overview",
                    priority="high",
                    triggered_by_question=r.question,
                ))
        return gaps
