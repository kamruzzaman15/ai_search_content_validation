"""
Revision Strategy Agent

Selects targeted, content-specific revision strategies based on identified
visibility gaps. Inspired by AutoGEO and AgenticGEO research but applied
responsibly: strategies improve clarity, structure, and findability —
not manipulation or fabrication.

Strategy inventory (maps to schema StrategyName literals):
  add_specificity          — replace vague claims with concrete, precise language
  surface_differentiators  — make unique selling points more prominent and findable
  add_faq                  — add a FAQ section for questions the content can already answer
  clarify_ambiguous        — rewrite ambiguous claims to be unambiguous and directly usable
  improve_use_case_depth   — expand use cases with specific context (who benefits, how, when)
  consolidate_evidence     — ensure key claims are paired with visible supporting context
  add_structured_summary   — add or improve a concise summary section for fast retrieval
"""
from __future__ import annotations
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import VisibilityGap, RevisionStrategy


SYSTEM_PROMPT = """\
You are a content strategy advisor for an AI-search optimization workflow.

You have been given a list of visibility gaps — specific problems identified when testing how well a product markdown file supports generative AI search answers.

Your job is to select targeted revision strategies to address these gaps. Each strategy must:
- Map to exactly one of the allowed strategy names (listed below).
- Target a specific section of the markdown.
- Include a concrete, actionable instruction for the writer — not generic advice.
- Be grounded in what the product facts already support. Do NOT suggest adding claims that were not in the original evidence.

Available strategy names (use exactly these strings):
  add_specificity          — replace vague or general language with specific, concrete facts
  surface_differentiators  — restructure or expand to make distinctive features more findable
  add_faq                  — add a FAQ block addressing questions the content can already answer
  clarify_ambiguous        — rewrite a specific claim or section to eliminate ambiguity
  improve_use_case_depth   — expand a use case with context about who benefits and how
  consolidate_evidence     — add visible supporting context alongside a key claim
  add_structured_summary   — add or improve a structured summary optimized for quick retrieval

Rules:
- Select only strategies that directly address a listed gap. Do not pad with generic suggestions.
- Prefer a small set of high-impact strategies (3–6) over a long list of weak ones.
- Each strategy must have a specific_instruction that tells the writer exactly what to do.
- Return ONLY valid JSON. No text outside the JSON.
"""

SCHEMA_EXAMPLE = """[
  {
    "strategy_name": "add_specificity | surface_differentiators | add_faq | clarify_ambiguous | improve_use_case_depth | consolidate_evidence | add_structured_summary",
    "target_section": "section name in the markdown",
    "rationale": "which gap this addresses and why this strategy fits",
    "specific_instruction": "concrete instruction: e.g. 'Replace the sentence X with specific percentages drawn from the Key Ingredients section'"
  }
]"""


def _build_prompt(gaps: list[VisibilityGap]) -> str:
    high = [g for g in gaps if g.priority == "high"]
    medium = [g for g in gaps if g.priority == "medium"]
    low = [g for g in gaps if g.priority == "low"]

    def fmt(g: VisibilityGap) -> str:
        return (
            f"- [{g.gap_type.upper()}] {g.description}\n"
            f"  Section: {g.target_section} | Priority: {g.priority}\n"
            f"  Exposed by: {g.triggered_by_question}"
        )

    gap_block = ""
    if high:
        gap_block += "HIGH PRIORITY GAPS:\n" + "\n".join(fmt(g) for g in high) + "\n\n"
    if medium:
        gap_block += "MEDIUM PRIORITY GAPS:\n" + "\n".join(fmt(g) for g in medium) + "\n\n"
    if low:
        gap_block += "LOW PRIORITY GAPS:\n" + "\n".join(fmt(g) for g in low)

    return f"""{SYSTEM_PROMPT}

--- VISIBILITY GAPS ---
{gap_block.strip()}
--- END GAPS ---

Select targeted revision strategies to address these gaps. Return a JSON array:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(
    gaps: list[VisibilityGap],
    model: str | None = None,
    token: str | None = None,
) -> list[RevisionStrategy]:
    if not gaps:
        return []

    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(gaps)
    raw = call_llm(prompt, model=model, token=token, max_new_tokens=1536, temperature=0.3)

    try:
        data = parse_json_response(raw)
        if isinstance(data, dict) and "strategies" in data:
            data = data["strategies"]
        return [RevisionStrategy.model_validate(item) for item in data]
    except Exception:
        # Fallback: one generic strategy per high-priority gap
        strategies: list[RevisionStrategy] = []
        for g in gaps:
            if g.priority == "high":
                strategies.append(RevisionStrategy(
                    strategy_name="add_specificity",
                    target_section=g.target_section,
                    rationale=f"Addresses high-priority gap: {g.description}",
                    specific_instruction=f"Review and improve the '{g.target_section}' section to address: {g.description}",
                ))
        return strategies
