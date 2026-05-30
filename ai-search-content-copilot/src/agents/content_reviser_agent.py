"""
Content Reviser Agent

Applies selected revision strategies to produce an improved version of the
product markdown. Strictly evidence-grounded: it may only use facts already
present in the product facts JSON — it cannot invent new claims.
"""
from __future__ import annotations
import os
from src.llm_client import call_llm
from src.schemas import ProductFacts, RevisionStrategy


SYSTEM_PROMPT = """\
You are a product content editor applying targeted revisions to a product markdown file.

You will receive:
1. The original markdown draft.
2. The structured product facts (your only permitted source of new content).
3. A prioritized list of revision strategies — each one names a section, explains the problem, and gives a specific instruction.

Your job is to apply every revision instruction and produce the improved markdown.

HARD RULES:
- You may ONLY add or change content that is supported by the product facts JSON.
- Do NOT invent claims, statistics, comparisons, or details not present in the facts.
- Do NOT remove sections. Every original section must still appear in the revised output.
- If a strategy asks you to add a FAQ section and one does not exist, add it at the end.
- Preserve the H1 product name heading and all existing H2 section headings.
- Keep the prose concise, factual, and non-hyped.
- Output ONLY the revised markdown. No preamble, no explanation, no commentary.
"""


def _build_prompt(
    original_markdown: str,
    facts: ProductFacts,
    strategies: list[RevisionStrategy],
) -> str:
    facts_json = facts.model_dump_json(indent=2)
    strategy_block = "\n".join(
        f"{i+1}. [{s.strategy_name.upper()}] Section: {s.target_section}\n"
        f"   Rationale: {s.rationale}\n"
        f"   Instruction: {s.specific_instruction}"
        for i, s in enumerate(strategies)
    )

    return f"""{SYSTEM_PROMPT}

--- REVISION STRATEGIES (apply all of these) ---
{strategy_block}
--- END STRATEGIES ---

--- PRODUCT FACTS (your only permitted source of content) ---
{facts_json}
--- END FACTS ---

--- ORIGINAL MARKDOWN ---
{original_markdown}
--- END ORIGINAL MARKDOWN ---

Apply all revision strategies and return the improved markdown:
"""


def run(
    original_markdown: str,
    facts: ProductFacts,
    strategies: list[RevisionStrategy],
    model: str | None = None,
    token: str | None = None,
) -> str:
    if not strategies:
        return original_markdown

    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(original_markdown, facts, strategies)
    revised = call_llm(prompt, model=model, token=token, max_new_tokens=3000, temperature=0.2)

    # Ensure output starts with a heading
    if not revised.strip().startswith("#"):
        revised = f"# {facts.product_name}\n\n{revised}"

    return revised.strip()
