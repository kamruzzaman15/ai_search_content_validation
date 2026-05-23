import os
from src.llm_client import call_llm
from src.schemas import ProductFacts


SYSTEM_PROMPT = """\
You are a product content writer for a marketing team. Your job is to convert structured product facts into a clean, factual markdown content file.

Rules:
- Write concise, clear, non-hyped prose.
- Do NOT invent claims not present in the product facts.
- If a section has no supporting facts, write: "Information not confirmed from source page."
- Mark any claims flagged for review with: *(claim requires internal confirmation)*
- Organize content for fast reading and reuse by the marketing team.
- Output ONLY the markdown content. No preamble, no explanation.
"""

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


def _build_prompt(facts: ProductFacts) -> str:
    facts_json = facts.model_dump_json(indent=2)
    sections = "\n".join(f"- {s}" for s in REQUIRED_SECTIONS)

    return f"""{SYSTEM_PROMPT}

PRODUCT FACTS (JSON):
{facts_json}

Write a markdown content file with ALL of the following sections (in order):
{sections}

Use the product name as the top-level H1 heading.

MARKDOWN:
"""


def run(facts: ProductFacts, model: str | None = None, token: str | None = None) -> str:
    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(facts)
    markdown = call_llm(prompt, model=model, token=token, max_new_tokens=2048, temperature=0.3)

    # Ensure the output starts with the product name heading
    if not markdown.strip().startswith("#"):
        markdown = f"# {facts.product_name}\n\n{markdown}"

    return markdown.strip()
