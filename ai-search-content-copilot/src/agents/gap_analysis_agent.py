from __future__ import annotations
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import ContentGaps, ProductFacts


SYSTEM_PROMPT = """\
You are a content strategy assistant helping a marketing team identify gaps in product knowledge files.

Your job is to analyze a structured product facts JSON and identify:
1. Sections that are missing or empty.
2. Claims that are vague or need clarification.
3. Questions a content team should ask internal subject-matter experts.
4. Improvements the product webpage itself could make.

Rules:
- Be specific. Do not give generic advice.
- Frame questions as things an intern could bring to a product expert meeting.
- Return ONLY valid JSON. No explanation outside the JSON.
"""

SCHEMA_EXAMPLE = """{
  "missing_sections": ["list of section names with no content"],
  "unclear_claims": ["list of specific vague or ambiguous claims"],
  "recommended_internal_questions": [
    "What formulation choices intentionally differentiate this from generic products?",
    "Which ingredients are central to the product story and why?",
    "Are there approved competitive comparisons we should include?"
  ],
  "recommended_page_improvements": ["list of specific webpage improvements"]
}"""


def _build_prompt(facts: ProductFacts) -> str:
    facts_json = facts.model_dump_json(indent=2)

    return f"""{SYSTEM_PROMPT}

PRODUCT FACTS:
{facts_json}

Identify all gaps, unclear claims, and generate internal expert interview questions.

Return JSON matching this schema:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(facts: ProductFacts, model: str | None = None, token: str | None = None) -> ContentGaps:
    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(facts)
    raw = call_llm(prompt, model=model, token=token, max_new_tokens=1024, temperature=0.3)

    try:
        data = parse_json_response(raw)
        return ContentGaps.model_validate(data)
    except Exception:
        return ContentGaps(
            missing_sections=facts.missing_information,
            recommended_internal_questions=[
                "What formulation choices intentionally differentiate this product?",
                "Which ingredients are most central to the product's story?",
                "What competitive comparisons has marketing approved?",
                "Are there common customer misconceptions this file should address?",
                "Are there safety or usage boundaries that should be emphasized?",
            ],
        )
