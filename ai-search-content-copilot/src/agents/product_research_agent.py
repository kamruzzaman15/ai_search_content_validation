from __future__ import annotations
import json
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import ProductFacts


SYSTEM_PROMPT = """\
You are a product research assistant. Your job is to extract structured product facts from evidence provided from a product webpage.

Rules:
- Use ONLY the provided evidence. Do not invent or infer claims not supported by the text.
- If a field is not found in the evidence, return an empty string or empty array.
- Attach at least one evidence snippet (short quote from the source) to each major extracted point.
- Mark any claim that sounds medical, comparative, or unverifiable in claims_needing_review.
- List missing information fields in missing_information.
- Return valid JSON that matches the schema exactly.
"""

SCHEMA_EXAMPLE = """{
  "product_name": "string",
  "product_line": "string",
  "product_category": "string",
  "short_summary": "one sentence",
  "formulation_philosophy": {"summary": "string", "evidence": ["quote from page"]},
  "key_ingredients": [{"ingredient": "string", "stated_role": "string", "evidence": ["quote"]}],
  "differentiators": [{"point": "string", "evidence": ["quote"]}],
  "use_cases": [{"point": "string", "evidence": ["quote"]}],
  "competitive_positioning": [{"positioning_statement": "string", "evidence": ["quote"], "review_needed": true}],
  "claims_needing_review": ["list of verbatim claims that need review"],
  "missing_information": ["list of important fields not found on the page"]
}"""


def _build_prompt(bundle: dict, product_line_hint: str = "") -> str:
    text_sections = []

    if bundle.get("page_title"):
        text_sections.append(f"PAGE TITLE: {bundle['page_title']}")
    if bundle.get("meta_description"):
        text_sections.append(f"META DESCRIPTION: {bundle['meta_description']}")
    if bundle.get("headings"):
        text_sections.append("HEADINGS:\n" + "\n".join(f"- {h}" for h in bundle["headings"]))
    if bundle.get("paragraphs"):
        combined = "\n\n".join(bundle["paragraphs"][:30])
        text_sections.append(f"PARAGRAPHS:\n{combined}")
    if bundle.get("bullets"):
        text_sections.append("BULLET POINTS:\n" + "\n".join(f"- {b}" for b in bundle["bullets"][:50]))
    if bundle.get("raw_extracted_text"):
        text_sections.append(f"FULL EXTRACTED TEXT (first 3000 chars):\n{bundle['raw_extracted_text'][:3000]}")

    evidence_block = "\n\n".join(text_sections)
    hint_line = f"\nProduct line hint: {product_line_hint}" if product_line_hint else ""

    return f"""{SYSTEM_PROMPT}

{hint_line}

SOURCE URL: {bundle.get('source_url', '')}

--- EVIDENCE ---
{evidence_block}
--- END EVIDENCE ---

Return ONLY valid JSON matching this schema:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(bundle: dict, product_line_hint: str = "", model: str | None = None, token: str | None = None) -> ProductFacts:
    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    prompt = _build_prompt(bundle, product_line_hint)
    raw = call_llm(prompt, model=model, token=token, max_new_tokens=2048, temperature=0.2)

    try:
        data = parse_json_response(raw)
        return ProductFacts.model_validate(data)
    except Exception:
        # Return a minimal valid object so the pipeline doesn't crash
        return ProductFacts(
            product_name=bundle.get("page_title", "Unknown"),
            missing_information=["Agent failed to parse structured output. Raw response saved."],
        )
