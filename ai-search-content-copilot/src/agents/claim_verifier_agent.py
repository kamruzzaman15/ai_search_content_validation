from __future__ import annotations
import json
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import ClaimAudit, ClaimAuditItem


SYSTEM_PROMPT = """\
You are a fact-checking reviewer for product marketing content. Your job is to audit every non-trivial factual claim in a product markdown file against the original source evidence.

Rules:
- Break the markdown into atomic, individual claims.
- Compare each claim to the provided evidence.
- Classify support conservatively using these labels:
    supported — claim is clearly backed by the evidence
    partially_supported — claim is partially backed but overstated
    unsupported — claim has no basis in the evidence
    needs_internal_confirmation — claim may be true but cannot be verified from public page alone
- Flag any medical-like language, strong superiority claims, or safety-sensitive wording with risk_level: high.
- Recommend: keep | soften | seek_internal_confirmation | remove
- Return ONLY valid JSON. No explanation outside the JSON.
"""

SCHEMA_EXAMPLE = """[
  {
    "claim": "verbatim or paraphrased claim from the markdown",
    "status": "supported | partially_supported | unsupported | needs_internal_confirmation",
    "evidence": "short quote from source evidence that supports or refutes, or empty string",
    "risk_level": "low | medium | high",
    "recommended_action": "keep | soften | seek_internal_confirmation | remove"
  }
]"""


def _build_prompt(bundle: dict, markdown: str) -> str:
    evidence_text = bundle.get("raw_extracted_text", "")[:3000]
    bullets = "\n".join(f"- {b}" for b in bundle.get("bullets", [])[:30])
    paragraphs = "\n\n".join(bundle.get("paragraphs", [])[:15])

    return f"""{SYSTEM_PROMPT}

--- SOURCE EVIDENCE ---
URL: {bundle.get('source_url', '')}

Paragraphs:
{paragraphs}

Bullets:
{bullets}

Raw text:
{evidence_text}
--- END EVIDENCE ---

--- MARKDOWN DRAFT TO AUDIT ---
{markdown}
--- END MARKDOWN ---

Return a JSON array of claim audit objects:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(bundle: dict, markdown: str, model: str | None = None, token: str | None = None) -> ClaimAudit:
    model = model or os.environ.get("REVIEWER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    prompt = _build_prompt(bundle, markdown)
    raw = call_llm(prompt, model=model, token=token, max_new_tokens=2048, temperature=0.1)

    try:
        data = parse_json_response(raw)
        if isinstance(data, dict) and "claims" in data:
            data = data["claims"]
        items = [ClaimAuditItem.model_validate(item) for item in data]
    except Exception:
        items = []

    audit = ClaimAudit(claims=items)
    audit.compute_summary()
    return audit
