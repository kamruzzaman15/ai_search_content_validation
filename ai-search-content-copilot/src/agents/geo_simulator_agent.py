"""
GEO Simulator Agent

Simulates how a generative search system would answer customer questions if it
only had the product markdown as its context (RAG simulation). This reveals
which facts the current content surfaces well and which it buries or omits.
"""
from __future__ import annotations
import os
from src.llm_client import call_llm, parse_json_response
from src.schemas import GEOSimResult, ProductFacts


SYSTEM_PROMPT = """\
You are simulating a generative AI search assistant. You have been given ONLY the product content below as your knowledge source.
A customer has asked you a question. Your job is to:

1. Answer the question using only the provided content (simulate what an AI search system would say).
2. List the specific facts you were able to use from the content to build your answer.
3. List any facts that a complete answer would ideally include but that were NOT clearly present in the content.
4. Give a coverage score from 0 to 10: how well did the content support a complete, useful answer?
   0 = content gave nothing useful, 10 = content fully supported a detailed answer.

Rules:
- Do not use any knowledge outside the provided content.
- If the content does not contain enough information to answer, say so explicitly.
- Be honest: a poor coverage score is a useful signal, not a failure.
- Return ONLY valid JSON. No text outside the JSON.
"""

SCHEMA_EXAMPLE = """{
  "question": "the customer question",
  "simulated_answer": "the answer a generative system would give based only on the content",
  "key_facts_surfaced": ["fact 1 the content made clear", "fact 2 ..."],
  "key_facts_missed": ["important fact not present or unclear in the content", "..."],
  "coverage_score": 7
}"""


def _build_prompt(markdown: str, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

--- PRODUCT CONTENT (your only knowledge source) ---
{markdown}
--- END CONTENT ---

Customer question: {question}

Return JSON:
{SCHEMA_EXAMPLE}

JSON:
"""


def run(
    markdown: str,
    facts: ProductFacts,
    questions: list[str],
    model: str | None = None,
    token: str | None = None,
) -> list[GEOSimResult]:
    model = model or os.environ.get("GENERATOR_MODEL", "Qwen/Qwen3-8B")
    results: list[GEOSimResult] = []

    for question in questions:
        prompt = _build_prompt(markdown, question)
        raw = call_llm(prompt, model=model, token=token, max_new_tokens=1024, temperature=0.2)

        try:
            data = parse_json_response(raw)
            result = GEOSimResult.model_validate(data)
        except Exception:
            result = GEOSimResult(
                question=question,
                simulated_answer="Simulation failed — could not parse model response.",
                coverage_score=0,
            )

        results.append(result)

    return results
