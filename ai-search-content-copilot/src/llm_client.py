from __future__ import annotations
import json
import os
import re
import time


def _is_ollama_model(model: str) -> bool:
    return "/" not in model


def _call_ollama(prompt: str, model: str, max_new_tokens: int, temperature: float) -> str:
    import requests as _req
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    # Use the native /api/chat endpoint so we can pass think:false.
    # Qwen3 and other reasoning models consume tokens on internal thinking before
    # writing the answer; disabling it gives faster, more predictable structured output.
    resp = _req.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_huggingface(prompt: str, model: str, token: str, max_new_tokens: int, temperature: float) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_llm(
    prompt: str,
    model: str | None = None,
    token: str | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    retries: int = 3,
) -> str:
    model = model or os.environ.get("GENERATOR_MODEL", "gemma:latest")
    tok = token or os.environ.get("HF_TOKEN", "")

    for attempt in range(retries):
        try:
            if _is_ollama_model(model):
                return _call_ollama(prompt, model, max_new_tokens, temperature)
            else:
                return _call_huggingface(prompt, model, tok, max_new_tokens, temperature)
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {exc}") from exc
    return ""


def extract_json_from_response(text: str) -> str:
    """Pull the first JSON object or array out of a model response."""
    # Try fenced code block first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()

    # Find whichever comes first: [ or { — so arrays aren't eaten by object search
    pos_obj   = text.find("{")
    pos_array = text.find("[")

    if pos_obj == -1 and pos_array == -1:
        return text

    if pos_array == -1 or (pos_obj != -1 and pos_obj < pos_array):
        pairs = [("{", "}")]
    elif pos_obj == -1 or (pos_array != -1 and pos_array < pos_obj):
        pairs = [("[", "]")]
    else:
        pairs = [("{", "}")]

    for opener, closer in pairs:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text


def parse_json_response(text: str) -> dict | list:
    raw = extract_json_from_response(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON from model response: {exc}\n\nRaw:\n{raw}") from exc
