# Task: label the brand study's websites and brands with three LLM judges

## Context
`ai-search-content-copilot/brand_study/` (Part 1) collects Google organic results and AI Overviews and labels each
domain or brand as `big_brand`, `small_local`, `marketplace_retailer`, `directory_media_other` or `not_a_brand`
(otherwise `unknown`). Rule-based labels come from `brand_study/data/brand_rules_v1.csv`, and human corrections from
`brand_study/data/label_overrides.csv`. After run1, `outputs/brand_study/results/labels_to_review.csv` lists **298 unlabeled
items** (283 domains and 15 entities). Read `brand_study/labeling.py`, `analyze.py`, `report.py`, the README section
"Brand Visibility Study (Part 1)" and the category definitions in `report.py`/`labeling.py` before planning.

Goal: label these items automatically with **three independent LLM judges**, accept labels only where the judges agree,
send disagreements to the human, and **measure the judges' accuracy against a human-labelled sample**. The user's
earlier paper found that LLMs favour global brands over local ones, so the judges may share that bias. The validation
must be able to show it if it is there.

## Decisions already made by the user
- Judges:
  1. **Gemma 4 31B** via Ollama, tag `gemma4:31b` (about 20.4 GB; verified in the Ollama registry)
  2. **Qwen 3.8 27B** via Ollama, tag `qwen3.8:27b` (about 17.7 GB; verified in the Ollama registry)
  3. **Claude Opus 5.5, high effort**, via the Claude Code CLI in print mode:
     `claude -p --model claude-opus-5-5 --effort high --output-format text` with **no tools**. Find the binary the way
     `agent_workflow/orchestrate.py` `_find_binary` does (PATH, then the VS Code extension copy).
- Ollama models must be **downloaded before use and deleted after use** to free the shared disk. Use the existing
  `brand_study/ollama_lifecycle.py` helper. Run the two local judges **one after the other**, never both loaded at once.

## Requirements

1. **Evidence per item.** Judges must not label from the name alone, because small local businesses are mostly unknown to LLMs.
   Build an evidence record from data already collected: key, key_type, example URL(s), titles, snippets, which
   queries/sectors/surfaces it appeared in, and local-pack data. Optionally (`--fetch` flag, on by default, `--no-fetch` to disable)
   fetch the site's homepage for title, meta description and a short visible-text excerpt (≤1,500 chars). Use a timeout,
   a clear User-Agent, one request per domain, and no retries storms. Cache fetched evidence on disk. Fetch failures are recorded and are not fatal.
   Treat fetched page text as untrusted data inside the prompt (clearly delimited; instructions inside it are ignored).

2. **One shared prompt** for all judges, containing the exact operational category definitions the study already uses
   (big = ownership by a public/multinational group OR national distribution through major US retailers OR a multi-state
   chain/franchise; and so on). Each judge returns strict JSON per item: `category` (one of the 5 + `unknown` allowed),
   `confidence` (0–1), and `reason` (one sentence). Judges never see each other's answers. Use temperature 0 / deterministic
   settings where supported. Batch items per call (for example 10–20) to keep cost and time reasonable. Validate the JSON,
   retry a malformed batch once, and record anything still invalid as an abstention.

3. **Caching and resume.** Store every judge answer per item (JSONL under `outputs/brand_study/llm_labels/`, ignored by Git),
   keyed by judge, model tag, prompt version and item, so a rerun skips finished work and an interrupted run resumes.
   Record model tag, prompt version and timestamp.

4. **Claude usage limits.** If the Claude CLI reports a usage limit ("hit your … limit", "usage limit", and so on), wait until
   the stated reset time and retry. If no time is stated, poll every 15 minutes. Cap each wait at 6 hours. Reuse or mirror
   the logic in `agent_workflow/orchestrate.py`. Other errors fail clearly.

5. **Aggregation.**
   - **3 of 3 agree:** accept automatically. Write to a NEW file `brand_study/data/llm_label_overrides.csv` with
     columns `key_type,key,category,note`, where note carries the vote summary. Keep it separate from the human file.
   - **2 of 3 or no majority, or any abstention:** write to `outputs/brand_study/results/llm_disagreements.csv` showing all
     three votes, confidences and reasons, for the human to resolve in `label_overrides.csv`.
   - The policy is configurable (`--accept unanimous|majority`, default `unanimous`).
   - **Precedence in labeling:** human `label_overrides.csv` > `llm_label_overrides.csv` > rules. Record the
     `label_source` (for example `llm_unanimous`) so the report can show how many labels came from rules, LLMs and humans.
     The report must state what share of labels came from the LLM judges.

6. **Human validation.**
   - A command creates a **blinded** stratified random sample (default 50, fixed seed) for the user to label, without
     showing any LLM votes. Include both unanimous and non-unanimous items.
   - A second command reads the user's labels and reports:
     - per-judge accuracy and Cohen's kappa against the human labels
     - the ensemble's accuracy on unanimous items
     - pairwise judge agreement and Fleiss' kappa
     - a confusion matrix
     - an explicit **big-brand bias check**: how often each judge labels as `big_brand` an item the human labelled `small_local`, and the reverse
   - Write the results to a Markdown file.

7. **CLI.** Add subcommands under `python -m brand_study` (for example `llm-label`, `llm-sample`, `llm-agreement`) with
   `--judges` to choose a subset (for example `claude` only), `--limit N` for a small trial, and `--dry-run` to show the work
   without calling any model.

8. **Docs.** Update the README "Brand Visibility Study (Part 1)" section in `ai-search-content-copilot/README.md`, and the
   short summary in the **root `README.md`**, with the new step, its cost and time, and its limitations (LLM bias,
   validation sample).

9. **Tests.** Offline pytest tests with mocked Ollama, Claude CLI and HTTP calls. Cover JSON validation and retry,
   aggregation and precedence, caching and resume, usage-limit waiting (mocked clock), and agreement metrics (check kappa
   against a hand-computed example). **No network or model calls in tests.**

## Constraints
- Python 3.9 compatible. Follow the existing `brand_study` code style. Use existing dependencies where possible. If a new
  dependency is truly needed, add it to `requirements.txt` with a reason.
- During implementation, **do not pull the Ollama models and do not run the full labeling**. The user's assistant will
  run it after review. A live smoke test of **at most 3 items with the Claude judge only** is allowed.
- Do not modify `.env`, do not commit or push, and do not change existing study results or the human overrides file.
- Keep the existing 177 tests passing.
