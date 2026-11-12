# AI Search Content Copilot

> A two-layer agentic pipeline that turns any public product webpage into structured, fact-grounded, AI-search-ready content — then tests and iteratively improves it using a GEO feedback loop.

---

## What This Is

**AI Search Content Copilot** is a human-in-the-loop agentic system for product content teams. Give it any public product page URL — it extracts and structures the product knowledge, drafts a reviewable markdown content file, audits every claim for factual support, and then runs a second optimization layer that simulates how generative AI search systems would represent the content and applies targeted revisions to improve coverage.

The system runs in two modes — a **Sequential** pipeline (two separate stages you control) or a **LangGraph** pipeline (one end-to-end run with an automatic iterative refinement loop).

The core principle: **the goal is not to game AI search rankings. The goal is to build content that is more explicit, structured, grounded, and measurable** — so that when AI systems encounter it, they can represent the product accurately.

---

## Demo

```
Input:  Any public product page URL
Output: 13-tab Streamlit UI with structured facts, markdown draft,
        claim audit, content gaps, GEO coverage scores, revised content,
        baseline comparison, and multi-product history
```

---

## Architecture

### Two pipeline modes

```
┌─────────────────────────────────────────────────────────────────────┐
│  SEQUENTIAL MODE                                                      │
│  Button 1 → Layer 1   Button 2 → Layer 2 (optional, one pass)       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  LANGGRAPH MODE                                                       │
│  One button → both layers, automatic iterative loop                  │
│                                                                       │
│  simulate_geo ──▶ should_continue? ──▶ "revise"                      │
│       ▲                    │               │                          │
│       │                    ▼               ▼                          │
│       │                  "end"     analyze_visibility                 │
│       │                    │               │                          │
│       │                   END      select_strategies                  │
│       │                                    │                          │
│       └────────────── revise_content ◀─────┘                         │
│                                                                       │
│  Stops when: max_iterations OR coverage_threshold OR plateau          │
└─────────────────────────────────────────────────────────────────────┘
```

### Full pipeline — 9 agents across 2 layers

```
Product URL
    │
    ▼
[1] Webpage Extractor
    requests + BeautifulSoup + trafilatura
    Playwright fallback for JS-heavy pages
    → Evidence Bundle JSON
    │
    ▼
[2] Product Research Agent          qwen2.5:latest
    Extracts structured facts from evidence only
    Attaches source quotes to every point
    → product_facts.json
    │
    ▼
[3] Markdown Writer Agent           qwen2.5:latest
    Writes 8-section markdown from facts only
    Cannot invent claims not present in facts
    → product_context.md
    │
    ▼
[4] Claim Verifier Agent            llama3.1:8b-instruct-q4_K_M
    Independent model audits every claim
    Labels: supported / partially_supported /
            unsupported / needs_internal_confirmation
    → claim_audit.json
    │
    ▼
[5] Gap Analysis Agent              qwen2.5:latest
    Identifies missing info and unclear claims
    Generates expert interview questions
    → content_gaps.json + inventory_row.json

    ── Layer 2 begins ──────────────────────────────────────────────

    ▼
[6] GEO Simulator                   qwen2.5:latest
    Simulates RAG-style answers to eval questions
    using only the markdown as context (0–10 score)
    → sim_results (before)
    │
    ▼
[7] Visibility Analyzer             qwen2.5:latest
    Compares simulated answers to full product facts
    Types: missing_fact / weak_coverage /
           ambiguous_claim / no_differentiator / poor_structure
    → visibility_gaps (prioritized)
    │
    ▼
[8] Revision Strategy Selector      qwen2.5:latest
    Picks content-specific strategies per gap:
    add_specificity / surface_differentiators /
    add_faq / clarify_ambiguous /
    improve_use_case_depth / consolidate_evidence /
    add_structured_summary
    → selected_strategies
    │
    ▼
[9] Content Reviser                 qwen2.5:latest
    Applies strategies → improved markdown
    Strictly evidence-grounded, cannot invent claims
    → revised_markdown
    │
    ▼
    simulate_geo again → check stopping conditions → loop or END
```

---

## Two-Model Design

| Model | Role | Why |
|---|---|---|
| `qwen2.5:latest` | Generator / Writer / Analyzer | Strong instruction following; handles all production and analysis steps |
| `llama3.1:8b-instruct-q4_K_M` | Independent Reviewer | Different model for claim auditing prevents self-confirmation bias |

Both run locally via **Ollama** with `think: false` passed through the native `/api/chat` endpoint for fast, deterministic structured output. A Hugging Face Inference API backend is also supported.

---

## Agent Communication

Agents communicate through **direct Python function calls orchestrated by the Streamlit app** (Sequential mode) or **LangGraph state passing** (LangGraph mode).

There is no message bus, shared memory system, or inter-agent HTTP protocol. Each agent's output is a Pydantic-validated model passed directly as input to the next step.

In LangGraph mode, state is a typed `GEOState` dict that flows through the graph. The `coverage_history` field uses `Annotated[list, operator.add]` so each simulation appends its score automatically — the loop's history builds up across iterations without manual accumulation.

### LangGraph stopping conditions

The loop exits when **any one** of these fires:

```python
if iteration >= max_iterations:       return "end"   # hard stop (default: 3)
if coverage_history[-1] >= threshold: return "end"   # target reached (default: 7.5/10)
if improvement < min_improvement:     return "end"   # plateau (default: 0.5 pts)
return "revise"                                       # loop back
```

---

## Why Not One-Shot?

A naive approach sends page text directly to an LLM and asks it to write markdown. This project deliberately rejects that:

| Problem with one-shot | How this system solves it |
|---|---|
| No transparency — where did facts come from? | Evidence bundle keeps all source quotes; every extracted fact has an attached snippet |
| Hallucination risk | Research agent instructed to return empty rather than invent; claim verifier catches anything that slips through |
| No claim accountability | Independent reviewer model (llama3.1) audits every non-trivial claim |
| No structure enforcement | Pydantic schemas validate all outputs |
| Hard to evaluate | Automated quality checks + one-shot baseline comparison built in |
| No iterative improvement | LangGraph feedback loop revises content until coverage plateaus |

**Measured result on tested product pages:**
- One-shot LLM: ~37% unsupported claims
- Agentic pipeline: ~0% unsupported claims

---

## Streamlit UI — 13 Tabs

Select **Sequential** or **LangGraph** at the top of the sidebar. Works with any public product URL.

| Tab | Contents |
|---|---|
| Extracted Evidence | Evidence bundle: paragraphs, bullets, headings, tables, JSON-LD |
| Product Facts | Structured facts JSON with source quotes per point |
| Markdown Draft | Side-by-side preview and raw markdown |
| Claim Audit | Per-claim status, risk level, recommended action, progress bar |
| Content Gaps | Missing sections, unclear claims, expert interview questions |
| Inventory Row | One-row tracking record: file status, claim review status, next step |
| Evaluation Questions | 8 auto-generated product questions for downstream testing |
| **GEO Simulation** | **Per-question before/after coverage score table + side-by-side simulated answers** |
| Visibility Gaps | Typed, prioritized gaps with the question that exposed each one |
| Revision Strategies | Selected strategies with specific instructions |
| Revised Content | Improved markdown + full GEO report JSON download |
| **Baseline Comparison** | **One-shot LLM vs agentic pipeline: summary table, winner-by-metric, claim-level icons** |
| **Product History** | **Multi-product comparison table accumulated across runs, CSV download** |

### LangGraph-specific sidebar settings

When **LangGraph** mode is selected, three sliders appear:

| Setting | Default | Effect |
|---|---|---|
| Max revision iterations | 3 | Hard stop on the refinement loop |
| Coverage threshold | 7.5 / 10 | Stop early if avg score reaches this |
| Min improvement | 0.5 pts | Stop if last cycle gained less than this (plateau) |

---

## Evaluation Framework

### CLI — automated quality checks

```bash
python3 evaluate.py "<product_url>"
```

| Check | Target |
|---|---|
| Schema validity | All required fields populated |
| Field coverage | Ingredients, differentiators, use cases, philosophy found |
| Evidence grounding | ≥ 70% of facts have source quotes |
| Claim audit sanity | Supported ≥ 60%, unsupported ≤ 20% |

### CLI — baseline comparison

```bash
python3 evaluate_baseline.py "<product_url>"
```

Runs one-shot LLM and full agentic pipeline on the same URL, scores both with the reviewer model, prints a side-by-side table, and saves all outputs to `outputs/eval/`.

### In-app — three comparison views

**GEO Simulation tab (before vs after revision)**
- Coverage score (0–10) per question, before and after
- Delta column shows which questions improved
- Facts missed before vs after per question
- Side-by-side simulated answers

**Baseline Comparison tab (one-shot vs agentic)**
- Summary table: sections found, word count, supported %, unsupported %
- Winner-by-metric breakdown
- Claim-level icons side by side (✅ ⚠️ ❌)

**Product History tab (multi-product)**
- One row per completed run, accumulated in session
- GEO before/after columns auto-fill when GEO loop is run
- CSV download

> **Honest limitation on GEO scores:** Coverage scores are self-assessed by the same LLM performing the simulation. This is an internal proxy metric, not a live test against Google AI Overviews, ChatGPT, or Perplexity. The before/after delta is meaningful as a relative comparison. External validation requires testing against real search systems.

---

## Repository Structure

```
ai-search-content-copilot/
│
├── app/
│   └── streamlit_app.py              — Streamlit UI (13 tabs, Sequential + LangGraph modes)
│
├── src/
│   ├── webpage_extractor.py          — HTML fetch, BeautifulSoup, evidence bundle builder
│   ├── llm_client.py                 — Ollama (/api/chat) + HuggingFace router with retry
│   ├── schemas.py                    — All Pydantic models (Layer 1 + Layer 2)
│   ├── pipeline_state.py             — GEOState TypedDict for LangGraph
│   ├── pipeline_langgraph.py         — LangGraph graph: nodes, conditional edge, compiled graph
│   └── agents/
│       ├── product_research_agent.py     — Layer 1: extracts structured facts
│       ├── markdown_writer_agent.py      — Layer 1: writes 8-section markdown
│       ├── claim_verifier_agent.py       — Layer 1: audits claims (reviewer model)
│       ├── gap_analysis_agent.py         — Layer 1: finds gaps, generates expert questions
│       ├── geo_simulator_agent.py        — Layer 2: RAG simulation per eval question
│       ├── visibility_analyzer_agent.py  — Layer 2: identifies typed, prioritized gaps
│       ├── revision_strategy_agent.py    — Layer 2: selects content-specific strategies
│       └── content_reviser_agent.py      — Layer 2: applies strategies, evidence-grounded
│
├── brand_study/                      — Part 1: brand visibility study (Google organic vs AI Overviews)
│   ├── collect.py / serpapi_client.py    — SerpApi collection with cache and budget guard
│   ├── extract.py / labeling.py          — Brand/domain extraction and big-vs-small labeling
│   ├── analyze.py / report.py            — Tidy CSVs and report.md
│   ├── ollama_lifecycle.py               — Pull → use → delete helper for Ollama models
│   └── data/                             — Query set, brand rules, label overrides
│
├── tests/test_brand_study/           — Offline tests with mocked SerpApi responses
│
├── evaluate.py                       — CLI: automated quality checks
├── evaluate_baseline.py              — CLI: one-shot vs agentic comparison
│
├── outputs/
│   ├── eval/                         — Saved markdown and audit files from CLI runs
│   └── brand_study/                  — SerpApi cache and study results (not committed)
│
├── .env                              — Model config (Ollama URL + model names) and SERPAPI_API_KEY
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running

### Pull the models

```bash
ollama pull qwen2.5:latest
ollama pull llama3.1:8b-instruct-q4_K_M
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Configure `.env`

```bash
OLLAMA_BASE_URL=http://localhost:11434
GENERATOR_MODEL=qwen2.5:latest
REVIEWER_MODEL=llama3.1:8b-instruct-q4_K_M
HF_TOKEN=          # only needed for Hugging Face backend
SERPAPI_API_KEY=   # only needed for the brand visibility study (see below)
```

### Run the app

```bash
python3 -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

Open **http://localhost:8501** in your browser.

> **Remote server via VSCode:** forward port 8501 in the Ports tab, then open http://localhost:8501 locally.

> **Keep alive across disconnects:**
> ```bash
> tmux new -s copilot
> python3 -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
> # Ctrl+B then D to detach
> # tmux attach -t copilot to return
> ```

### Run CLI evaluation

```bash
python3 evaluate.py "https://example.com/products/my-product"
python3 evaluate_baseline.py "https://example.com/products/my-product"
```

---

## Pydantic Schemas

### Layer 1

| Schema | Purpose |
|---|---|
| `EvidenceBundle` | Structured webpage extraction output |
| `ProductFacts` | Structured product knowledge with evidence quotes |
| `ClaimAudit` | Per-claim audit with status, risk, and recommended action |
| `ContentGaps` | Missing sections, unclear claims, expert questions |
| `InventoryRow` | One-row content tracking record per product |
| `EvaluationQuestions` | Auto-generated question set for downstream testing |

### Layer 2

| Schema | Purpose |
|---|---|
| `GEOSimResult` | Simulated answer per question with coverage score and facts surfaced/missed |
| `VisibilityGap` | One typed, prioritized content gap identified from simulation results |
| `RevisionStrategy` | One targeted revision with a concrete instruction for the writer |
| `GEOFeedbackReport` | Full Layer 2 output: sim results before/after, gaps, strategies, revised markdown |

### LangGraph

| Schema | Purpose |
|---|---|
| `GEOState` | TypedDict passed through all graph nodes; `coverage_history` uses `operator.add` for automatic list accumulation across iterations |

---

## Research Foundation

| Paper | Connection to this project |
|---|---|
| **GEO** (Aggarwal et al.) | Framework for measuring content visibility in generative engines; motivation for the evaluation design |
| **AutoGEO** | LLMs learning generative engine preferences; the revision strategy taxonomy (add_specificity, add_faq, surface_differentiators, etc.) is directly inspired by AutoGEO's strategy inventory |
| **AgenticGEO** | Adaptive multi-step optimization with feedback-driven refinement; the LangGraph iterative loop architecture mirrors this design |
| **SAGEO Arena** | Full-pipeline evaluation (retrieval + reranking + generation); influenced the decision to simulate complete RAG-style answers rather than just scoring text quality |

---

## Responsible Use

Hard constraints baked into every agent:

1. Agents extract **only from provided evidence** — they cannot invent claims
2. A **separate reviewer model** (llama3.1) audits every non-trivial claim independently
3. Uncertain or unverifiable claims are **escalated, not smoothed over**
4. The content reviser is **explicitly forbidden** from adding claims not already in the product facts
5. The gap analysis generates **questions for human experts** — it does not fill gaps with guesses
6. Coverage scores are presented as **internal proxies**, not real-world ranking signals

---

## Example Results

Tested on a Life's Abundance canned cat food product page:

| System | Sections Found | Supported Claims | Unsupported Claims |
|---|---|---|---|
| One-shot LLM | 8/8 | 63.2% | 36.8% |
| Agentic Pipeline | 8/8 | **81.8%** | **0.0%** |

LangGraph run on same page:

| Metric | Value |
|---|---|
| GEO coverage before revision | 7.12 / 10 |
| GEO coverage after 1 revision cycle | 7.25 / 10 |
| Loop exit reason | Plateau detected (improvement 0.13 < threshold 0.5) |

---

## Brand Visibility Study (Part 1)

A self-contained study module in `brand_study/`, separate from the GEO pipeline in `src/`. It asks one question:
**are small/local businesses less visible than big brands in Google's organic top 10 and in Google AI Overviews?**
It covers two sectors (coffee shops, skincare) and one location (Florida, US). The only data source is Google
results through [SerpApi](https://serpapi.com): the organic top 10, the AI Overview (AIO) and, for description
only, the local pack.

### Design

| Step | Module | What it does |
|---|---|---|
| Queries | `data/queries_v1.csv`, `queries.py` | 50 versioned queries (25 coffee, 25 skincare), broad and Florida-local, over Miami, Tampa, Orlando, Jacksonville, St. Petersburg and Fort Lauderdale. Broad queries use `location=Florida,United States`, and local queries use the city. Ids are never reused: if a query's text changes, it gets a new id |
| Collection | `collect.py`, `serpapi_client.py`, `cache.py`, `budget.py` | `engine=google`, `gl=us`, `hl=en`, desktop, `no_cache=true`. When the AIO comes back only as a `page_token`, the `google_ai_overview` follow-up is made immediately, because the token expires within about a minute. Every raw response is cached under `outputs/brand_study/raw/<run>/` |
| Extraction | `extract.py`, `domains.py`, `entities.py` | Organic domains with rank, AIO cited sources, AIO brand mentions and local-pack places. Domains are normalised (`www.`/`m.` stripped, Google redirects unwrapped, hosting-platform subdomains such as `*.square.site` kept whole) |
| Labeling | `labeling.py`, `data/brand_rules_v1.csv`, `data/label_overrides.csv` | Transparent rules plus your overrides |
| LLM labeling (optional) | `llm_evidence.py`, `llm_judges.py`, `llm_aggregate.py`, `llm_validation.py`, `data/llm_label_overrides.csv` (generated) | Three independent LLM judges label what the rules leave `unknown`. Only unanimous labels are used, disagreements go to you, and a blinded human sample measures the judges' accuracy and big-brand bias (see [LLM labeling](#llm-labeling-optional-after-the-last-collection-run)) |
| Analysis | `analyze.py`, `report.py` | Tidy CSVs plus `report.md` in `outputs/brand_study/results/` |

SerpApi field names and billing facts, with sources, are recorded in `brand_study/SERPAPI_NOTES.md`.

### Categories and the big-vs-small test

| Category | Meaning |
|---|---|
| `big_brand` | National or global chains and brands, including national service chains |
| `small_local` | Small or independent businesses. For local queries: independent local businesses, including Florida-only chains. For broad product queries: small or independent brands |
| `marketplace_retailer` | Amazon, Walmart, Target, Sephora, Ulta, CVS, delivery apps, Google Shopping panels cited by AIOs (`shopping.google.com`) and similar |
| `directory_media_other` | Yelp, Tripadvisor, Reddit, social media, publishers, `.gov`/`.edu` |
| `not_a_brand` | Ingredients and attributes (retinol, wifi, ...). Excluded from headline numbers |
| `unknown` | Not labeled yet. Never silently counted as small: it appears in the bounds and the unresolved share |

A brand or business is `big_brand` if **any** of these holds, and `small_local` only if none does:
**(a) ownership:** it is majority-owned by a public company or a multinational group;
**(b) distribution:** it is sold nationally through major US retailers;
**(c) footprint:** it has locations in several US states or franchises nationally.
Each seeded `big_brand` rule records which criterion applies in its `basis` column.

Label precedence: override → exact domain rule → parent-domain rule → name rule → hosting-platform heuristic
(`small_local`) → `.gov`/`.edu` (`directory_media_other`) → LLM label (optional, only for rows still unknown after
every rule and override path) → `unknown`. Brand names that are also common words
have an `ambiguous` mode in the rule file:
- `case`: case-sensitive match ("The Ordinary" matches, "an ordinary moisturizer" does not);
- `case_mid`: case-sensitive, and not at the start of a sentence, list item, heading or cell ("sold at Target" matches, "Target dark spots" does not);
- `candidate`: produces an unconfirmed candidate only (a lone "illy"). Only an override can confirm it.

An AIO mention is *confirmed* by a dictionary match, by a name rule or override, or by matching a local-pack
business for the same query (`local_pack_name`). Unconfirmed candidates are left out of the point estimate but
are counted in the bounds.

### Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
# put SERPAPI_API_KEY=... in .env or the shell; the key is never printed, logged or cached
python -m pytest tests -q      # offline; no network is used in the tests
```

### Steps

All commands run from `ai-search-content-copilot/`.

1. `python -m brand_study check-queries`: validates the query file and prints counts.
2. `python -m brand_study account`: shows the remaining searches (free Account API).
3. `python -m brand_study collect --run run1 --dry-run`: shows the plan, the study total so far, the cap and the estimate. Spends nothing.
4. `python -m brand_study collect --run run1`: asks for confirmation, then collects. Add `--yes` for non-interactive use.
5. At least 12–24 h later: `python -m brand_study collect --run run2`. It refuses to start within 12 h of run1 unless you pass `--force-spacing`.
6. Optional, after the **last** collection run: label the remaining unknowns with three LLM judges (see [LLM labeling](#llm-labeling-optional-after-the-last-collection-run)).
7. `python -m brand_study analyze`. The first pass is a **labeling pass**, so expect a PROVISIONAL banner.
8. Open `outputs/brand_study/results/labels_to_review.csv`. Copy the corrections you agree with into `brand_study/data/label_overrides.csv` (`key_type,key,category,note`). Suggestions (`small_local` for local-pack websites and names, `not_a_brand` for ingredient-like headings) are never applied automatically. An AIO product name (for example "Minimalist 10% Vitamin B5 Gel Face Moisturizer") appears under its full normalised key. Either override that key, or add a `name` rule for the brand to `brand_rules_v1.csv`, so that every product from that brand is matched.
9. Re-run `analyze` until the banner clears or the bounds are tight enough.
10. `python -m brand_study scan-leaks`: checks the output folder for the API key. It prints file names and a count only.

Useful `collect` options: `--sector`, `--query-ids a,b`, `--limit N`, `--max-new N` (hard limit for this
invocation), `--retries` (default 1), `--retry-failed-aio --query-ids ...` (re-requests cached searches whose AIO
follow-up failed, and moves the old files to `_superseded/`).

### Cost in SerpApi searches

| What | SerpApi searches |
|---|---|
| Account API, `--dry-run`, `analyze`, `check-queries`, `scan-leaks` | 0 |
| Each run | 50 main searches plus at most 50 AIO follow-ups. The guard plans for this plus 10 retry headroom (≤ 110) |
| Full study (2 runs) | ≤ 200 in the normal worst case, ≤ 220 with retry headroom, against a study-wide cap of 300 |
| `--retry-failed-aio` | ≤ 2 attempts per main search and ≤ 2 per follow-up for each listed query, within the same cap |
| Smoke test | ≤ 4 (`--max-new 4 --retries 0`) |
| Hard guarantee | Ledger attempts never exceed `--study-cap` or the account's remaining searches |

How the budget works:
- `--study-cap` (default 300, env `BRAND_STUDY_CAP`) covers the **whole study**: every run, including smoke tests and earlier invocations. It is counted from `outputs/brand_study/ledger.jsonl`, which has one line per HTTP attempt. Every attempt counts as spent, even when SerpApi probably did not bill it.
- Before collecting, the free Account API is called. A run is refused if the estimate is above the cap or above the remaining searches. During the run, each attempt is charged before it is sent, and a main search starts only if one search is left over for its possible follow-up.
- A cached `(run, query, request)` is never re-requested. Failed main searches are not cached, so they can be retried. Failed AIO follow-ups *are* cached as failures, because their token has already expired.
- After a run, the summary shows logical requests, attempts, retries and the Account API usage delta.

### Outputs

`outputs/brand_study/results/` holds `query_status.csv`, `organic_results.csv`, `aio_references.csv`,
`aio_mentions.csv`, `local_pack.csv`, `summary_by_segment.csv` (long format, with `bound_low`, `bound_high`,
`unresolved_share` and `provisional`), `overlap_by_query.csv`, `stability_by_query.csv` (Jaccard run1 vs run2),
`labels_to_review.csv` and `report.md`. Everything under `outputs/brand_study/` is ignored by Git.
The optional LLM step adds `llm_disagreements.csv`, `llm_validation_sample.csv`, `llm_validation.md` and
`llm_validation_overrides_snippet.csv` to `results/`, its cache under `outputs/brand_study/llm_labels/`, and the
generated `brand_study/data/llm_label_overrides.csv` (tracked, so the labels used are reviewable).

### Limitations

- **Observational only:** one geography, specific points in time, SerpApi's simulated location and device, and no login or personalisation. AI Overviews are non-deterministic, and cells are small (10–25 queries), so CIs are wide.
- Domain-level labels undercount small businesses, which appear mostly inside directories, listicles and the local pack. Cited pages are not scraped.
- Before labels are reviewed, the small-business share of AIO mentions is biased down. The bounds show how far it could move, and the upper bound is loose.
- Social profiles are labeled as directories, even when they belong to a small business.
- Mention detection is dictionary-based plus list-title candidates.
- Optional LLM labels can share the LLMs' preference for global brands over local ones. Their accuracy and the bias test rest on a small blinded human sample (default 50 items), so the confidence intervals are wide. See [LLM labeling](#llm-labeling-optional-after-the-last-collection-run).

### LLM labeling (optional, after the last collection run)

After the rules and your overrides, some domains and entities are still `unknown` (298 after run1: 283 domains
and 15 entities). `llm-label` asks three independent judges to classify each one, with one shared prompt that
contains the study's category definitions and the big-vs-small operational test:

| Judge | Backend | Model |
|---|---|---|
| `gemma` | Ollama, downloaded before use and deleted afterwards | `gemma4:31b` (about 20.4 GB) |
| `qwen` | Ollama, downloaded before use and deleted afterwards | `qwen3.8:27b` (about 17.7 GB) |
| `claude` | Claude Code CLI in print mode, high effort, no tools | `claude-opus-5-5` |

The two Ollama tags were verified in the Ollama registry on 2026-11-10. Override them with `BRAND_STUDY_GEMMA_TAG` /
`BRAND_STUDY_QWEN_TAG` (and the Claude model with `BRAND_STUDY_CLAUDE_MODEL`). The Part 2 `JUDGE_MODEL` setting is
separate and unchanged.

**How it works**

- **Evidence, not names.** Small businesses are mostly unknown to LLMs, so each item gets an evidence record from data already collected: example URLs, titles and snippets, the queries, sectors and surfaces it appeared in, local-pack places and AIO text. With `--fetch` (the default), each selected domain's homepage is fetched once: the title, meta description and up to 1,500 characters of visible text. The fetch uses a timeout, a clear User-Agent, at most 3 manual redirects with a public-IP check on every hop, no retries and no robots.txt check, and it is cached under `llm_labels/fetch/`. Failures are recorded and are not fatal. Scraped text is passed to the judges as clearly delimited, untrusted data.
- **Judges are independent** and never see each other's answers. Items go in batches of 12. Each answer is strict JSON (`category`, `confidence`, one-sentence `reason`). A malformed batch is retried once for the bad items, and anything still invalid is recorded as an abstention. The Ollama models run at temperature 0, one after the other; a model is only pulled when it has pending work. Claude runs from an empty temporary directory outside the repository with `--tools ""` and the isolation flags the installed CLI supports (probed from `--help` and recorded on every vote). On a usage limit it waits until the stated reset time (or polls every 15 minutes, at most 6 hours per wait) and retries. A generic rate-limit, quota or HTTP 429 message that states no reset time is retried for at most 24 polls (about 6 hours) and then stops the run with an error, so a persistent non-limit failure cannot stall it for days.
- **Caching and resume.** Every answer is stored in `llm_labels/votes_<judge>.jsonl`, keyed by judge, model, prompt version, item and evidence hash, so a rerun skips finished work and an interrupted run resumes.
- **Aggregation.** 3 of 3 agreeing (not `unknown`) → accepted into the generated `brand_study/data/llm_label_overrides.csv` (`key_type,key,category,note`, with the votes in the note; do not edit it). Anything else (2 of 3, no majority, an abstention, or a conflict with a candidate rule) → `results/llm_disagreements.csv` with all votes, confidences and reasons, for you to resolve **by adding rows to `label_overrides.csv`**. `--accept majority` also accepts 2-of-3 labels.
- **Precedence: human override > rules > LLM fallback.** An LLM label only fills a row that is still `unknown` after every rule and override path, so it never beats a rule or a human label. For the items the LLM step labels (which the rules leave unknown), this gives the same result as "human > LLM > rules". The report's "Labeling coverage" section shows how many rows came from rules, human overrides and the LLM judges, and states the LLM share. An LLM label can also confirm a candidate-only AIO mention whose rule category it matches; this is reported separately (`llm_confirm`).

**Steps** (from `ai-search-content-copilot/`, after the last collection run)

1. `python -m brand_study llm-label --fetch-only` — fetch the homepages (about 5–10 min) and write `llm_labels/items.jsonl`. Like every live `llm-label` run, it asks for confirmation before the first request (`--yes` for non-interactive use).
2. `python -m brand_study llm-label --dry-run` — counts per judge, pull status, Claude flags, and `llm_labels/dry_run_prompt.txt`. Calls no model.
3. `python -m brand_study llm-label` — runs gemma → qwen → claude and aggregates. Asks for confirmation before any homepage request or model call (`--yes` for non-interactive use). When Claude has pending work, the Claude CLI is located and probed before that prompt, so the plan shows its isolation flags and a missing or unsuitable CLI stops the run before any fetch or Ollama download. Useful options: `--judges claude`, `--limit 3` (trial run: never writes the accepted file), `--retry-abstentions`, `--aggregate-only` (always the full item set; cannot be combined with `--limit`). A transport error or HTTP 5xx from Ollama is retried once after 30 s; a second failure stops the run and deletes the model, and finished batches stay cached.
4. `python -m brand_study llm-sample` — draws a blinded, stratified sample of 50 items (fixed seed) into `results/llm_validation_sample.csv`. Fill `human_category` (and optionally `human_note`) **before** you open `llm_disagreements.csv`, and do not open `llm_labels/validation_snapshot.json`. Saving the file from Excel or LibreOffice is fine (BOM, CRLF, padded `#` lines and Excel's Windows-1252 "CSV (comma delimited)" are all read); keep the comma separator, and do not add, rename or reorder `sample_id`s. The sample scores the acceptance policy `llm-label` actually applied (read from the `# policy=` line of `llm_label_overrides.csv`); an `--accept` that differs from it is refused.
5. `python -m brand_study llm-agreement` — writes `results/llm_validation.md` and `results/llm_validation_overrides_snippet.csv`. If the bias check is inconclusive, `llm-sample --extend 30` adds items (existing labels are kept); label them and run `llm-agreement` again. Items you have since resolved in `label_overrides.csv` stay eligible for the extension (their evidence comes from `llm_labels/items.jsonl`), so each stratum is still drawn at random. Rows you leave blank are treated as missing at random within their stratum (weights use only the answered rows) and the report flags the results as preliminary; label every row before relying on them.
6. Append the snippet rows you agree with to `label_overrides.csv` (your labels then win everywhere).
7. Resolve `llm_disagreements.csv` by adding rows to `label_overrides.csv`.
8. `python -m brand_study llm-label --aggregate-only`, then `python -m brand_study analyze`.

`analyze` uses the accepted file automatically. `--no-llm-labels` ignores it, and `--llm-overrides PATH` (or env
`BRAND_STUDY_LLM_OVERRIDES`) points to another file.

**Runs.** Label once, after the last collection run. Items first seen in a later run, and items whose evidence
changes, become pending. Until `llm-label` finishes again, the previous accepted file stays in use, and `analyze`
and the report warn when its `runs=` header differs from the runs being analysed. `--reuse-stale-votes` aggregates
with each item's newest vote instead (marked `stale_evidence`).

**Validation and the bias test, in plain words.** The sample is stratified by the judges' majority category
(big, small, other) and whether they agreed, so it deliberately over-represents some groups. `llm-agreement` reports
per-judge accuracy and Cohen's κ, the accuracy of the labels aggregation actually accepted (frozen in the snapshot;
a unanimous vote sent to you because it conflicts with a candidate rule does not count) and of all unanimous-vote
items, pairwise agreement and Fleiss' κ,
and confusion matrices. The **big-brand bias check** is one test decided before you label: "the judges move
businesses you call small to big more often than the reverse" (one-sided, α = 0.05). It uses a sample-weighted
estimate (the weighted net shift) with a stratified bootstrap, because the raw counts from a stratified sample can
show an asymmetry that does not exist, or hide one that does. It needs at least 5 big↔small disagreements with you
to give any verdict. With 50 items, only a fairly strong bias is detectable. The unweighted McNemar numbers are shown
for description only.

**Cost and time** (estimates; the first real run will confirm them)

| Step | Estimate |
|---|---|
| Homepage fetch, about 283 domains | 5–10 min |
| Batches | 298 / 12 ≈ 25 per judge, plus a few small retries. A batch with only some items pending (e.g. after `--retry-abstentions`) is resent whole, so within one run over the same selection every judge sees byte-identical batches; this costs extra Claude usage for the already-answered items. Batches are built from the selection, so votes from a `--limit` trial (or a `--judges` subset run with a different `--limit`) were made with different batch neighbours than a later full run; for strictly identical prompts across judges, do trials in a scratch output directory or run every judge over the same selection |
| Gemma 4 31B | about 20 GB download (5–20 min), then about 0.5–2 h |
| Qwen 3.8 27B | about 18 GB download, similar time |
| Claude Opus 5.5, high effort | about 25 CLI calls of about 8–10k input tokens each, about 0.5–1.5 h. This counts against your Claude plan's usage limits; the run waits automatically when it hits one |
| Total | about 2–5 h of wall time. Your time: the 50 sample items plus the disagreements, about 30–60 min plus the disagreement count (`--extend 30` adds about 20 min) |
| Labeling before run2 | means another pull of both models and a partial rerun later |

**Limitations**

- LLMs may favour global brands over local ones, and all three judges can share that bias. The validation can show it, but with 50 items its power is limited, the CIs are wide, and the percentile bootstrap can undercover with small strata.
- Claude cannot run at temperature 0, and its isolation is limited to the flags the installed CLI supports (the user-level `CLAUDE.md`/auto-memory cannot be switched off without breaking the login, so the CLI may add a little context of its own; `--claude-diagnose` shows the token counts).
- Evidence comes from data already collected plus a single homepage (at most 3 redirects, no robots.txt check; a DNS-rebinding gap remains between the IP check and the request).
- LLM labels can confirm candidate AIO mentions (reported separately). An entity whose candidate rules differ between sectors always goes to you.
- Ollama tags can be re-pushed. The model digests and the Claude CLI version are recorded on every vote.

### Ollama model lifecycle helper (for Part 2)

`brand_study/ollama_lifecycle.py` downloads a model just before use and deletes it afterwards, to save space on
the shared disk:

```python
from brand_study import config
from brand_study.ollama_lifecycle import ollama_model

with ollama_model(config.REWRITER_MODEL) as tag:
    ...  # use the model through OLLAMA_BASE_URL
# deleted here, also after errors, Ctrl+C and SIGTERM
```

Safety rules:
- **A model that was already present is never deleted.** Only a model the helper pulled itself in this run is removed.
- Before pulling, it checks free disk space at the models directory and refuses if less than the margin (default 10 GB) would remain. If a models folder doesn't exist yet or can't be read, it measures the nearest existing parent folder instead, which may be on a different filesystem. On a shared server, set `BRAND_STUDY_OLLAMA_MODELS_DIR` to the server's models directory so the right disk is checked. If no directory can be checked, it refuses unless `allow_unknown_disk=True`.
- It reports free GPU memory with `nvidia-smi` and warns loudly when it is below 24 GB, for example because another job holds about 20 GB. Pass `strict_gpu=True` to refuse instead.
- It deletes through the HTTP API (`DELETE /api/delete`). If that fails and the model is still listed, it falls back to `ollama rm`, with `OLLAMA_HOST` derived from `OLLAMA_BASE_URL` (this overrides any inherited `OLLAMA_HOST`). If no CLI host can be derived (a proxy path prefix, or credentials in the URL), the fallback is disabled and the exact manual command is printed.
- Limits: if someone else pulls the same tag during your run, it is deleted too. On a remote server, the disk and GPU checks describe this machine only.

Part 2 model tags (not verified yet): the rewriter is `gemma4:31b` (env `STUDY_REWRITER_MODEL`), and the judge is a
**placeholder** (`STUDY_JUDGE_MODEL`, which must be set to the real Qwen 3.x 27B tag). The helper refuses
placeholder tags.

---

## Future Scope

### Near-term

- **Multi-iteration convergence tracking** — plot coverage score across all iterations to visualize where the loop plateaued and why
- **Revised markdown claim re-audit** — after content reviser produces improved markdown, re-run claim verifier to confirm no new unsupported claims were introduced
- **Persistent storage** — save run history and GEO reports to disk so comparisons survive session reloads
- **Batch processing** — accept a CSV of product URLs and run the full pipeline on each without manual tab-switching

### Medium-term

- **External GEO validation** — test structured markdown against real generative search systems (Perplexity API, SGE environments) to validate whether internal coverage scores correlate with real-world visibility
- **Retrieval evaluation** — BM25 or embedding-based test: does retrieving from structured markdown outperform raw webpage text? Metrics: Recall@1, Recall@3, MRR
- **RAG answer quality evaluation** — LLM-as-judge scoring on correctness, completeness, groundedness, and differentiator quality
- **LangGraph human-in-the-loop checkpoint** — add `interrupt_before=["revise_content"]` so a human can approve or modify selected strategies before each revision cycle

### Longer-term

- **Competitor analysis layer** — structured comparison against publicly available competitor product pages with conservative sourcing and review gates
- **Strategy performance tracking** — log which revision strategies produced the highest coverage improvements; use to prioritize strategies for future runs
- **CMS export** — push approved content files to a headless CMS via API
- **Ablation study** — formally compare four variants (one-shot / extract+write / extract+write+verify / full system with LangGraph loop) on a fixed product set with human rubric scoring
