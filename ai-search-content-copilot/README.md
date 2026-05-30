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
├── evaluate.py                       — CLI: automated quality checks
├── evaluate_baseline.py              — CLI: one-shot vs agentic comparison
│
├── outputs/
│   └── eval/                         — Saved markdown and audit files from CLI runs
│
├── .env                              — Model config (Ollama URL + model names)
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
