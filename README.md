# AI Search Content Copilot

> A two-layer agentic prototype that turns any public product webpage into structured, fact-grounded, AI-search-ready content — and then tests and iteratively improves that content using a GEO feedback loop.

---

## What This Is

**AI Search Content Copilot** is a human-in-the-loop agentic pipeline for product content teams. It takes a public product page URL, extracts and structures the product knowledge, drafts a reviewable markdown content file, audits every claim for factual support, and then runs a second optimization layer that simulates how generative AI search systems would represent the content and applies targeted revisions to improve coverage.

The system is designed around a core principle: **the goal is not to game AI search rankings. The goal is to build content that is more explicit, structured, grounded, and measurable** — so that when AI systems encounter it, they can represent the product accurately.

---

## Architecture: Two Layers

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Content Production                        │
│                                                       │
│  Product URL                                          │
│      ↓                                                │
│  [1] Webpage Extractor                               │
│      requests + BeautifulSoup + trafilatura           │
│      Playwright fallback for JS-heavy pages           │
│      → Evidence Bundle JSON                          │
│      ↓                                                │
│  [2] Product Research Agent  (qwen2.5:latest)        │
│      Extracts structured facts from evidence only     │
│      Attaches source quotes to every extracted point  │
│      → product_facts.json                            │
│      ↓                                                │
│  [3] Markdown Writer Agent  (qwen2.5:latest)         │
│      Writes 8-section markdown from facts only        │
│      Cannot invent claims not in the facts            │
│      → product_context.md                            │
│      ↓                                                │
│  [4] Claim Verifier Agent  (llama3.1:8b-instruct)   │
│      Independent model audits every claim             │
│      Labels: supported / partially_supported /        │
│               unsupported / needs_internal_confirm    │
│      → claim_audit.json                              │
│      ↓                                                │
│  [5] Gap Analysis Agent  (qwen2.5:latest)            │
│      Identifies missing info, unclear claims          │
│      Generates expert interview questions             │
│      → content_gaps.json + inventory_row.json        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  LAYER 2 — GEO Feedback Loop                         │
│                                                       │
│  [6] GEO Simulator Agent  (qwen2.5:latest)           │
│      Simulates RAG-style answers to eval questions    │
│      using only the markdown as context               │
│      Scores how well each question is answered (0-10) │
│      → sim_results (before)                          │
│      ↓                                                │
│  [7] Visibility Analyzer Agent  (qwen2.5:latest)     │
│      Compares simulated answers to full product facts │
│      Identifies typed, prioritized gaps:             │
│      missing_fact / weak_coverage / ambiguous_claim / │
│      no_differentiator / poor_structure               │
│      → visibility_gaps                               │
│      ↓                                                │
│  [8] Revision Strategy Agent  (qwen2.5:latest)       │
│      Selects content-specific strategies per gap:    │
│      add_specificity / surface_differentiators /      │
│      add_faq / clarify_ambiguous /                   │
│      improve_use_case_depth / consolidate_evidence /  │
│      add_structured_summary                           │
│      → selected_strategies                           │
│      ↓                                                │
│  [9] Content Reviser Agent  (qwen2.5:latest)         │
│      Applies strategies to produce improved markdown  │
│      Strictly evidence-grounded — no new claims       │
│      → revised_markdown                              │
│      ↓                                                │
│  [10] Re-simulation                                   │
│      Runs eval questions against revised content      │
│      → sim_results (after) + before/after delta      │
└─────────────────────────────────────────────────────┘
```

---

## Two-Model Design

| Model | Role | Why |
|---|---|---|
| `qwen2.5:latest` | Generator / Writer / Analyzer | Strong instruction following; used for all production and analysis steps |
| `llama3.1:8b-instruct` | Independent Reviewer | Different model for claim auditing prevents self-confirmation bias |

Both models run locally via **Ollama**. A Hugging Face Inference API backend is also supported for cloud deployment.

---

## Key Design Choice: Why Not One-Shot?

A naive approach sends the page text directly to an LLM and asks it to write the markdown. This prototype deliberately rejects that approach:

| Problem with one-shot | How this system solves it |
|---|---|
| No transparency — where did facts come from? | Evidence bundle keeps all source quotes; every extracted fact has an attached snippet |
| Hallucination risk — model invents claims | Research agent is instructed to return empty rather than invent; claim verifier catches anything that slips through |
| No claim accountability | Independent reviewer model audits every non-trivial claim in the draft |
| No structure enforcement | Pydantic schemas validate all outputs; pipeline fails loudly rather than silently degrading |
| Hard to evaluate | Automated quality checks + one-shot baseline comparison built in |
| Hard to improve | GEO feedback loop identifies specific gaps and applies targeted strategies |

**Measured result on tested product pages:**
- One-shot LLM: ~37% unsupported claims
- Agentic pipeline: ~0% unsupported claims

---

## Streamlit UI — 13 Tabs

Run on any public product URL. The sidebar accepts any URL — not restricted to any specific brand or domain.

| Tab | What You See |
|---|---|
| Extracted Evidence | Evidence bundle: paragraphs, bullets, headings, tables, JSON-LD |
| Product Facts | Structured facts JSON with source quotes per point |
| Markdown Draft | Side-by-side preview and raw markdown |
| Claim Audit | Per-claim audit: status, risk level, recommended action, progress bar |
| Content Gaps | Missing sections, unclear claims, expert interview questions |
| Inventory Row | One-row tracking record: file status, claim review status, next step |
| Evaluation Questions | 8 auto-generated product questions for downstream testing |
| **GEO Simulation** | **Before/after coverage score table per question; side-by-side simulated answers showing facts surfaced vs missed** |
| Visibility Gaps | Typed, prioritized gap list with the question that exposed each gap |
| Revision Strategies | Selected strategies with specific instructions for the reviser |
| Revised Content | Improved markdown with download; full GEO report JSON |
| **Baseline Comparison** | **One-shot LLM vs agentic pipeline: summary table, winner-by-metric, claim-level icons side by side** |
| **Product History** | **Multi-product comparison table accumulated across runs; CSV download** |

---

## Evaluation Framework

### Layer 1 — Automated quality checks (`evaluate.py`)

```bash
python3 evaluate.py "<product_url>"
```

| Check | Target |
|---|---|
| Schema validity | All required fields populated |
| Field coverage | Ingredients, differentiators, use cases, philosophy found |
| Evidence grounding | ≥ 70% of facts have source quotes |
| Claim audit sanity | Supported rate ≥ 60%, unsupported ≤ 20% |

### Layer 1 — Baseline comparison (`evaluate_baseline.py`)

```bash
python3 evaluate_baseline.py "<product_url>"
```

Runs one-shot LLM and full agentic pipeline on the same URL, scores both with the reviewer model, and prints a side-by-side comparison table. Saves all outputs to `outputs/eval/`.

### Layer 2 — GEO coverage (in the UI)

The **GEO Simulation tab** shows:
- **Coverage score (0–10) per question, before and after revision** — the primary GEO improvement signal
- **Delta column** — which questions improved, stayed the same, or got worse
- **Facts missed count before vs after** — concrete measure of retrieval completeness
- **Side-by-side simulated answers** per question

> **Honest limitation:** Coverage scores are self-assessed by the same LLM performing the simulation. This is an internal proxy metric, not a live test against Google AI Overviews, ChatGPT, or Perplexity. The before/after delta is meaningful as a relative comparison. External validation requires testing against real search systems.

---

## Repository Structure

```
ai-search-content-copilot/
│
├── app/
│   └── streamlit_app.py          — Streamlit UI (13 tabs, 2 pipeline stages)
│
├── src/
│   ├── webpage_extractor.py      — HTML fetch, BeautifulSoup parsing, evidence bundle
│   ├── llm_client.py             — Ollama + HuggingFace router with retry + JSON extraction
│   ├── schemas.py                — All Pydantic models (Layer 1 + Layer 2)
│   └── agents/
│       ├── product_research_agent.py    — Layer 1: extracts structured facts
│       ├── markdown_writer_agent.py     — Layer 1: writes 8-section markdown
│       ├── claim_verifier_agent.py      — Layer 1: audits claims (reviewer model)
│       ├── gap_analysis_agent.py        — Layer 1: finds gaps, generates questions
│       ├── geo_simulator_agent.py       — Layer 2: RAG simulation per eval question
│       ├── visibility_analyzer_agent.py — Layer 2: identifies typed, prioritized gaps
│       ├── revision_strategy_agent.py   — Layer 2: selects AutoGEO-inspired strategies
│       └── content_reviser_agent.py     — Layer 2: applies strategies, evidence-grounded
│
├── evaluate.py                   — CLI: automated quality checks (4 checks, scored)
├── evaluate_baseline.py          — CLI: one-shot vs agentic comparison
│
├── outputs/
│   └── eval/                     — Saved markdown and audit files from CLI runs
│
├── .env                          — Model config (Ollama URL + model names)
├── .env.example                  — Template
└── requirements.txt              — Python dependencies
```

---

## Setup and Running

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running locally
- Models pulled:
  ```bash
  ollama pull qwen2.5:latest
  ollama pull llama3.1:8b-instruct-q4_K_M
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure `.env`

```bash
OLLAMA_BASE_URL=http://localhost:11434
GENERATOR_MODEL=qwen2.5:latest
REVIEWER_MODEL=llama3.1:8b-instruct-q4_K_M
HF_TOKEN=                         # only needed for Hugging Face backend
```

### Run the app

```bash
python3 -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

Open **http://localhost:8501** in your browser.

> **For remote servers:** forward port 8501 in VSCode's Ports tab, then open http://localhost:8501 locally.

> **To keep the app alive across disconnects:**
> ```bash
> tmux new -s copilot
> python3 -m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
> # Ctrl+B then D to detach — tmux attach -t copilot to return
> ```

### Run CLI evaluation scripts

```bash
# Quality checks on any product URL
python3 evaluate.py "https://example.com/products/my-product"

# Agentic vs one-shot baseline comparison
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
| `GEOSimResult` | One simulated generative answer per eval question, with coverage score and facts surfaced/missed |
| `VisibilityGap` | One typed, prioritized content gap identified from simulation results |
| `RevisionStrategy` | One targeted revision strategy with a concrete instruction for the writer |
| `GEOFeedbackReport` | Full Layer 2 output: sim results before/after, gaps, strategies, revised markdown, coverage deltas |

---

## Research Foundation

This prototype is informed by four bodies of work on Generative Engine Optimization:

- **GEO** (Aggarwal et al.) — introduced the framework for measuring and improving content visibility in generative engines
- **AutoGEO** — explored how LLMs can learn generative engine preferences and rewrite content systematically; the revision strategy taxonomy in this project draws from AutoGEO's strategy inventory
- **AgenticGEO** — explored adaptive, multi-step optimization using agentic reasoning and feedback-driven refinement; the two-layer architecture and iterative simulation-revision loop in this project is directly inspired by this work
- **SAGEO Arena** — argued that realistic GEO evaluation must consider the full retrieval + reranking + generation pipeline, not just rewritten text in isolation; this influenced the decision to simulate full RAG-style answers rather than scoring text quality alone

The prototype applies these research ideas to a practical marketing workflow with a responsible framing: structured content production, human-in-the-loop review, and controlled evaluation rather than manipulation.

---

## Responsible Use

This system is designed to make product content **more accurate and trustworthy**, not to manipulate AI search systems.

Hard constraints baked into every agent:

1. Agents extract only from provided evidence — they cannot invent claims
2. A separate reviewer model audits every non-trivial claim independently
3. Uncertain or unverifiable claims are escalated, not smoothed over
4. The content reviser is explicitly forbidden from adding claims not already in the product facts
5. The gap analysis generates questions for human experts — it does not fill gaps with guesses
6. Coverage scores are presented as internal proxies, not as real-world ranking signals

---

## Future Scope

### Near-term improvements

- **Multi-iteration GEO loop** — run simulator → analyzer → reviser → simulator as a loop until coverage score plateaus or a maximum number of iterations is reached; currently the loop runs once
- **Persistent storage** — save run history and GEO reports to disk so comparisons survive session reloads
- **Batch processing** — accept a CSV of product URLs and run the full pipeline on each; produce a multi-product comparison report without manual tab-switching
- **Revised markdown claim re-audit** — after the content reviser produces the improved markdown, re-run the claim verifier to confirm the revision did not introduce unsupported claims

### Medium-term

- **External GEO validation** — test the structured markdown files against real generative search systems (Perplexity API, SGE test environments) to validate whether internal coverage scores correlate with real-world visibility
- **Retrieval evaluation** — build a small BM25 or embedding-based retrieval test: given product questions, does retrieving from structured markdown outperform retrieving from raw webpage text? Metrics: Recall@1, Recall@3, MRR
- **RAG answer quality evaluation** — use an LLM-as-judge to score generated answers on correctness, completeness, groundedness, and differentiator quality when using raw text vs structured markdown as context
- **Company-level knowledge layer** — inject a brand knowledge file (mission, quality standards, sustainability principles) as additional context for the research and writing agents

### Longer-term

- **Competitor analysis layer** — structured comparison against publicly available competitor product pages, with conservative sourcing and legal/brand review gates
- **Reviewer feedback loop** — capture human edits to the generated markdown and use them to refine agent prompts over time
- **CMS export** — push approved content files directly to a headless CMS or content platform via API
- **Strategy performance tracking** — log which revision strategies produced the highest coverage score improvements across products; use this to prioritize strategies for future runs
- **Ablation study** — formally compare four system variants (one-shot / extract+write / extract+write+verify / full system) on a fixed product set with human rubric scoring and retrieval metrics, following the evaluation framework outlined in the original design document

---

## Example Results

Tested on a Life's Abundance canned cat food product page:

| System | Sections Found | Supported Claims | Unsupported Claims |
|---|---|---|---|
| One-shot LLM | 8/8 | 63.2% | 36.8% |
| Agentic Pipeline | 8/8 | **81.8%** | **0.0%** |

The evidence-grounding step in the Product Research Agent is the primary driver of this improvement: the model cannot invent claims if the prompt contains only verified page content.
