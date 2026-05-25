# AI Search Content Copilot — Process Flow

A practical guide to what was built, how it works, and how to run it.

---

## 1. What This System Does

Takes a public product webpage URL and runs it through a multi-step agentic pipeline to produce:

- Structured product facts (JSON)
- A clean markdown content file
- A claim audit report (which facts are supported vs unsupported)
- Content gap analysis and internal expert questions
- A baseline comparison showing whether the agentic approach is better than a simple one-shot LLM call

---

## 2. System Components

```
Remote Server (GPU machine)
├── Ollama                        — runs LLMs locally on GPU
│   ├── qwen2.5:7b-instruct       — generator model (research, writing, gap analysis)
│   └── llama3.1:8b               — reviewer model (claim verification)
│
├── Streamlit app (port 8501)     — visual UI for running the pipeline
│
├── evaluate.py                   — automated quality checks (CLI)
└── evaluate_baseline.py          — one-shot vs agentic comparison (CLI)

Your Laptop
└── Browser                       — accesses the UI via VSCode port forwarding
```

---

## 3. How the Pipeline Works (Step by Step)

```
Product URL
    │
    ▼
[1] Webpage Extractor
    - Fetches HTML (with Playwright fallback for JS-heavy pages)
    - Extracts: title, meta description, headings, paragraphs,
      bullets, tables, JSON-LD structured data, raw text
    - Saves everything as an Evidence Bundle (JSON)
    │
    ▼
[2] Product Research Agent  (qwen2.5:7b-instruct)
    - Reads the Evidence Bundle
    - Extracts structured facts: product name, category, summary,
      formulation philosophy, key ingredients, differentiators,
      use cases, competitive positioning
    - Attaches source evidence quotes to each fact
    - Flags claims needing review and missing information
    - Output: product_facts.json
    │
    ▼
[3] Markdown Writer Agent  (qwen2.5:7b-instruct)
    - Reads product_facts.json
    - Writes a clean markdown file with 8 required sections:
        Product Overview, Formulation Philosophy, Key Ingredients,
        Differentiators, Use Cases, Competitive Positioning,
        Claims Requiring Review, Open Questions for Internal Experts
    - Never invents claims not present in the facts
    - Output: product_context.md
    │
    ▼
[4] Claim Verification Agent  (llama3.1:8b)
    - Reads the Evidence Bundle + markdown draft
    - Breaks markdown into atomic claims
    - Labels each claim: supported / partially_supported /
      unsupported / needs_internal_confirmation
    - Assigns risk level (low / medium / high)
    - Recommends action: keep / soften / seek_internal_confirmation / remove
    - Output: claim_audit.json
    │
    ▼
[5] Gap Analysis Agent  (qwen2.5:7b-instruct)
    - Reads product_facts.json
    - Identifies missing sections, unclear claims
    - Generates interview questions for internal product experts
    - Output: content_gaps.json
    │
    ▼
[6] Inventory Builder  (built into pipeline)
    - Creates a one-row tracking record:
      product name, line, URL, file status, claim review status,
      missing info count, unsupported claim count, next step
    - Output: inventory_row.json
```

---

## 4. Two Models, Two Roles

| Model | Role | Why |
|---|---|---|
| `qwen2.5:7b-instruct` | Generator | Strong instruction following, good at structured extraction and long-form writing |
| `llama3.1:8b` | Reviewer | Acts as independent critic — using a different model reduces self-confirmation bias |

The reviewer model never writes content. It only audits what the generator wrote.

---

## 5. How to Run — Every Time You Come Back

### Step 1 — Connect to the server (VSCode)
1. Open VSCode → connect to remote server
2. Open the **Ports** tab → forward port `8501`

### Step 2 — Check if the app is already running
```bash
ps aux | grep streamlit | grep -v grep
```

### Step 3 — If not running, start it
```bash
cd /home/kamruzzaman1/ai_search/ai-search-content-copilot
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true &
```

**Tip:** Use tmux so the app survives disconnects:
```bash
tmux new -s copilot
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
# Ctrl+B then D to detach
# Next time: tmux attach -t copilot
```

### Step 4 — Open the UI
Go to **http://localhost:8501** in your browser.

### Step 5 — Run the pipeline
1. Sidebar already shows: `Ollama (local)`, `qwen2.5:7b-instruct`, `llama3.1:8b`
2. Paste a Life's Abundance product URL
3. Optionally select a product line hint
4. Click **Run Agentic GEO Workflow**
5. Results appear across 7 tabs:
   - Extracted Evidence
   - Product Facts
   - Markdown Draft
   - Claim Audit
   - Content Gaps
   - Inventory Row
   - Evaluation Questions

---

## 6. How to Evaluate — After Running the UI

Run both scripts in the terminal on the server using the **same URL** you used in the UI:

```bash
cd /home/kamruzzaman1/ai_search/ai-search-content-copilot

# Automated quality checks (4 checks, scored 0-4)
python3 evaluate.py "<product URL>"

# Side-by-side comparison: one-shot LLM vs full agentic pipeline
python3 evaluate_baseline.py "<product URL>"
```

### What each script checks

**evaluate.py — 4 automated checks:**

| Check | What it tests | Target |
|---|---|---|
| Schema validity | Are required fields populated? | All pass |
| Field coverage | Ingredients, differentiators, use cases, philosophy found? | All present |
| Evidence grounding | Are facts backed by source quotes? | ≥ 70% |
| Claim audit sanity | Supported rate ≥ 60%, unsupported ≤ 20%? | Both pass |

**evaluate_baseline.py — comparison table:**

| Metric | One-shot | Agentic |
|---|---|---|
| Sections found | x/8 | x/8 |
| Supported claims % | x% | x% |
| Unsupported claims % | x% | x% |
| Needs internal review | x | x |

Outputs saved to `outputs/eval/` for manual side-by-side review.

---

## 7. Example Results (Instinctive Choice Cat Food)

URL tested:
```
https://lifesabundance.com/Pets/InstinctiveChoice/InstinctiveChoiceWhy.aspx
```

| Metric | One-shot | Agentic |
|---|---|---|
| Sections found | 8/8 | 8/8 |
| Supported claims % | 63.2% | **81.8%** |
| Unsupported claims % | 36.8% | **0.0%** |

**Key finding:** The agentic pipeline had zero unsupported claims vs 36.8% for one-shot,
because the evidence-grounding step prevents the model from inventing claims.

---

## 8. Project File Structure

```
ai-search-content-copilot/
│
├── app/
│   └── streamlit_app.py          — Streamlit UI (7 tabs)
│
├── src/
│   ├── webpage_extractor.py      — HTML fetch + evidence bundle builder
│   ├── llm_client.py             — Ollama + HuggingFace LLM router
│   ├── schemas.py                — Pydantic models for all outputs
│   └── agents/
│       ├── product_research_agent.py   — extracts structured facts
│       ├── markdown_writer_agent.py    — writes markdown content file
│       ├── claim_verifier_agent.py     — audits claims against evidence
│       └── gap_analysis_agent.py       — finds gaps, generates questions
│
├── evaluate.py                   — automated quality checks (CLI)
├── evaluate_baseline.py          — one-shot vs agentic comparison (CLI)
│
├── outputs/
│   └── eval/                     — saved markdown and audit files
│
├── .env                          — model config (Ollama URLs and model names)
└── requirements.txt              — Python dependencies
```

---

## 9. Environment Configuration (.env)

```bash
OLLAMA_BASE_URL=http://localhost:11434
GENERATOR_MODEL=qwen2.5:7b-instruct
REVIEWER_MODEL=llama3.1:8b
HF_TOKEN=...                      # only needed for Hugging Face backend
```

To switch to Hugging Face models instead of Ollama, change the model names to
include a `/` (e.g. `Qwen/Qwen3-8B`) and select "Hugging Face" in the UI sidebar.

---

## 10. Why Agentic Instead of One-Shot?

| Problem with one-shot | How this system solves it |
|---|---|
| No transparency — where did facts come from? | Evidence bundle keeps all source quotes |
| Hallucination risk — model invents claims | Research agent only uses provided evidence |
| No claim review | Independent reviewer model audits every claim |
| No structure | Pydantic schemas enforce consistent output format |
| Hard to evaluate | Automated checks + baseline comparison built in |
| Hard to maintain | Each agent has a single clear responsibility |

---

## 11. How This Prototype Connects to the AI Search Optimization Intern Role

This prototype was built specifically to think concretely about the work described in this
internship. Every feature maps directly to something in the job description.

---

### Role Responsibility → What This Prototype Does

| What the Role Asks For | How This Prototype Addresses It |
|---|---|
| Write structured markdown content files for each product | The Markdown Writer Agent produces a clean, consistently structured markdown file for any product URL in seconds |
| Cover formulation philosophy, key ingredients, differentiators, use cases, competitive positioning | These are the exact 8 sections the agent is instructed to fill — they match the role description word for word |
| Research Life's Abundance products across all lines | The Product Research Agent extracts structured facts from any public Life's Abundance page automatically |
| Audit existing website pages | The Claim Verification Agent audits every factual statement in the draft — flagging unsupported, overstated, or risky claims before human review |
| Interview internal team members to surface hidden product knowledge | The Gap Analysis Agent generates specific, ready-to-use interview questions for product experts based on what the page does not explain |
| Maintain a content inventory tracking file status | The Inventory Builder produces a one-row tracking record per product: file status, claim review status, missing info count, recommended next step |
| Prioritize which products need AI context files first | The inventory row includes a missing information count and unsupported claim count — making prioritization data-driven |
| Participate in review cycles to sharpen content quality | The claim audit report gives a reviewer a structured, claim-by-claim breakdown with recommended actions: keep, soften, seek confirmation, or remove |
| Collaborate with SEO intern to align AI and traditional search | Structured markdown with clear headings, grounded facts, and consistent schema benefits both AI retrieval and traditional on-page SEO |

---

### What This Prototype Demonstrates About the Candidate

**Research skill:**
The system does not guess. Every extracted fact is attached to a source quote from the page.
The gap analysis explicitly lists what information was missing — the same judgment a careful
researcher would apply before publishing.

**Writing and structure:**
The markdown output follows a fixed, reviewable structure. Each section is written in
concise, non-hyped prose. Claims flagged for review are marked explicitly rather than
smoothed over.

**Factual accuracy and claim discipline:**
An independent reviewer model audits every non-trivial claim in the draft. In testing on a
Life's Abundance canned cat food page, the agentic pipeline produced 0% unsupported claims
vs 36.8% for a naive one-shot approach. The system is designed to escalate uncertainty
rather than hide it.

**Curiosity about AI and GEO:**
The project is grounded in published GEO research (Aggarwal et al., AutoGEO, SAGEO Arena,
AgenticGEO) but applies it to a real marketing workflow rather than an academic benchmark.
The framing is deliberately responsible: the goal is content readiness, not ranking
manipulation.

**Ability to work independently and hit targets:**
The pipeline runs end-to-end from a single URL input. A human reviewer receives a complete
package — structured facts, markdown draft, claim audit, gap questions, inventory row —
without needing to prompt or guide each step.

**Comfort with new tools:**
The system was built using small open-source models (Qwen2.5, Llama 3.1) running locally
on GPU via Ollama, with a Streamlit UI accessible from any browser. The evaluation
framework includes automated scoring and a one-shot baseline comparison to prove the
multi-step design actually adds value.

---

### Interview Talking Points

> "I built this prototype to think concretely about how I would approach the internship.
> Rather than describing what I would do, I wanted to show it. The system takes a Life's
> Abundance product page, extracts evidence, converts it into the exact structured content
> the role describes — formulation philosophy, ingredients, differentiators, use cases —
> and then audits its own claims before handing off to a human reviewer.
>
> I kept the system grounded in public source content only. I used a separate reviewer
> model to reduce unsupported statements. And I built an evaluation framework to test
> whether the structured files are actually better than a simpler one-shot approach —
> because I think being able to measure your own work is part of doing it well.
>
> I would not claim this replaces the intern. The gap analysis agent is specifically
> designed to surface the questions that need to go to internal product experts — because
> that human knowledge is exactly what the role exists to capture."

---

### What This Prototype Does Not Claim

- It does not guarantee better visibility in ChatGPT, Claude, Gemini, or Perplexity.
- It does not replace human reviewers, product experts, or brand judgment.
- It does not handle internal documents, proprietary formulation data, or approved
  competitive comparisons — those require the human intern to gather and verify.
- It is a drafting and quality-control assistant, not a final authority.

These boundaries are intentional. The most valuable thing an AI Search Optimization Intern
can do is produce well-structured, factually grounded, reviewable content — not automate
away the judgment that makes that content trustworthy.
