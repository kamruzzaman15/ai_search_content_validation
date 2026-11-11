# Task: Part 1, the small-vs-big brand visibility study (Google search vs Google AI Overviews)

## Context
This repository (`ai-search-content-copilot`) already has a GEO content pipeline (`src/`, `app/`, `evaluate*.py`).
We are adding a new, self-contained study module. Part 1 measures the problem:
**Are small/local businesses less visible than big brands in Google organic results and in Google AI Overviews?**
Part 2 (rewriting pages with Ollama models and judging them) is OUT OF SCOPE for this task, except for the
Ollama model lifecycle helper described below, which Part 2 will reuse.

## Decisions already made by the user
- Sectors: **coffee shops** and **skincare**.
- Location: **Florida, US**.
- Data source: **only Google results via SerpApi**, meaning organic top 10 and AI Overviews. No ChatGPT/Gemini APIs.
- Later Ollama models: rewriter **Gemma 4 31B**, judge **Qwen 3.x 27B**. The exact Ollama tags are not yet
  verified. Make them configurable, with `gemma4:31b` as the rewriter default and a clearly marked placeholder for the Qwen tag.

## Requirements

1. **Query set.** About 50 realistic commercial questions, roughly 25 coffee shops and 25 skincare, mixing broad
   queries ("best moisturizer for oily skin") and Florida-local ones ("best coffee shop in Tampa"). Spread them over
   several Florida cities (for example Miami, Tampa, Orlando, Jacksonville). Store them as a versioned data file with
   id, sector, query, query_type (broad/local), and city where relevant.

2. **SerpApi collection.**
   - Read the key from the `SERPAPI_API_KEY` environment variable or `.env` via python-dotenv. Never print, log or write the key.
   - Use the Google engine with US/English settings and a Florida location. Collect the organic top 10 and the AI Overview.
     When the AI Overview comes back only as a `page_token`, make the follow-up request
     (Google AI Overview API) to fetch it. Verify parameter and field names against SerpApi's documentation
     (serpapi.com/search-api, serpapi.com/google-ai-overview-api). Handle queries with no AI Overview.
   - Support **2 runs per query** (a run label such as `run1`/`run2`, meant to be collected at different times).
   - **Cache every raw JSON response on disk** and never re-request a cached query, run or request pair.
   - **Budget guard:** before collecting, call SerpApi's free Account API, estimate the searches needed, and refuse to
     start if the estimate exceeds a configurable cap (default 300) or the remaining searches. Report searches used.
   - A dry-run mode that shows what would be requested without spending searches.

3. **Extraction.** From each response, extract domains/brands in the organic top 10 (with rank), and the brands mentioned
   plus the sources/links cited in the AI Overview. Normalise domains (for example strip `www.`).

4. **Brand labeling (big vs small/local).** A transparent rule-based first pass, for example a curated list of
   national/global chains and brands, marketplaces, and directories/review sites such as Yelp, TripAdvisor and Reddit
   as a separate category. Add a manual override CSV so the user can correct labels. Categories should at least
   distinguish: big brand, small/local business, marketplace/retailer, directory/review/media/other.

5. **Analysis (Part 1 results).** Per sector and query type, report:
   - how often big vs small/local businesses appear in the organic top 10 and in AI Overviews;
   - overlap between domains cited in AI Overviews and the organic top 10 for the same query;
   - the share of queries that show an AI Overview;
   - run-to-run stability (for example Jaccard similarity between run1 and run2) when both runs exist.
   Write tidy CSVs and a short Markdown summary report. State clearly in the report that results are observational
   (one location, one point in time, and personalisation/location effects apply).

6. **Ollama model lifecycle helper (for Part 2).** The user wants each Ollama model **downloaded before use and
   deleted after the run finishes** to free disk space (the disk is about 93% full and shared). Download it again next time.
   - Context-manager style: pull → use → `ollama rm` in a `finally` block, so it also deletes on errors and Ctrl+C.
   - Before pulling, check free disk space (refuse if below a configurable margin) and report free GPU memory.
     The GPU is shared and another user's job may hold about 20 GB, so warn clearly when it is not enough.
   - Never delete a model the helper did not pull itself in that run, so it doesn't remove someone else's model.
   - Unit-test it with the Ollama calls mocked. Do not actually pull large models in this task.

7. **Entry points and docs.** Simple CLI commands (for example `python -m ... collect --run run1 --dry-run`,
   `... analyze`). Add a README section explaining the steps, the costs in SerpApi searches, and the limitations.

8. **Tests.** pytest tests using saved/mock SerpApi responses. **No network in tests.**

## Constraints
- Python 3.9 compatible, since the machine runs Python 3.9.12. Follow the existing code style (`from __future__ import annotations`,
  small functions, sparse comments). Prefer existing dependencies (requests, pandas, python-dotenv).
- Live SerpApi calls during implementation: **at most 5 searches in total** for a smoke test, and only after the dry
  run works. Do not run the full collection. The user will run it.
- Do not modify `.env`. Do not commit or push. Do not delete or rewrite existing pipeline code.
- Keep generated data (raw cache, results) out of Git unless they are small, curated input files such as the query list
  or the brand list.
