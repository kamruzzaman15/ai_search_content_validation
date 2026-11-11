# Plan: Part 1, the small-vs-big brand visibility study (Google organic vs AI Overviews), revision 3

## What changed since round 2

- **Retries and the budget (issue 1):** the ledger now has one line per HTTP attempt. `RuntimeBudget` is charged for each attempt, retries only happen if budget is left after the reserve, and the estimate adds a separate retry headroom. One invariant holds throughout: the number of attempts ever made is at most `min(study cap − earlier ledger total, account searches left)`. A test checks this with scripted 5xx responses near the limit.
- **`ollama rm` fallback (issue 2):** the subprocess runs with `OLLAMA_HOST` set from `OLLAMA_BASE_URL`, which overrides any inherited value. If the base URL can't be turned into a CLI host, the fallback is turned off. The fallback is also skipped when `list_tags()` shows the tag is already gone. Tests cover both.
- **Bias against small businesses before review (issue 3):**
  - Every share now comes with lower and upper bounds. The upper bound for `small_local` counts both unknown items and unconfirmed candidates as small.
  - A `provisional` flag and an `unresolved_share` column are added to `summary_by_segment.csv`.
  - The report opens with a banner whenever results are provisional. The main big-vs-small table shows unknown/unconfirmed counts and the bounds in the same row.
  - A new confirmation method, `local_pack_name`, confirms an AIO mention when it matches a local-pack business for the same query.
  - `labels_to_review.csv` suggests `small_local` for local-pack websites. These are suggestions only and are never applied automatically.
- **Ambiguous aliases (issue 4):** the rule file gets an `ambiguous` column with the values `no`, `case`, `case_mid` and `candidate`. Tests cover "target dark spots", "Target dark spots…" at the start of a sentence, "an ordinary moisturizer" and "silly".
- **Key-leak check (issue 5):** a new offline command, `python -m brand_study scan-leaks`. It loads the key through dotenv, refuses keys that are empty or too short, and prints only file names and a count.
- **Test layout (issue 6):** tests move to `tests/test_brand_study/`. No test directory has an `__init__.py`, and test file basenames are unique.
- **Smaller points (issue 7):**
  - `--max-searches` is renamed `--study-cap` and is documented as covering the whole study. A per-invocation `--max-new` is added. Dry-run prints the ledger total.
  - Operational big-vs-small rules for broad queries, with an auditable `basis` column.
  - The FakeSession key includes `location`.
  - A local query no longer needs the city in its text.
  - Zero-denominator behaviour for `domain_share` is defined.

---

## 0. Goal, and how the docs get checked

**Goal:** add a self-contained module that can be tested offline. It:
1. collects the Google organic top 10 and AI Overviews (AIO) through SerpApi for about 50 Florida commercial queries, over 2 runs, within a strict budget;
2. extracts and normalises domains and brands;
3. labels each one with transparent rules plus user overrides;
4. writes tidy CSVs and a Markdown report.

It also adds a reusable Ollama model lifecycle helper for Part 2.

**Checking the docs (the implementer's first step, before writing any client code):**
1. Read serpapi.com/search-api, /ai-overview, /google-ai-overview-api, /account-api, /organic-results, /local-results, /locations-api and /pricing (or the FAQ on billing), using WebFetch.
2. If WebFetch is denied, use the `web-fetch` agent with the same URLs.
3. If both fail, **tell the user explicitly** and carry on. Field names are then checked only against the smoke-test responses, and the README and report say "field names verified against live responses only, not against the documentation".

Facts to confirm and record in `brand_study/SERPAPI_NOTES.md`, with the source URL and the date read:
- `page_token` expiry. I recall about 4 minutes, which is why the follow-up is made immediately.
- Whether a `google_ai_overview` follow-up is billed as a search.
- Whether searches that return an error, "no results", 429 or 5xx are billed. This only sets the `charged_assumed` annotation. The guard always counts every attempt (step 7).
- What a 429 means on SerpApi: a throughput limit or an exhausted plan, and what body text tells them apart.
- Whether `no_cache=true` costs the same as a normal search.
- Whether `num` is deprecated or ignored.
- The canonical location strings, through the free Locations API, especially St. Petersburg and Fort Lauderdale.
- Account API field names.
- The shapes of `ai_overview` (`text_blocks` types, `references`, `reference_indexes`) and `local_results` (`places`, `website`, `title`).

Each such assumption below is marked **[VERIFY]**.

### What the repo looks like (relevant facts)
- The git root is `ai_search_content_validation/`, and the project lives in `ai-search-content-copilot/`.
  - There is **no `.gitignore`**.
  - `.env` and `.env.example` are excluded only through `.git/info/exclude`, which we leave alone.
  - The repo already tracks `outputs/eval/*` and `__pycache__/*.pyc`, and we leave those untouched too.
  - There is no `tests/` directory.
- Code style (`src/llm_client.py`): `from __future__ import annotations`, small functions, `requests` imported directly, and Ollama at `OLLAMA_BASE_URL` (default `http://localhost:11434`, `src/llm_client.py:14`) through its HTTP API.
- Existing code calls `load_dotenv`, sometimes with `override=True`. Our module calls it without `override`, so a key set in the shell wins and `.env` is never written.
- Dependencies are requests, pandas>=2.0 and python-dotenv.
  - pytest is not in `requirements.txt`. Whether it is installed is checked with `python -m pytest --version`. If it is missing, ask the user before installing.
  - `tabulate` is not listed, so we write our own table formatter.

---

## 1. Files to create or change

All paths are relative to `ai-search-content-copilot/`. No existing pipeline file changes, apart from a new README section.

```
brand_study/
  __init__.py
  __main__.py            # python -m brand_study ...
  cli.py                 # argparse; main(argv=None, load_env=True)
  config.py              # paths, defaults, categories, env helpers, Part 2 model tags
  queries.py             # load + validate versioned query file
  serpapi_client.py      # request builders, single-attempt HTTP, key redaction
  cache.py               # cache paths, atomic writes, scrubbing, lock file, supersede
  budget.py              # Account API parsing, estimate, guard, per-attempt runtime budget, ledger
  collect.py             # plan + collection loop (retry policy lives here, budget-aware)
  leaks.py               # scan files for the API key; names only
  domains.py             # URL / domain normalisation
  entities.py            # entity_norm(), entity_norm_cased(), alias matching with ambiguity modes
  extract.py             # raw JSON -> tidy rows (status, organic, AIO refs, AIO mentions, local pack)
  labeling.py            # rules + overrides -> category, label_source, confirmed
  analyze.py             # metrics (shares, bounds, provisional flag, overlap, AIO rate, stability, Wilson CIs)
  report.py              # Markdown report (own small table formatter)
  ollama_lifecycle.py    # Part 2 helper: pull -> use -> rm in finally
  SERPAPI_NOTES.md       # verified doc facts + source/date (written in step 0)
  data/
    queries_v1.csv       # curated, tracked
    brand_rules_v1.csv   # curated, tracked
    label_overrides.csv  # user-edited, tracked (header + commented examples)
tests/                   # NO __init__.py anywhere under tests/
  conftest.py            # sys.path insert of project root, no-network guard, load_dotenv blocked, fake key, tmp output dir
  test_brand_study/      # name differs from the package to avoid shadowing
    fixtures/
      search_inline_aio.json
      search_page_token_only.json
      search_no_aio.json
      search_aio_error.json
      search_no_results.json
      search_few_organic_local_pack.json
      search_ambiguous_aliases.json   # "target dark spots", "an ordinary moisturizer", local-pack names in AIO
      ai_overview_followup.json
      ai_overview_followup_error.json
      account.json       # fake values, incl. fake api_key/email to test scrubbing
    test_bs_queries.py
    test_bs_domains.py
    test_bs_entities.py
    test_bs_serpapi_client.py
    test_bs_cache.py
    test_bs_budget.py
    test_bs_collect.py
    test_bs_extract.py
    test_bs_labeling.py
    test_bs_analyze.py
    test_bs_report.py
    test_bs_leaks.py
    test_bs_cli.py
    test_bs_ollama_lifecycle.py
.gitignore               # NEW: outputs/brand_study/, .pytest_cache/
requirements-dev.txt     # NEW: pytest>=7
README.md                # ADD section "Brand Visibility Study (Part 1)"
```

The `test_bs_` prefix keeps basenames unique, which matters in pytest's default `prepend` import mode when there is no `__init__.py`. That way, future tests elsewhere in the repo can't collide with these. `tests/conftest.py` puts `PROJECT_ROOT` at the front of `sys.path`, so `import brand_study` always resolves to the package.

Generated output, ignored by Git:
```
outputs/brand_study/
  raw/<run>/<query_id>.search.json
  raw/<run>/<query_id>.ai_overview.json
  raw/<run>/_superseded/<query_id>.<kind>.<collected_at>.json
  ledger.jsonl
  .collect.lock
  results/
    query_status.csv
    organic_results.csv
    aio_references.csv
    aio_mentions.csv
    local_pack.csv
    summary_by_segment.csv
    overlap_by_query.csv
    stability_by_query.csv
    labels_to_review.csv
    report.md
```

`__pycache__/` is left out of the new `.gitignore` on purpose. Some `.pyc` files are already tracked, and ignoring the directory would hide changes to them. The implementer tells the user that new `brand_study/__pycache__` files will show as untracked.

The study is a top-level `brand_study/` package, separate from the GEO pipeline in `src/`. `python -m brand_study` works from the project directory without packaging.

---

## 2. Step-by-step implementation

### Step 1: `config.py`
**Paths**
- `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
- `DATA_DIR`
- `OUTPUT_DIR = PROJECT_ROOT / "outputs/brand_study"`, which env `BRAND_STUDY_OUTPUT_DIR` can override.

**SerpApi defaults**
- `SERPAPI_ENDPOINT = "https://serpapi.com/search.json"`
- `ACCOUNT_ENDPOINT = "https://serpapi.com/account.json"`
- `GL="us"`, `HL="en"`, `GOOGLE_DOMAIN="google.com"`, `DEVICE="desktop"`
- `STATE_LOCATION = "Florida,United States"`
- `CITY_LOCATIONS`, filled from the Locations API **[VERIFY]**

**Budget defaults**
- `DEFAULT_STUDY_CAP = 300`, overridable with env `BRAND_STUDY_CAP`. It is study-wide and includes every earlier ledger attempt, smoke runs included.
- `DEFAULT_FOLLOWUP_RATE = 1.0` (worst case).
- `DEFAULT_RETRIES = 1`, so each logical request makes at most 2 attempts.
- `DEFAULT_RETRY_HEADROOM = 10` searches per invocation.
- `DEFAULT_SLEEP_S = 1.0`
- `DEFAULT_MIN_HOURS_BETWEEN_RUNS = 12`

**Categories**
- `CATEGORIES = ("big_brand", "small_local", "marketplace_retailer", "directory_media_other", "not_a_brand", "unknown")`
- `HEADLINE_EXCLUDED = ("not_a_brand",)`
- `CATEGORY_DEFINITIONS` holds the text used in the README and report (step 10), including the operational big-vs-small test.

**Analysis defaults**
- `DEFAULT_PROVISIONAL_THRESHOLD = 0.10`, compared against the unresolved share (unknown plus unconfirmed).

**Runs**
- `DEFAULT_RUNS = ("run1", "run2")`
- `RUN_LABEL_RE = r"^[a-z0-9_-]{1,32}$"`

**API key**
- `get_api_key(required=True)` reads only `os.environ["SERPAPI_API_KEY"]`.
- `.env` is loaded only in `cli.main()` when `load_env=True`, with `load_dotenv(path)` and no `override`.
- A missing key raises `SystemExit("SERPAPI_API_KEY not set (env or .env)")`. The key's value never appears in any message.

**Ollama**
- `OLLAMA_BASE_URL` uses the existing env var and defaults to `http://localhost:11434`.
- `OLLAMA_MODELS_DIR = os.environ.get("BRAND_STUDY_OLLAMA_MODELS_DIR")`

**Part 2 models**
- `REWRITER_MODEL = os.environ.get("STUDY_REWRITER_MODEL", "gemma4:31b")`, commented as unverified.
- `JUDGE_MODEL = os.environ.get("STUDY_JUDGE_MODEL", "REPLACE_ME_qwen3_27b_tag")`, commented `# PLACEHOLDER: set the real Qwen 3.x 27B Ollama tag`.
- `is_placeholder_tag(tag)` returns True for tags starting with `REPLACE_ME`.

### Step 2: the query set, `data/queries_v1.csv` and `queries.py`
The columns are `id, sector, query, query_type, city, notes`.
- `sector` is `coffee` or `skincare`. `query_type` is `broad` or `local`. `city` is empty for broad queries.
- ids are stable and never reused: `cof_b_01..`, `cof_l_01..`, `skn_b_01..`, `skn_l_01..`. Changing a query's text means giving it a new id.

**Location rule** (documented in the README and report):
- Broad queries use `location="Florida,United States"`.
- Local queries use the city location.
- The city name usually appears in the query text. "Near me" queries rely on the location alone, so the same `q` text can appear under different ids and cities. This is noted in `notes`.

Cities: Miami, Tampa, Orlando, Jacksonville, St. Petersburg, Fort Lauderdale.

Draft list (25 + 25). The user can edit it.

**Coffee, broad (10):**
- best coffee beans for espresso
- best coffee subscription
- best cold brew coffee brand
- best decaf coffee beans
- where to buy specialty coffee beans online
- best medium roast whole bean coffee
- best cuban coffee brand
- best coffee for french press
- best organic coffee brand
- best single origin coffee to try

**Coffee, local (15):**
- best coffee shop in Tampa
- best coffee shop in Miami
- best cuban coffee in Miami
- best coffee shop in Orlando
- coffee shops open late in Orlando
- best coffee shop in Jacksonville
- specialty coffee roaster in Jacksonville
- best coffee shop in St. Petersburg FL
- best coffee shop to work from in Tampa
- best espresso in Fort Lauderdale
- local coffee roasters in Miami
- best brunch and coffee spot in Orlando
- dog friendly coffee shop in St. Petersburg FL
- coffee shop near me (Tampa)
- coffee shop near me (Miami)

**Skincare, broad (13):**
- best moisturizer for oily skin
- best sunscreen for humid weather
- best retinol serum for beginners
- best cleanser for acne-prone skin
- best vitamin c serum
- best skincare for hyperpigmentation
- best moisturizer for dry sensitive skin
- best mineral sunscreen for face
- best niacinamide serum
- best skincare routine products for rosacea
- best eye cream for dark circles
- best clean skincare brands
- best body lotion for eczema

**Skincare, local (12):**
- best facial spa in Miami
- best dermatologist for acne in Tampa
- best esthetician in Orlando
- best medical spa in Jacksonville
- where to buy korean skincare in Orlando
- best hydrafacial in Fort Lauderdale
- best skincare clinic in St. Petersburg FL
- chemical peel near Tampa
- best facial in Jacksonville
- local skincare boutique in Miami
- dermatologist near me (Orlando)
- facial spa near me (Tampa)

**`queries.py`**
- `load_queries(path) -> list[Query]`, where `Query` is a dataclass with `id, sector, query, query_type, city, location`. `location` is derived.
- Validation errors name the row number. The checks:
  - ids are unique and match `^[a-z0-9_]+$`;
  - sector and type come from the allowed sets;
  - a `local` query needs a city that is in `CITY_LOCATIONS`;
  - a `broad` query must have an empty city;
  - the query text is non-empty after stripping.
- Two cases only **warn**. Neither fails:
  - A `local` query whose text contains neither the city name nor "near me".
  - A sector count far from 25.
- Duplicate `q` text is allowed when the `(q, location)` pairs differ. It fails when both `q` and `location` are the same.
- CLI `check-queries` prints counts by sector, type and city.

### Step 3: `domains.py` (URL and domain normalisation)
- `unwrap_google_redirect(url)` handles `google.com/url?q=` and `url=`.
- `normalize_url(url)`:
  - lowercases the scheme and host;
  - drops the fragment, including `#:~:text=`;
  - drops tracking params (`utm_*`, `gclid`, `srsltid`);
  - strips a trailing `/`.
- `host(url)`:
  - lowercases the host;
  - strips the port;
  - strips a leading `www.`, `m.` or `amp.`;
  - converts IDN to punycode safely;
  - returns `""` for empty or invalid input.
- `root_domain(host)`:
  - takes the last two labels;
  - takes three labels for a small hard-coded set of multi-part suffixes (`co.uk`, `com.au`, …);
  - **keeps the full host for hosting platforms**: `square.site`, `myshopify.com`, `wixsite.com`, `godaddysites.com`, `business.site`, `wordpress.com`, `blogspot.com`, `squarespace.com`, `weebly.com`, `toasttab.com`, `sites.google.com`, `linktr.ee`.

  There is no `tldextract` dependency.
- Tests cover:
  - `www.`/`m.` stripping;
  - `shop.starbucks.com` → `starbucks.com`;
  - platform hosts kept whole;
  - redirect unwrapping;
  - fragment and UTM removal;
  - garbage input.

### Step 4: `entities.py` (name normalisation and alias matching)
`entity_norm(text) -> str` is applied the same way to rule aliases, override keys and AIO text:
1. Apply NFKD and drop combining marks. "Café Bustelo" becomes "cafe bustelo".
2. Map curly quotes to straight ones, then delete apostrophes. "Dunkin'" becomes "dunkin", and "Kiehl's" becomes "kiehls".
3. Replace `&` and `+` between words with ` and `.
4. `casefold()`.
5. Replace other non-alphanumeric characters with a space, then collapse the whitespace and strip it.
6. Strip trailing corporate suffixes: `inc`, `llc`, `co`, `corp`, `ltd`. A leading "the" is not stripped.

`entity_norm_cased(text)` applies steps 1–3 and 5 **without** casefold. It is used only for ambiguous aliases.

**Ambiguity modes** (new, from the rule file's `ambiguous` column):

| Mode | Matching | Seeded for |
|---|---|---|
| `no` (default) | `\b<alias>\b` on `entity_norm` text | most brands (CeraVe, Starbucks, …) |
| `case` | `\b<alias>\b` on `entity_norm_cased` text, case-sensitive (`The Ordinary`, `Ideal Image`, `Death Wish`, `Clean and Clear`, `Pilon`, `Caribou`, `Black Rifle`) | multi-word or capitalised names that are also common phrases |
| `case_mid` | as `case`, but **not** at the start of a sentence, a list item, a heading or a table cell | single common words: `Target`, `Heyday` |
| `candidate` | case-insensitive match, but it produces a `list_title`-style **candidate** only, which never self-confirms | `illy`. The unambiguous aliases `illy coffee` and `illy caffe` stay `no`. |

`First Aid Beauty` is stored only as the full name, never as "first aid". The implementer looks over the seed list for other common-word aliases and gives each one a mode.

Text segmentation: `aio_text` (step 9) returns a list of `(segment_text, segment_kind)` items, where `segment_kind` is `sentence`, `list_item`, `heading` or `cell`. That makes "start of segment" well defined for `case_mid`. Sentences are split on `[.!?]\s+`.

`find_aliases(segments, alias_index) -> dict[entity, set[method]]`:
- Aliases are compiled once, sorted longest first. Matched spans are consumed, so "la roche posay" wins over "roche".
- The method is `dictionary` for modes `no`, `case` and `case_mid`, and `candidate` for mode `candidate`.

Tests cover:
- each normalisation rule;
- `entity_norm("Dunkin’") == entity_norm("DUNKIN")`;
- "target dark spots" gives no match, and so does "Target dark spots with vitamin C." at the start of a sentence;
- "available at Target and CVS" matches Target;
- "an ordinary moisturizer" gives no match, while "The Ordinary Niacinamide" matches;
- "silly" gives no `illy` match, and a lone "illy" gives a candidate only;
- longest match first.

### Step 5: `serpapi_client.py` (requests and redaction)
- `build_search_params(query)`:
  - `engine="google"`, `q`, `location`, `gl`, `hl`, `google_domain`, `device`, `no_cache="true"`;
  - **no `num`** **[VERIFY]**.
- `build_aio_params(page_token)` returns `{"engine": "google_ai_overview", "page_token": token}` **[VERIFY]**.
- `get_json_once(url, params, api_key, session=None, timeout=60) -> Response(status, body, error_kind, retryable)` makes **one attempt only**. Retries move to `collect.py`, so every attempt passes through the budget.
  - It adds `api_key` inside the function only.
  - It catches every `requests` exception and returns `error_kind="transport", retryable=True` with a redacted message.
  - `retryable` is True for 5xx, transport errors and 429 throughput errors.
  - A 429 whose body says the plan or searches are exhausted **[VERIFY wording]** gets `error_kind="quota"`. It is not retryable and aborts the run (exit 2).
  - A JSON `"error"` body is returned as data, for the caller to classify.
- `AuthError` is raised on 401/403 and aborts the collection (exit 3).
- `redact(text, key)` replaces the key with `***`. `scrub(obj, key)` recursively drops `api_key`, `account_email` and `email`, and redacts any string that contains the key.
- `fetch_account(api_key)` returns only whitelisted numeric fields **[VERIFY]**. The raw response is never cached or logged.

### Step 6: `cache.py`
- The path is `raw/<run>/<query_id>.<kind>.json`, where `kind` is `search` or `ai_overview`.
- Each file is a wrapper:
  ```json
  {"_meta": {"run", "query_id", "kind", "query", "location", "params_without_key",
             "collected_at_utc", "http_status", "serpapi_search_id", "attempts",
             "outcome", "error_kind", "error_message_redacted", "supersedes"},
   "response": {...scrubbed... or {} on failure}}
  ```
- `outcome` is `ok`, `no_results` or `error`. `error_kind` is `transport`, `http_4xx`, `http_5xx`, `api_error`, `empty`, `quota`, `budget_exhausted` or null.
- `write_cached` writes atomically (a temp file in the same directory, then `os.replace`) after `scrub()`. It refuses to write if the key string still appears in the serialised text.

**Caching rules:**

| Request | Final result (after any budget-allowed retries) | Cached? | Why |
|---|---|---|---|
| main search | ok, or "no results" | yes (`ok` / `no_results`) | A real observation. |
| main search | transport / 5xx / 429 throughput / other `error` body | **no**, ledger lines only | Can be retried later for the same query and run. |
| main search | retry wanted but budget exhausted | **no** | Same. The loop stops. |
| main search | 401/403, quota 429 | no; abort | Key or account problem. |
| AIO follow-up | ok with content | yes (`ok`) | |
| AIO follow-up | **any** failure, including `budget_exhausted` on a retry | **yes**, `outcome=error` | The token is short-lived, so a later retry is pointless. Caching keeps "never re-request" true. 401/403 or quota also abort after the file is written. |

- **Never re-request:** collection skips any `(run, query_id, kind)` that already has a file. The only exception is `--retry-failed-aio` (step 8).
- If the cached `_meta.query` or `location` differs from the current query file, collection refuses and tells the user to bump the id.
- `supersede(path)` moves a file to `raw/<run>/_superseded/…`. Files are never deleted.
- `CollectLock` uses `O_CREAT|O_EXCL` and writes the PID and time. It is removed in `finally`. A second `collect` refuses and explains how to remove a stale lock.

### Step 7: `budget.py` (per-attempt accounting)
**Ledger grain:** one line per HTTP attempt, including transport failures that got no response. Fields: `{ts, run, query_id, kind, request_key, attempt, http_status, outcome, error_kind, reason, charged_assumed}`.
- `request_key` is `<run>/<query_id>/<kind>/<seq>` and groups the attempts of one logical request.
- `attempt` counts from 1.
- `reason` is `normal`, `retry` or `retry_failed_aio`.
- `charged_assumed` is an annotation taken from the verified billing facts. **The guard ignores it and counts every ledger line as spent.**

**`ledger_total(ledger)`** is the number of ledger lines over the whole study, all runs including `smoke`.

**`estimate(plan, followup_rate, retry_headroom)`** returns `Estimate(main, followup_max, retry_headroom, retry_main, retry_followup, total)`:
- `main` is the number of items with action `search` or `retry_search`;
- `followup_max = ceil(main × followup_rate)`;
- `total = main + followup_max + retry_headroom`.

The headroom is only a planning allowance. The hard limit is enforced at runtime, as described below.

**`check_budget(est, cap, ledger_total, account)`** refuses (exit 2) in three cases:
- `ledger_total + est.total > cap`;
- `est.total > account.total_searches_left`;
- account data is unavailable in a non-dry run.

**`RuntimeBudget(allowed)`**, where `allowed = min(cap − ledger_total, total_searches_left, max_new or ∞)`:
- `can_spend(n_needed)` and `charge()` are called **once per attempt, before it is sent**.
- A main-search attempt (first try or retry) needs `remaining ≥ 2`: the attempt itself plus one reserved first attempt of the possible follow-up. This is how the reserve keeps working alongside retries.
- A first follow-up attempt needs `remaining ≥ 1`, which the reserve guarantees.
- A follow-up retry needs `remaining ≥ 1`. If that isn't available, the follow-up is cached as failed with `error_kind=budget_exhausted`.
- If a main search can't get its 2 searches (first attempt or retry), the item is not cached and the loop stops with "budget exhausted, N items left".
- **Invariant:** `charged ≤ allowed`, so lines in the ledger never exceed the study cap or the account's remaining searches. A property test with random scripted failures checks it.

**After a run**, the report of searches used shows:
- logical requests and attempts, by kind;
- retries made;
- the Account API `this_month_usage` delta. This is the ground truth, with a note that it may include other use of the same account.

### Step 8: `collect.py`
**`build_plan(queries, run, *, sector=None, query_ids=None, limit=None, retry_failed_aio=False)`** returns a list of `PlanItem(run, query, main_cached, aio_cached, aio_state, action)`, where `action` is `skip`, `search` or `retry_search`.

**`--retry-failed-aio`:**
- It is the only path that re-requests a cached main search, and it needs an explicit `--query-ids` list.
- It applies only to items whose AIO state is `followup_failed` (including `stale_token` and `budget_exhausted`).
- The plan shows these items as `RETRY (re-requests cached search)`, and the confirmation prompt names them.
- Old files are moved with `supersede()`, and `_meta.supersedes` records the old `collected_at`.
- Ledger rows get `reason=retry_failed_aio`.
- Analysis uses the newer files. The report lists superseded items with their old and new times.

**`run_collection(plan, api_key, *, dry_run, cap, max_new, followup_rate, retries, retry_headroom, sleep_s, yes, no_account, session=None)`:**
1. Print the plan table.
2. Call the Account API, which is free. Dry-run calls it too, unless `--no-account` is set or there is no key, in which case remaining searches show as "unknown".
3. Build the estimate and run `check_budget`. Print `ledger_total (study so far)`, `study cap`, `left under cap`, `account searches left`, `max-new`, and the estimate broken into main, follow-up and retry headroom. **Dry-run stops here** and prints "would spend ≤ N searches (study total after: ≤ M of cap C)".
4. Unless `--yes`, prompt `About to spend up to N searches (study so far L, cap C, account left R). Continue? [y/N]`. On a non-TTY without `--yes`, refuse.
5. Hold `CollectLock`. For each item, `request_with_retries(kind, params)` loops up to `1 + retries` attempts:
   - check `RuntimeBudget` before each attempt;
   - write one ledger line per attempt;
   - retry only when the response is `retryable`, with a backoff of 2 s and then 5 s;
   - after the main search finishes, cache it according to step 6;
   - if the response has only a `page_token`, make the follow-up **immediately** with the same retry loop, then cache it according to step 6;
   - sleep `sleep_s` between live requests.
6. On `KeyboardInterrupt`, stop cleanly and print the summary. If an item is interrupted after its main search, its token becomes `stale_token`.
7. Print a summary: logical requests and attempts by kind, retries, AIO states, failures and the Account API delta.

**Run spacing:** run2 is refused if it starts less than `--min-hours-between-runs` (default 12) after the newest run1 `collected_at`, unless `--force-spacing` is given.

### Step 9: `extract.py`
All functions are pure and take wrapper dicts.

**`query_status_row`** has the columns `run, query_id, sector, query_type, city, location, collected_at, superseded, search_outcome, aio_status, aio_detail, n_organic, n_organic_domains, has_local_pack, n_local_pack, n_aio_refs`. One mapping is used everywhere:

| Situation | `aio_status` | `aio_detail` | In AIO rate | In AIO content metrics |
|---|---|---|---|---|
| no search file, ledger shows failure | `search_failed` | error_kind | excluded | excluded |
| no search file, no ledger row | `not_collected` | | excluded | excluded |
| search ok, no `ai_overview` | `absent` | | denominator | n/a |
| `ai_overview.text_blocks` inline | `present_inline` | | shown | yes |
| page_token, follow-up file `ok` | `present_followup` | | shown | yes |
| page_token, follow-up file `error` | `followup_failed` | error_kind (incl. `budget_exhausted`) | shown | excluded |
| page_token, no follow-up file | `followup_failed` | `stale_token` | shown | excluded |
| `ai_overview.error` in main response | `error` | message | denominator, not shown | excluded |
| search `no_results` | `absent` | `no_results` | denominator | n/a |

**`organic_rows`** reads `organic_results` **[VERIFY fields]**.
- It skips entries without a `link`, sorts by `position`, keeps the first 10 and renumbers `rank` from 1.
- Columns: `run, query_id, rank, url, url_norm, host, root_domain, title, source`.

**`aio_reference_rows`** reads `ai_overview.references` **[VERIFY]**. Columns: `run, query_id, ref_index, url, url_norm, host, root_domain, source, title, cited_in_blocks`.

**`aio_text`** recursively walks `text_blocks` (paragraph, heading, list, nested list, table, comparison, expandable) **[VERIFY types]**.
- It returns `(segment_text, segment_kind)` segments for `entities.find_aliases`.
- For unknown shapes, it falls back to strings under `snippet`, `title` or `text`.

**`aio_mention_rows(ai_overview, alias_index, local_pack_names)`** uses four methods:
- `dictionary`: aliases in modes `no`, `case` and `case_mid`.
- `candidate`: aliases in mode `candidate`.
- `list_title`: `list[].title` and `snippet_highlighted_words` strings, trimmed of a trailing `:` or `-`, 2–60 characters long, with a stoplist applied.
- `local_pack_name` (new): a `list_title` candidate whose `entity_norm` equals, or contains as a whole-word match, the `entity_norm` of a local-pack place title for the **same run and query**. This is evidence that it is a real business.

Rows are deduplicated per `(run, query_id, entity_norm)`, with `methods` as a `|`-joined set. Columns: `run, query_id, entity, entity_norm, methods, block_types, local_pack_match_domain`.

**`local_pack_rows`** reads `local_results.places` **[VERIFY shape]**.
- Columns: `run, query_id, position, title, entity_norm, website, root_domain, rating, reviews, type`.
- It is used for the descriptive report section, for `local_pack_name` confirmation and for review suggestions. **It is not in the headline metrics.**

**Robustness:** every accessor uses `.get` with defaults, and wrong types are ignored. One fixture has missing or odd fields.

### Step 10: `labeling.py` and `data/brand_rules_v1.csv`
**Category definitions** (shared by the README and report):
- `big_brand`: national or global chains and brands. It includes national service chains (subtype `national_chain`).
- `small_local`: small or independent businesses.
  - **For local queries**, these are independent local businesses, including Florida-only or regional chains (subtype `regional_chain`, which an override can reverse).
  - **For broad product queries**, these are small or independent brands, because broad queries have no local businesses.
  - Results are always split by `query_type`.
- `marketplace_retailer`: Amazon, Walmart, Sephora, Ulta, delivery apps and similar.
- `directory_media_other`: directories, review sites, social media, publishers, and government or education sites.
- `not_a_brand`: ingredients, features and attributes. These are excluded from headline numerators and denominators, with the count reported.
- `unknown`: not yet labeled. It is never silently treated as small, and it is always shown next to big and small (step 11).

**Operational big-vs-small test** (new; in the README, the report and the rule file's `basis` column). A brand or business is `big_brand` if **any** of these holds:
- **(a) ownership:** it is majority-owned by a publicly traded company or a multinational group. Examples: Blue Bottle (Nestlé), Stumptown (JDE Peet's), Tatcha, Drunk Elephant, Café Bustelo.
- **(b) distribution:** it is sold nationally through major US retailers (Walmart, Target, Costco, Ulta, Sephora, CVS, Walgreens, or national grocery chains).
- **(c) footprint:** it is a service or store chain with locations in several US states, or one that franchises nationally.

It is `small_local` only when none of these holds. If the evidence is unclear, it stays `unknown` with a note. The implementer **checks ownership and distribution facts before seeding a rule** and records them in `notes`, with the source where practical. Rows the implementer can't verify are left out of the seed rather than guessed, so they appear in `labels_to_review.csv`.

**Rule file columns:** `match_type (domain|name), pattern, entity, category, sector (coffee|skincare|any), subtype, basis (a|b|c|marketplace|directory|ingredient|attribute), ambiguous (no|case|case_mid|candidate), notes`.

**Seed contents** (about 130 rows, curated; each `big_brand` row has its `basis` filled in):

`big_brand`
- **Coffee:** Starbucks, Dunkin', Dutch Bros, Tim Hortons, Peet's, Caribou (`case`), McDonald's/McCafé, Panera, Krispy Kreme, Nespresso, Keurig, Folgers, Maxwell House, Lavazza, illy coffee/illy caffe (and `illy` as `candidate`), Death Wish (`case`), Black Rifle (`case`), Café Bustelo, Pilon (`case`), Stumptown, Blue Bottle.
- **Skincare:** CeraVe, La Roche-Posay, Neutrogena, Cetaphil, Olay, Aveeno, Eucerin, Clinique, Estée Lauder, Kiehl's, The Ordinary (`case`) and Deciem, Paula's Choice, Drunk Elephant, Tatcha, First Aid Beauty (full name only), Glossier, Dermalogica, Murad, EltaMD, Supergoop!, Vanicream, Differin, Clean & Clear (`case`), Biore.
- **National service chains:** European Wax Center, Hand & Stone, Massage Envy, Heyday (`case_mid`), Ideal Image (`case`), LaserAway.

`marketplace_retailer`
- amazon, walmart, target (domain `target.com`; name `Target` is `case_mid`), sephora, ulta, cvs, walgreens, dermstore, nordstrom, macys, costco, iherb, yesstyle, sokoglam, credobeauty, ubereats, doordash, grubhub, instacart.

`directory_media_other`
- **Directories and social:** yelp, tripadvisor, reddit, facebook, instagram, tiktok, youtube, foursquare, `maps.google.com`, opentable, groupon, quora, wikipedia.
- **Media:** timeout, eater, thrillist, tampabay.com, miaminewtimes, orlandoweekly, cltampa, jacksonville.com, healthline, allure, byrdie, vogue, nytimes, cosmopolitan, verywellhealth, aad.org.

`not_a_brand` (names)
- hyaluronic acid, niacinamide, retinol, vitamin c, salicylic acid, ceramides, zinc oxide, spf, oil-free, fragrance-free, non-comedogenic, ambiance, price, location, wifi, outdoor seating, cold brew, espresso, latte.

**Overrides (`label_overrides.csv`)** have the columns `key_type (domain|entity), key, category, note`.
- Domain keys are normalised with `root_domain(host(key))`, and entity keys with `entity_norm`.
- Validation names the line number for an invalid category, an invalid `key_type` or an empty key.
- If a key appears twice, the last row wins and the loader warns.

**`label(root_domain=None, entity_norm=None, sector=None, ...) -> (category, entity, label_source)`** applies this precedence:
1. Override.
2. Exact domain rule.
3. Parent-domain rule.
4. Name rule.
5. The `platform_hosted` heuristic, which labels the domain `small_local`.
6. The `tld` heuristic (`.gov`, `.edu`), which labels it `directory_media_other`.
7. Otherwise `unknown`.

Sector-specific rules apply only within their own sector.

**Mention confirmation:** `confirmed = True` if any one of these holds:
- `methods` includes `dictionary` or `local_pack_name`;
- the entity is matched by an override or a name rule. This is how a `candidate` or `list_title` entry gets confirmed.

Category for a confirmed `local_pack_name` mention:
- first, the label of its name;
- otherwise, the label of the matched place's website domain;
- otherwise `unknown`. It is confirmed as a business but not yet classified, so it counts toward the unresolved share.

Unconfirmed candidates stay in `aio_mentions.csv` with `confirmed=False, category=unknown`. They are left out of the point estimate but **are included in the bounds** (step 11).

**Domains** (organic results, AIO references and local-pack websites) need no confirmation. They can still be labeled `not_a_brand` by override.

**`build_review_table`** writes `labels_to_review.csv`.
- Columns: `key_type, key (normalised), example_raw, count, n_queries, sectors, surfaces, example_url, example_title, suggested_category, suggest_reason, category (empty)`.
- Suggestions only, never applied automatically:
  - an unknown domain that is the website of a local-pack place → `small_local` (`local_pack_website`);
  - a `local_pack_name`-confirmed unknown mention → `small_local` (`local_pack_name`);
  - a `list_title` candidate that matches the `not_a_brand` stoplist pattern → `not_a_brand`.
- Rows are sorted by frequency, and suggested rows come first within each frequency tier.

### Step 11: `analyze.py`
- Inputs are the labeled tidy frames.
- Only `--runs run1,run2` is analysed by default.
- Every table has `run, sector, query_type` columns plus an `all` roll-up, and shows the `n_queries` actually used.

**Unit and denominator**
- The headline unit is the distinct root domain per query (for organic results and AIO references), or the distinct confirmed `entity_norm` per query (for mentions). `not_a_brand` is excluded.
- **`domain_share` (point estimate):**
  - **Numerator:** for each query, the distinct eligible items in the category.
  - **Denominator:** the distinct eligible items in that query. Eligible means not `not_a_brand`, with `unknown` included in the denominator.
  - The per-query shares are then averaged over queries (a macro average).
  - Example: two Yelp URLs plus 8 other distinct domains give Yelp a share of 1/9.
- **Zero denominator** (new): if a query has no eligible items on a surface (for example every domain is `not_a_brand`, or the AIO has no references), its share is NaN. It is **dropped** from the macro average and counted in `n_queries_empty`. `pct_queries_present` still includes it in the denominator as "not present", since the surface existed. The same rule applies to `top3_share`. Tests pin this.
- **`result_share` (secondary):** the category's share of raw result slots, `/ n_organic`. In the example, Yelp gets 2/10.

**Bounds, the unresolved share and the provisional flag** (new, from issue 3). For each surface, segment and category in `{big_brand, small_local}`:
- `unresolved` items per query are the `unknown` items, plus the **unconfirmed candidates** on the mention surface.
- `domain_share`: the point estimate as defined above. Unconfirmed candidates are not in it.
- `bound_low`: category items ÷ (eligible + unconfirmed). Unresolved items count as "not this category".
- `bound_high`: (category items + unresolved) ÷ (eligible + unconfirmed). Unresolved items count as "this category".
- `unresolved_share`: the macro average of unresolved ÷ (eligible + unconfirmed).
- `provisional = unresolved_share > provisional_threshold` (default 0.10, flag `--provisional-threshold`).

`bound_high` for mentions is deliberately loose, because unconfirmed candidates can include non-brands. The report says so. Its purpose is to show how far the small-business share *could* go before review.

**Metrics**
1. **Organic visibility**, per segment × category:
   - `domain_share` with `bound_low`/`bound_high` (for big and small);
   - `pooled_domain_share`;
   - `pct_queries_present`, with a Wilson 95% CI computed in about 10 lines (no scipy);
   - `top3_share`;
   - `pct_rank1`;
   - `mean_best_rank` for small_local and big_brand;
   - `result_share` (secondary).
2. **AIO visibility:** the same metrics over AIO reference domains and over confirmed AIO mentions. Only queries with `aio_status` `present_*` count.
3. **AIO rate:** `(present_* + followup_failed) / (present_* + followup_failed + absent + error)`. All counts are reported.
4. **Overlap** (`overlap_by_query.csv`), per run × query with AIO content:
   - `n_aio_domains`, `n_organic_domains`, `n_shared`;
   - `aio_in_organic = n_shared / n_aio_domains`;
   - `jaccard`;
   - `url_level_shared`;
   - `shared_small_local`, `shared_big`.

   If there are zero AIO references, these are NaN and the query is counted.
5. **Stability** (`stability_by_query.csv`), when both runs exist:
   - Jaccard of organic domains, AIO reference domains and confirmed mention entities;
   - `aio_presence_agreement`.

   Two empty sets give NaN. If only one run exists, the file is written with a note.
6. **Headline:** `big_to_small_ratio` per surface, computed from the point estimates. When the small share is 0, it is reported as "∞ (provisional)" rather than as a number. It always sits next to the bounds.
7. **Local pack** (descriptive, local queries only):
   - local pack rate;
   - mean number of places;
   - category shares of the places, with unknown shown.

**Output:** `summary_by_segment.csv` in long format, with columns `run, sector, query_type, surface, category, metric, value, bound_low, bound_high, unresolved_share, provisional, n_queries, n_queries_empty, ci_low, ci_high`.

### Step 12: `report.py`, `leaks.py` and `cli.py`
**`md_table(df, floatfmt=".2f")`** is our own pipe-table formatter.

**`report.md` sections:**
0. **A provisional banner** (new), shown when any headline row is provisional: "**PROVISIONAL: X% of organic domains and Y% of AIO mentions are unlabeled or unconfirmed. Small-business shares are lower bounds until `labels_to_review.csv` is reviewed and `analyze` is re-run.**"
1. Title, collection date range, location settings, query file version, runs and counts, and searches used (ledger attempts, retries and the Account delta). Superseded items and the field-name verification status.
2. **Observational disclaimer** in bold. It names:
   - one geography and specific point(s) in time;
   - SerpApi's simulated location and device;
   - no login or personalisation;
   - that AIO is non-deterministic;
   - that labels come from rules plus manual overrides;
   - small cells, so the CIs are wide.
3. Category definitions and the operational big-vs-small test.
4. AIO rate by segment.
5. **Main big-vs-small table**, one per surface (organic, AIO references, AIO mentions). Columns: `segment | n | big (low–high) | small_local (low–high) | marketplace | directory/media | unknown+unconfirmed | provisional`. One line gives `result_share`. It also shows the count of `not_a_brand` exclusions and of unconfirmed candidates.
6. Overlap.
7. Stability, or "run2 not collected yet".
8. Local pack (descriptive).
9. Labeling coverage: label sources, the override count, and how many review suggestions are still open.
10. **Limitations:**
    - domain-level labels undercount small businesses, which appear mostly inside directories and listicles and in the local pack;
    - small-business AIO mentions depend on candidates being confirmed, so before review the small share is biased down;
    - the mention upper bound is loose;
    - social profiles are labeled as directories;
    - dictionary-based detection.
11. Method notes: the local pack is excluded from the headline, how stale or failed AIOs are handled, the retry policy, and the ambiguity modes.

**`leaks.py`** (new):
- `scan_for_key(paths, key) -> list[Path]` reads files as bytes. It looks for the raw key and its URL-encoded form.
- It refuses (`ValueError`, then exit 3) if the key is empty or shorter than 16 characters, so a short or empty string can't match every file.
- It never returns or prints matched content.

**CLI** (`python -m brand_study <cmd>`). `cli.main(argv=None, load_env=True)` calls `load_dotenv(PROJECT_ROOT / ".env")` (no override) only when `load_env` is true. `__main__` calls `main()`. Commands:
- `check-queries [--queries PATH]`
- `account`: free; prints whitelisted fields.
- `collect --run RUN [--dry-run] [--limit N] [--sector coffee|skincare] [--query-ids a,b] [--study-cap 300] [--max-new N] [--followup-rate 1.0] [--retries 1] [--retry-headroom 10] [--sleep 1.0] [--yes] [--no-account] [--retry-failed-aio] [--min-hours-between-runs 12] [--force-spacing]`
  - Help text for `--study-cap`: "Total SerpApi searches allowed across the WHOLE study (all runs, including smoke and earlier invocations, counted from ledger.jsonl). Not per invocation."
  - Help text for `--max-new`: "Optional hard limit on searches in THIS invocation."
- `analyze [--runs run1,run2] [--overrides PATH] [--provisional-threshold 0.10]`: offline, 0 searches.
- `scan-leaks [--path DIR ...]`: offline, 0 searches. It defaults to `outputs/brand_study/` and prints `clean (N files scanned)` or `LEAK in K files:` followed by the relative paths only.

**Exit codes:**
- 0: ok.
- 2: budget, quota or confirmation refusal.
- 3: config, validation, auth or invalid-key error.
- 4: leaks found.

No output ever prints a params dict that contains the key.

### Step 13: `ollama_lifecycle.py` (for Part 2)
Interface:
```python
class OllamaAPI:
    def __init__(self, base_url: str | None = None, session=None): ...
    base_url: str
    def version(self) -> str                   # GET /api/version
    def list_tags(self) -> set[str]            # GET /api/tags -> names (accepts "name" or "model")
    def pull(self, tag: str) -> None           # POST /api/pull {"model", "name", "stream": true}
    def unload(self, tag: str) -> None         # POST /api/generate {"model": tag, "keep_alive": 0}
    def delete(self, tag: str) -> None         # DELETE /api/delete {"model", "name"}  (== `ollama rm`)

@dataclass
class GpuInfo: index: int; name: str; total_mb: int; used_mb: int; free_mb: int

def query_gpus(run=subprocess.run) -> list[GpuInfo] | None
def free_disk_gb(path) -> float
def disk_check_paths(explicit=None) -> list[Path]
def cli_host_from_base_url(base_url) -> str | None           # NEW
def cli_rm(tag, base_url, run=subprocess.run, which=shutil.which) -> bool   # CHANGED

@contextmanager
def ollama_model(tag, *, api=None, expected_size_gb=25.0, min_free_margin_gb=10.0,
                 required_gpu_gb=24.0, strict_gpu=False, keep=False,
                 models_path=None, allow_unknown_disk=False,
                 gpu_query=query_gpus, disk_free=free_disk_gb, rm_fallback=cli_rm, log=print):
    ...
    yield tag
```

**API details** **[VERIFY against the installed Ollama]**:
- Log `version()` first. If the server is unreachable, fail clearly before doing anything else.
- `pull` and `delete` send both `model` and `name`.
- `pull` streams NDJSON. **Any `"error"` line raises `OllamaPullError`, even with HTTP 200.** The final status must be `success`. Progress is printed roughly every 5%.
- `delete` treats 404 as "not found" and any other non-2xx as a failure.

**Targeting the CLI fallback at the right server** (new, from issue 2):
- `cli_host_from_base_url(base_url)` parses the URL. It requires a scheme of `http` or `https`, a host, an optional port, and an empty path or `/`. It returns `"<scheme>://<host>:<port>"`, using port 11434 when none is given. It returns `None` for anything else, such as a path prefix behind a proxy, credentials in the URL, or a parse failure.
- `cli_rm(tag, base_url)`:
  - if the host is `None` or `ollama` is not on PATH, it logs "CLI fallback disabled: …" and returns False;
  - otherwise it runs `ollama rm <tag>` with `env = {**os.environ, "OLLAMA_HOST": host}`, so the **inherited `OLLAMA_HOST` is always overridden** and the CLI talks to the same server as the HTTP API;
  - it logs the host it used.
- Before the fallback, the helper calls `api.list_tags()` (best effort). If that works and the tag is absent, the delete has already taken effect, so the fallback is skipped. If it fails, the fallback proceeds against the derived host.
- The manual-command message reads: `could not delete <tag>; run manually: OLLAMA_HOST=<host or base_url> ollama rm <tag>`.

**Choosing the disk-check paths:**
1. Use the explicit `models_path` argument or `BRAND_STUDY_OLLAMA_MODELS_DIR` if set. That path alone is checked.
2. Otherwise, take every existing candidate: client-side `OLLAMA_MODELS`, `~/.ollama/models` and `/usr/share/ollama/.ollama/models`. Use the nearest existing parent of each one, and take the **minimum** free space across them. Log each path with its free GB.
3. If the base URL's host is not local, warn loudly that the local disk and GPU checks do not describe the server.
4. If no path can be checked, raise `DiskCheckUnavailable` unless `allow_unknown_disk=True`.

**Behaviour:**
1. Normalise the tag: with no colon it becomes `:latest`. Refuse placeholder tags.
2. Check the server version and reachability.
3. `present_before = tag in api.list_tags()`. If true, log that the model is already present and will NOT be deleted, skip the pull, and yield.
4. Disk check: if `min_free − expected_size_gb < min_free_margin_gb`, raise `InsufficientDiskError` without pulling.
5. GPU report:
   - log each GPU's free/total;
   - if `max(free) < required_gpu_gb`, print a loud WARNING: the shared GPU may be held by another job (about 20 GB), so the model will partly offload to CPU and run slowly;
   - `strict_gpu=True` raises instead;
   - if `nvidia-smi` is missing, warn "GPU memory unknown";
   - if the server is remote, note that the report is for this machine.
6. In `try`: set `pull_started = True`, run `api.pull(tag)`, then `yield tag`.
7. In `finally`, if `pull_started and not present_before and not keep`:
   - call `api.unload` (best effort), then `api.delete(tag)`;
   - on failure, do the `list_tags` pre-check, then `rm_fallback(tag, api.base_url)`;
   - if both fail, log the manual command shown above;
   - a 404 after an interrupted pull counts as fine, with a warning that partial blobs may remain;
   - never `raise` from `finally`;
   - log free disk after.
8. Signals:
   - `KeyboardInterrupt` passes through `finally` without extra handling.
   - SIGTERM (main thread only) gets a temporary handler that raises `SystemExit(143)`. The previous handler is restored afterwards.
9. Documented limits:
   - if another user pulls the same tag during our run, we delete it, because it was absent when we checked;
   - we cannot stop other processes using the model while it is deleted;
   - on a remote server, the checks only describe this machine.

### Step 14: `.gitignore`, `requirements-dev.txt` and README
- The new `.gitignore` contains `outputs/brand_study/` and `.pytest_cache/`.
- `requirements-dev.txt` contains `pytest>=7`.
- README section "Brand Visibility Study (Part 1)":
  - design;
  - category definitions and the operational big-vs-small test;
  - setup;
  - steps:
    1. `check-queries`
    2. `account`
    3. `collect --run run1 --dry-run`
    4. `collect --run run1`
    5. after ≥12–24 h, `collect --run run2`
    6. `analyze`. The first pass is a **labeling pass**: expect a PROVISIONAL banner.
    7. fill in `label_overrides.csv` from `labels_to_review.csv` (suggestions, `not_a_brand`, local-pack candidates)
    8. re-run `analyze` until the banner clears or the bounds are tight enough
    9. `scan-leaks`
  - **cost table:**

    | What | SerpApi searches |
    |---|---|
    | Account API, `--dry-run`, `analyze`, `check-queries`, `scan-leaks` | 0 |
    | Each run | 50 main + ≤50 follow-ups; guard plans for this + 10 retry headroom (≤110) |
    | Full study (2 runs) | ≤ 200 normal worst case, ≤ 220 with retry headroom, against a study-wide cap of 300 |
    | `--retry-failed-aio` | ≤ 2 attempts per main + ≤ 2 per follow-up per listed query, within the same cap |
    | Smoke test | ≤ 4 (enforced with `--max-new 4`) |
    | Hard guarantee | ledger attempts never exceed `--study-cap` or the account's remaining searches |

  - `--study-cap` is study-wide and counts the smoke run;
  - cache, ledger (one line per attempt) and "never re-request" semantics, with `--retry-failed-aio` as the only exception;
  - label precedence and the ambiguity modes;
  - limitations, matching the report;
  - the Ollama helper's safety rules, including `BRAND_STUDY_OLLAMA_MODELS_DIR` on a shared server and the `OLLAMA_HOST` handling of the fallback;
  - Part 2 model tags.

---

## 3. Data and interfaces summary

| Artifact | Key / grain | Main columns |
|---|---|---|
| `queries_v1.csv` | `id` | sector, query, query_type, city, notes |
| `brand_rules_v1.csv` | (match_type, pattern) | entity, category, sector, subtype, basis, ambiguous, notes |
| `label_overrides.csv` | (key_type, normalised key) | category, note |
| raw cache | (run, query_id, kind) | `_meta` (incl. attempts) + scrubbed `response` |
| `ledger.jsonl` | **one HTTP attempt** | ts, run, query_id, kind, request_key, attempt, http_status, outcome, error_kind, reason |
| `query_status.csv` | (run, query_id) | aio_status, aio_detail, n_organic, n_organic_domains, has_local_pack, superseded |
| `organic_results.csv` | (run, query_id, rank) | url_norm, root_domain, entity, category, label_source |
| `aio_references.csv` | (run, query_id, ref_index) | root_domain, source, category, label_source |
| `aio_mentions.csv` | (run, query_id, entity_norm) | entity, methods, block_types, local_pack_match_domain, confirmed, category, label_source |
| `local_pack.csv` | (run, query_id, position) | title, entity_norm, website, root_domain, category |
| `summary_by_segment.csv` | (run, sector, query_type, surface, category, metric) | value, bound_low, bound_high, unresolved_share, provisional, n_queries, n_queries_empty, ci_low, ci_high |
| `overlap_by_query.csv` | (run, query_id) | n_shared, aio_in_organic, jaccard, url_level_shared |
| `stability_by_query.csv` | query_id | organic_jaccard, aio_ref_jaccard, aio_mention_jaccard, aio_presence_agreement |
| `labels_to_review.csv` | (key_type, key) | example_raw, count, n_queries, surfaces, suggested_category, suggest_reason, category (empty) |

---

## 4. Edge cases (each with a test)

**Query file**
- Duplicate ids.
- A local query without a city.
- A city missing from the location map.
- A local query whose text has no city: it warns and does not fail.
- The same `q` with different locations is allowed.
- The same `q` with the same location is rejected.
- Changed query text with an existing cache is refused.

**SerpApi responses**
- No `ai_overview`.
- Only `page_token`.
- `ai_overview.error`.
- Follow-up failures (error body, empty, transport error, 5xx after retries, `budget_exhausted`), all cached.
- A stale token.
- A "no results" search.
- 401: the run aborts and the key is redacted.
- Quota 429: the run aborts with exit 2.
- A throughput 429 or 5xx on the main search: retried within budget, and not cached if it still fails.
- Fewer than 10 organic results.
- A missing `link`.
- Duplicate domains.
- Redirect URLs.
- Platform-hosted sites.
- Social profiles.
- A local pack with and without `website`.

**Budget and retries** (new)
- A 5xx on a main search when `remaining == 2`: the first attempt is charged, there is no budget for a retry, the item is not cached and the loop stops. The ledger has 1 line.
- A 5xx on a follow-up when `remaining == 1` after the first follow-up attempt: it is cached as failed with `budget_exhausted`.
- A retry that succeeds: there are 2 ledger lines with the same `request_key`.
- A property test over random failure scripts asserts that the number of ledger lines is at most `allowed`.
- `--max-new` stops the loop.
- The estimate goes over the cap once the earlier ledger total is included.
- The estimate goes over the remaining searches.
- The Account API is unavailable: a dry run proceeds and a real run refuses.
- The reserve-2 stop.
- Lock.
- Ctrl+C.
- run2 too soon.
- `--retry-failed-aio` without `--query-ids` (refused), and with them.
- Dry-run output shows the ledger total and the cap.

**Mentions and labels**
- An ingredient heading that is not confirmed.
- An override that confirms a candidate.
- `not_a_brand` exclusion.
- `entity_norm` matching across apostrophes, accents and `&`.
- Ambiguity modes: "target dark spots", "Target dark spots…" at the start of a sentence, "sold at Target", "an ordinary moisturizer", "The Ordinary Niacinamide", "silly", and a lone "illy" (candidate).
- A `local_pack_name` confirmation with an unknown category (counted in the unresolved share).
- Review suggestions for local-pack websites.
- Invalid or duplicate overrides.
- All labels unknown.

**Key secrecy**
- The fake key never appears in cache files, the ledger, stdout or stderr, exception messages or `account` output.
- `scan-leaks` finds a planted key, including its URL-encoded form.
- `scan-leaks` refuses an empty or short key.
- `scan-leaks` prints names only (checked with `capsys`, so the key and matched content never appear).

**Analysis**
- A missing run.
- Only one run.
- Zero AIO.
- Empty-set Jaccard.
- A query whose domains are all `not_a_brand`: its share is NaN, it is dropped from the mean, and it is counted in `n_queries_empty` and in the denominator of `pct_queries_present`.
- Bounds on a hand-computed example: 1 big, 0 small, 2 unknown and 1 unconfirmed candidate give a small-share point estimate of 0, a lower bound of 0/4 and an upper bound of 3/4.
- The `provisional` flag on either side of the threshold.
- A zero small share gives a ratio of "∞ (provisional)".
- The banner appears in `report.md` exactly when some row is provisional.

**CLI**
- Exit codes 2, 3 and 4.
- A non-TTY without `--yes` refuses.
- `--dry-run` makes 0 paid calls.
- The `load_env=False` path.
- Output is redacted.
- `--study-cap` help text.

**Ollama helper**
- Tag already present.
- Placeholder tag.
- Low disk.
- Several candidate paths.
- No checkable path.
- Remote base URL.
- No `nvidia-smi`.
- Low GPU.
- An error line in the stream with HTTP 200.
- An exception inside the block.
- `KeyboardInterrupt`.
- API delete fails and the CLI fallback succeeds. **The subprocess env has `OLLAMA_HOST` equal to the derived base-URL host even when the parent env has `OLLAMA_HOST=http://other:1234`.**
- A path-prefixed base URL means no subprocess call, and the manual message includes `OLLAMA_HOST=`.
- After the API delete fails, `list_tags` shows the tag is gone, so no fallback runs.
- Both deletes fail while another exception is in flight.
- `keep=True`.

---

## 5. Testing and verification

**Unit tests:** run `python -m pytest tests -q` from `ai-search-content-copilot/`. No `__init__.py` exists under `tests/`, and test basenames are unique.

`tests/conftest.py`:
- puts `PROJECT_ROOT` at the front of `sys.path`.

Its autouse fixtures:
- monkeypatch `socket.socket.connect` and `requests.sessions.Session.request` to raise `RuntimeError("network disabled in tests")`;
- monkeypatch `dotenv.load_dotenv` and `brand_study.cli.load_dotenv` with a stub that **fails the test if called**. CLI tests also pass `load_env=False`.
- set `SERPAPI_API_KEY=test_key_DO_NOT_LEAK_1234567890`, which is at least 16 characters so `scan-leaks` accepts it;
- delete any inherited `OLLAMA_HOST` and `OLLAMA_BASE_URL`, so tests don't depend on the machine;
- point `BRAND_STUDY_OUTPUT_DIR` at `tmp_path`.

SerpApi calls go through a `FakeSession`. It is keyed by `(engine, q, location)` for main searches and by `(engine, page_token)` for follow-ups. It can be scripted to fail per attempt (5xx, transport, 401, quota 429) and records every call.

The test files:
- `test_bs_collect.py` covers:
  - end-to-end over 3 fake queries, including the two "near me" queries with the same text;
  - a repeat run makes 0 calls;
  - one ledger line per attempt;
  - budget refusals;
  - the immediate follow-up;
  - follow-up failure caching;
  - the reserve-2 stop;
  - retry crossing the budget limit;
  - the property test;
  - retry-failed-aio.
- `test_bs_cli.py` calls `cli.main([...], load_env=False)` with an injected session factory. It checks exit codes, non-TTY refusal, dry-run with 0 paid calls, and that the key never appears in output. It also covers `scan-leaks`.
- `test_bs_extract.py`, `test_bs_labeling.py`, `test_bs_analyze.py` and `test_bs_report.py` use hand-computed expectations:
  - Yelp gets a `domain_share` of 1/9 and a `result_share` of 2/10;
  - Jaccard of 3 shared out of 5 is 0.6;
  - the Wilson CI;
  - NaN and zero-denominator handling;
  - the bounds example;
  - the `aio_status` table, row by row;
  - `report.md` contains the disclaimer, the definitions, the undercount limitation and the provisional banner.
- `test_bs_ollama_lifecycle.py` uses `FakeOllamaAPI` plus injected `gpu_query`, `disk_free`, `rm_fallback`, `run` and `which`. No real subprocess and no HTTP.

**Live smoke test** (≤ 5 searches in total, only after all tests pass and the dry run works):
1. `python -m brand_study check-queries`
2. `python -m brand_study account` (free).
3. `python -m brand_study collect --run smoke --query-ids skn_b_01,cof_l_01 --max-new 4 --retries 0 --dry-run`
4. `python -m brand_study collect --run smoke --query-ids skn_b_01,cof_l_01 --max-new 4 --retries 0 --yes`. This is at most 2 main + 2 follow-ups = 4 attempts, a limit the runtime budget enforces. `--retries 0` keeps the count exact.
5. Inspect the cached JSON: `organic_results`, `ai_overview.text_blocks`, `references`, `page_token` and `local_results.places`. Fix the extractors and fixtures if they differ. If the docs couldn't be read, record "verified against live responses only" and tell the user.
6. `python -m brand_study scan-leaks`. It prints file names and a count only, and the key never goes to the shell.
7. `python -m brand_study analyze --runs smoke`. A PROVISIONAL banner is expected.
8. Confirm the Account API delta is ≤ 4 and that the ledger has ≤ 4 lines, and report the exact numbers to the user. These count toward the study cap of 300.

**Then stop. The user runs run1 and run2.**

---

## 6. Out of scope
- Part 2: page rewriting, judging, and any real `ollama pull`. Only the helper and its mocked tests are in scope.
- Any source other than SerpApi Google. Scraping cited pages or extracting businesses from inside listicles (named as a limitation instead).
- Local-pack data in the headline metrics. It is descriptive only and used for confirmation and suggestions. Pages 2+, mobile, other states.
- Classifying businesses automatically with an LLM or business-size data. Applying review suggestions automatically.
- Committing, pushing, editing `.env` or `.git/info/exclude`, or changing existing pipeline code, apart from the README section and the new `.gitignore`.
- Significance testing beyond Wilson CIs.

---

## Response to review

I accept all seven issues. Here is where each one lands, with the few places where I chose a specific option or went a little further:

1. **Retries vs the budget: accepted, using your first option (count each attempt).** I didn't make it depend on billing facts. Even if the docs say 429 and 5xx aren't billed, the guard still counts them, so it stays worst-case and doesn't rely on a doc page being right. The verified facts only fill in `charged_assumed`. To do this cleanly, retries moved out of `serpapi_client` and into `collect.py`, so every attempt passes through `RuntimeBudget`. The reserve of 2 now applies before every main-search attempt, including retries, which is what removes the conflict you found. I also reduced the default to 1 retry and added a separate retry headroom to the estimate. That keeps the worst case at 220 searches for two runs, well under the 300 cap. The property test checks the invariant directly: ledger lines ≤ allowed.
2. **`ollama rm` targeting: accepted, using both of your fixes in layers.** The subprocess always gets `OLLAMA_HOST` derived from `OLLAMA_BASE_URL`, so the inherited value is overridden rather than just compared. If the URL can't be turned into a CLI host (a path prefix, credentials or a parse failure), the fallback is turned off. I also added a `list_tags` pre-check so the fallback doesn't run when the HTTP delete actually succeeded. The manual-command message includes `OLLAMA_HOST=` for the same reason.
3. **Bias against small businesses before review: accepted.** You asked for three things, and each is in the plan:
   - an upper bound that counts unconfirmed candidates and unknown items as small;
   - a `provisional` flag column plus a report banner;
   - unknown and unconfirmed counts in the same row as big vs small.

   I went one step further to reduce the bias at its source rather than only flagging it. A new `local_pack_name` method confirms AIO list candidates that match a local-pack business for the same query, and `labels_to_review.csv` suggests `small_local` for local-pack websites. Both stay transparent: confirmation doesn't assign a category, and suggestions are never applied automatically. One caveat is now in the report: the mention upper bound is loose, because candidates can be non-brands. I kept the point estimate restricted to confirmed items so it stays conservative, and the bounds show the full range.
4. **Ambiguous aliases: accepted, refined into three modes.** Case-sensitive matching alone doesn't fix "Target dark spots…" at the start of a sentence, so single common words get `case_mid`, which also excludes sentence and list-item starts. Lowercase brands like `illy` get `candidate`, and phrases like "The Ordinary" get `case`. Your two tests are in, plus the sentence-start case and "silly".
5. **Leak check: accepted.** `scan-leaks` loads the key through dotenv, refuses an empty or short key, scans for the raw and URL-encoded forms, and prints names and a count only. It replaces the grep in the smoke test.
6. **Test directory shadowing: accepted, using both safeguards.** The directory is renamed `tests/test_brand_study/`, there is no `__init__.py` anywhere under `tests/`, and basenames get a `test_bs_` prefix.
7. **Smaller points: all accepted.** I went beyond documenting `--max-searches` and **renamed it to `--study-cap`**. A name that reads as per-invocation would mislead even with good help text, and the task only asks that the cap be configurable, not that it have a particular name. I added `--max-new` for real per-invocation limits, and the smoke test now uses it. The broad-query big-vs-small test has three criteria (ownership, national retail distribution, multi-state footprint) and an auditable `basis` column. I added one constraint: the implementer checks ownership facts before seeding and leaves out what they can't verify instead of guessing. The FakeSession key includes `location`, and validation only warns when a local query doesn't name its city. The zero-denominator rule is defined and tested: the query is dropped from the mean, counted in `n_queries_empty`, and still counted as "not present" in `pct_queries_present`.