# SerpApi facts used by `brand_study`

The table below was written from the documentation pages listed, read on 2026-11-10 during an earlier
implementation session. Those pages were **not** re-fetched afterwards. If SerpApi changes its API, re-check these
facts and the extractors in `extract.py`.

**What was double-checked.** The last column says whether the live smoke run (3 searches, 2026-11-10) also confirmed
the fact. "docs only" means the fact rests on that single documentation read: in particular billing, `page_token`
expiry, the 429/401 error bodies, the `num` parameter and the Locations strings were not exercised live.

| Fact | Value used | Source | Live-checked |
|---|---|---|---|
| Search endpoint | `GET https://serpapi.com/search.json`, `engine=google` | serpapi.com/search-api | yes |
| Parameters | `q`, `location`, `gl=us`, `hl=en`, `google_domain=google.com`, `device=desktop`, `no_cache=true` | serpapi.com/search-api | yes (request accepted) |
| `num` | Not documented (pagination is `start`), so it is not sent | serpapi.com/search-api | docs only |
| Cache | "Cached searches are free". The cache lasts 1 h and needs identical parameters. `no_cache=true` forces a fresh search, which counts as a normal search | serpapi.com/search-api | docs only |
| AI Overview key | `ai_overview`, only for `hl=en` and a limited set of `gl` values | serpapi.com/ai-overview | yes |
| `text_blocks[].type` | `paragraph`, `heading`, `list`, `expandable`, `comparison` (also `table` and `top_stories` in examples). Sub-fields: `snippet`, `snippet_highlighted_words`, `reference_indexes`, `list[]` (`title`, `snippet`, nested `list`), `table`, `comparison`, nested `text_blocks` | serpapi.com/ai-overview | partly (paragraph, list; product lists differ, see below) |
| `references[]` | `title`, `link`, `snippet`, `source`, `index`. `reference_indexes` point at `index` | serpapi.com/ai-overview | yes |
| Follow-up needed | When Google loads the AIO separately, `ai_overview` holds only `page_token` and `serpapi_link` | serpapi.com/ai-overview | yes |
| `page_token` expiry | **Expires within 1 minute** of the search, so the follow-up is made immediately | serpapi.com/google-ai-overview-api | docs only |
| Follow-up request | `engine=google_ai_overview`, `page_token=<token>`. The response shape is not documented, so the extractor accepts both `{"ai_overview": {...}}` and a bare `text_blocks` object | serpapi.com/google-ai-overview-api | yes (shape recorded below) |
| Billing | "Only successful searches are counted ... Cached, errored, and failed searches are not." Empty result sets count as 1 search. Follow-up billing is not stated separately, so it is assumed charged | serpapi.com/pricing (FAQ) | partly (Account delta 3 for 3 successful searches) |
| Account API | `GET https://serpapi.com/account.json`, free. Fields used: `plan_name`, `searches_per_month`, `plan_searches_left`, `extra_credits`, `total_searches_left`, `this_month_usage`, `this_hour_searches`, `last_hour_searches`, `account_rate_limit_per_hour`. `api_key` and `account_email` are dropped | serpapi.com/account-api | yes |
| Organic results | `organic_results[]`: `position`, `title`, `link`, `redirect_link`, `displayed_link`, `snippet`, `source` | serpapi.com/organic-results | yes |
| Local pack | Google Search returns `local_results: {"places": [...]}`. The `google_local` engine returns a list, and both shapes are accepted. Place fields: `position`, `title`, `rating`, `reviews`, `type`, `address`, `place_id`, `links.website` | serpapi.com/search-api, serpapi.com/local-results | yes (`places`, `links.website`) |
| Locations | Florida is `Florida,United States` (State). The cities are `Miami,Florida,United States`, `Tampa,…`, `Orlando,…`, `Jacksonville,…`, `St. Petersburg,Florida,United States`, `Fort Lauderdale,Florida,United States` | serpapi.com/locations.json?q=… | Tampa and Florida only (used in smoke) |
| HTTP 429 | Ambiguous: the hourly throughput limit **or** an account that has run out of searches. The out-of-searches body is `{"error": "Your account has run out of searches."}`. The throughput message is not documented | serpapi.com/api-status-and-error-codes | docs only |
| HTTP 401 | `{"error": "Invalid API key. ..."}`. 403 means the account lacks permission | serpapi.com/api-status-and-error-codes | docs only |
| Other errors | 500/503 are server errors. `search_metadata.status == "Error"` with "We couldn't get valid results ...". Empty results return 200 with `error: "Google hasn't returned any results for this query."` | serpapi.com/api-status-and-error-codes | docs only |

## How the code uses these facts

- `charged_assumed` in the ledger is `true` only for attempts that returned HTTP 200 without an `error` body, plus the
  "no results" case. **The budget guard ignores this annotation and counts every attempt.**
- An error body containing "run out of searches", "out of searches", "searches left" or "plan limit"
  (`QUOTA_MARKERS` in `serpapi_client.py`) is classified as `quota` on **any** HTTP status, including 200, and aborts
  the run (exit 2). "searches left" could also match an hourly-limit message; that errs on the safe side (the run
  stops instead of spending). Any other 429 is treated as a throughput limit (retryable within budget).
- Field names were checked against the documentation above and against the live smoke-test responses (see the README).

## Checked against live responses (smoke run, 2026-11-10, 3 searches)

- `organic_results[]` had `position`, `link`, `title`, `source`, `redirect_link`, as documented. Only 7–8 organic
  results came back, so fewer than 10 is normal.
- The follow-up `engine=google_ai_overview` response is `{"search_metadata", "search_parameters", "ai_overview":
  {"text_blocks", "references"}}`. `references[]` had `index`, `link`, `title`, `source`, `snippet`.
- **Product lists inside an AIO:** list items have no `title`. They carry `shopping_result: {title, price, source,
  product_link, rating, reviews}` plus a nested `list` of snippets. `extract.py` reads `shopping_result.title`.
- **Most references for a product query were Google Shopping panels** (`https://www.google.com/search?ibp=oshop...`).
  `domains.host()` maps them to `shopping.google.com`, which is kept as its own unit and labeled
  `marketplace_retailer`.
- `local_results` was `{"places": [...], "more_locations_link"}`. The website is in `places[].links.website`.
- The Account API returned all whitelisted fields, including `this_hour_searches`.
