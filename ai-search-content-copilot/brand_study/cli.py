from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from brand_study import budget as bud
from brand_study import config
from brand_study.cache import LockHeld
from brand_study.config import ConfigError
from brand_study.serpapi_client import AuthError, QuotaError, redact

EXIT_OK, EXIT_BUDGET, EXIT_CONFIG, EXIT_LEAK, EXIT_MODEL = 0, 2, 3, 4, 5


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m brand_study",
                                description="Part 1: small vs big brand visibility in Google organic and AI Overviews")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("check-queries", help="validate the query file and print counts (0 searches)")
    q.add_argument("--queries", default=str(config.DEFAULT_QUERIES))

    sub.add_parser("account", help="show SerpApi account usage (free Account API, 0 searches)")

    c = sub.add_parser("collect", help="collect Google organic + AI Overview results via SerpApi")
    c.add_argument("--run", required=True, help="run label, e.g. run1, run2, smoke")
    c.add_argument("--queries", default=str(config.DEFAULT_QUERIES))
    c.add_argument("--dry-run", action="store_true", help="show the plan and estimate without spending searches")
    c.add_argument("--limit", type=int)
    c.add_argument("--sector", choices=config.SECTORS)
    c.add_argument("--query-ids", help="comma-separated query ids")
    c.add_argument("--study-cap", type=int, default=config.DEFAULT_STUDY_CAP,
                   help="Total SerpApi searches allowed across the WHOLE study (all runs, including smoke and "
                        "earlier invocations, counted from ledger.jsonl). Not per invocation.")
    c.add_argument("--max-new", type=int, help="Optional hard limit on searches in THIS invocation.")
    c.add_argument("--followup-rate", type=float, default=config.DEFAULT_FOLLOWUP_RATE,
                   help="planning assumption: share of searches needing an AI Overview follow-up (1.0 = worst case)")
    c.add_argument("--retries", type=int, default=config.DEFAULT_RETRIES,
                   help="extra attempts per request on 5xx/429/transport errors (each attempt is charged)")
    c.add_argument("--retry-headroom", type=int, default=config.DEFAULT_RETRY_HEADROOM,
                   help="searches added to the estimate for retries")
    c.add_argument("--sleep", type=float, default=config.DEFAULT_SLEEP_S, help="seconds between live requests")
    c.add_argument("--yes", action="store_true", help="do not prompt for confirmation")
    c.add_argument("--no-account", action="store_true", help="skip the Account API call (dry run only)")
    c.add_argument("--retry-failed-aio", action="store_true",
                   help="re-request cached searches whose AI Overview follow-up failed (needs --query-ids)")
    c.add_argument("--min-hours-between-runs", type=float, default=config.DEFAULT_MIN_HOURS_BETWEEN_RUNS)
    c.add_argument("--force-spacing", action="store_true")

    a = sub.add_parser("analyze", help="build tidy CSVs and report.md from the cache (0 searches)")
    a.add_argument("--runs", default=",".join(config.DEFAULT_RUNS))
    a.add_argument("--queries", default=str(config.DEFAULT_QUERIES))
    a.add_argument("--rules", default=str(config.DEFAULT_RULES))
    a.add_argument("--overrides", default=str(config.DEFAULT_OVERRIDES))
    a.add_argument("--provisional-threshold", type=float, default=config.DEFAULT_PROVISIONAL_THRESHOLD)
    a.add_argument("--llm-overrides", help="generated LLM labels file (default brand_study/data/llm_label_overrides.csv "
                                           "or env BRAND_STUDY_LLM_OVERRIDES)")
    a.add_argument("--no-llm-labels", action="store_true", help="ignore the generated LLM labels")

    def common(sp):
        sp.add_argument("--runs", default=",".join(config.DEFAULT_RUNS))
        sp.add_argument("--queries", default=str(config.DEFAULT_QUERIES))
        sp.add_argument("--rules", default=str(config.DEFAULT_RULES))
        sp.add_argument("--overrides", default=str(config.DEFAULT_OVERRIDES))

    ll = sub.add_parser("llm-label", help="label the unknown domains/entities with three LLM judges")
    common(ll)
    ll.add_argument("--judges", default=",".join(config.LLM_JUDGE_ORDER),
                    help="comma-separated subset of: " + ", ".join(config.LLM_JUDGE_ORDER))
    ll.add_argument("--limit", type=int, help="only the first N items (fixed shuffled order); a trial run")
    ll.add_argument("--batch-size", type=int, default=config.DEFAULT_LLM_BATCH)
    ll.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True,
                    help="fetch each selected domain's homepage once as extra evidence (cached); --no-fetch makes "
                         "no requests (cached homepages are still used)")
    ll.add_argument("--refetch", action="store_true", help="ignore the homepage cache")
    ll.add_argument("--fetch-only", action="store_true", help="fetch homepages, write items.jsonl and stop")
    ll.add_argument("--retry-abstentions", action="store_true")
    ll.add_argument("--rejudge-on-digest-change", action="store_true")
    ll.add_argument("--reuse-stale-votes", action="store_true",
                    help="aggregate with the newest vote of an item whose evidence changed since it was judged")
    ll.add_argument("--accept", choices=("unanimous", "majority"), default="unanimous")
    ll.add_argument("--aggregate-only", action="store_true", help="no model calls; rebuild the output files")
    ll.add_argument("--llm-overrides", help="where to write the accepted labels")
    ll.add_argument("--claude-diagnose", action="store_true",
                    help="send one batch to Claude with JSON output and print usage details; writes no votes")
    ll.add_argument("--dry-run", action="store_true", help="show the work without calling any model or fetching")
    ll.add_argument("--yes", action="store_true", help="do not prompt for confirmation")

    ls = sub.add_parser("llm-sample", help="draw the blinded, stratified human-validation sample")
    common(ls)
    ls.add_argument("--n", type=int, default=config.DEFAULT_SAMPLE_N)
    ls.add_argument("--seed", type=int, default=config.DEFAULT_SAMPLE_SEED)
    ls.add_argument("--extend", type=int, metavar="K", help="add K more items to the existing sample")
    ls.add_argument("--accept", choices=("unanimous", "majority"),
                    help="acceptance policy to evaluate (default: the `# policy=` header of the accepted LLM labels "
                         "file written by llm-label, else unanimous); must match that header when both exist")
    ls.add_argument("--llm-overrides", help="accepted LLM labels file whose policy is used (default "
                                            "brand_study/data/llm_label_overrides.csv or env BRAND_STUDY_LLM_OVERRIDES)")
    ls.add_argument("--allow-pending", action="store_true")
    ls.add_argument("--force", action="store_true")

    la = sub.add_parser("llm-agreement", help="score the judges against your labels in the validation sample")
    la.add_argument("--sample")
    la.add_argument("--snapshot")
    la.add_argument("--out")
    la.add_argument("--snippet-out")

    s = sub.add_parser("scan-leaks", help="check output files for the API key; prints file names only (0 searches)")
    s.add_argument("--path", action="append", help="directory or file to scan (repeatable)")
    return p


def _cmd_check_queries(args, out) -> int:
    from brand_study.queries import load_queries, query_counts
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qs = load_queries(args.queries)
    for w in caught:
        out(f"WARNING: {w.message}")
    counts = query_counts(qs)
    out(f"{len(qs)} queries OK ({Path(args.queries).name})")
    for name, c in counts.items():
        out(f"  by {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    return EXIT_OK


def _cmd_account(args, out, session) -> int:
    from brand_study.serpapi_client import fetch_account
    key = config.get_api_key()
    try:
        fields = fetch_account(key, session=session)
    except RuntimeError as exc:
        out(f"error: {exc}")
        return EXIT_CONFIG
    for k, v in fields.items():
        out(f"{k}: {v}")
    return EXIT_OK


def _cmd_collect(args, out, session, input_fn, isatty, sleep_fn) -> int:
    from brand_study.collect import build_plan, run_collection
    from brand_study.queries import load_queries
    if args.no_account and not args.dry_run:
        raise ConfigError("--no-account is only allowed with --dry-run")
    qs = load_queries(args.queries, warn=False)
    ids = [x.strip() for x in args.query_ids.split(",") if x.strip()] if args.query_ids else None
    plan = build_plan(qs, args.run, sector=args.sector, query_ids=ids, limit=args.limit,
                      retry_failed_aio=args.retry_failed_aio)
    key = config.get_api_key(required=not args.dry_run)
    run_collection(plan, key, dry_run=args.dry_run, cap=args.study_cap, max_new=args.max_new,
                   followup_rate=args.followup_rate, retries=args.retries, retry_headroom=args.retry_headroom,
                   sleep_s=args.sleep, yes=args.yes, no_account=args.no_account,
                   min_hours_between_runs=args.min_hours_between_runs, force_spacing=args.force_spacing,
                   session=session, log=out, input_fn=input_fn, isatty=isatty, sleep_fn=sleep_fn)
    return EXIT_OK


def _runs(text: str) -> list[str]:
    return [r.strip() for r in text.split(",") if r.strip()]


def _cmd_analyze(args, out) -> int:
    from brand_study.analyze import run_analysis
    from brand_study.labeling import Labeler, load_llm_overrides, load_overrides, load_rules
    from brand_study.llm_evidence import collected_runs
    from brand_study.llm_validation import validation_summary_line
    from brand_study.queries import load_queries
    from brand_study.report import build_report, write_report
    runs = _runs(args.runs)
    qs = load_queries(args.queries, warn=False)
    overrides = load_overrides(args.overrides)
    if args.llm_overrides and not args.no_llm_labels and not Path(args.llm_overrides).exists():
        raise ConfigError(f"--llm-overrides {args.llm_overrides}: file not found")
    llm, meta = ({}, {}) if args.no_llm_labels else load_llm_overrides(args.llm_overrides)
    llm_meta = {"policy": meta.get("policy", ""), "prompt_version": meta.get("prompt_version", "")}
    if llm:
        have = collected_runs(runs)
        labelled = _runs(meta.get("runs", ""))
        if labelled and set(labelled) != set(have):
            llm_meta["runs_note"] = (f"the LLM labels were made from runs {','.join(labelled)} but runs "
                                     f"{','.join(have) or 'none'} are being analysed; re-run llm-label to refresh them.")
            out(f"WARNING: {llm_meta['runs_note']}")
        llm_meta["validation_summary"] = validation_summary_line()
    labeler = Labeler(load_rules(args.rules), overrides, llm)
    ledger = bud.read_ledger()
    frames = run_analysis(qs, runs, labeler, threshold=args.provisional_threshold, ledger_rows=ledger)
    text = build_report(frames, runs, queries_file=Path(args.queries).name, ledger_rows=ledger,
                        overrides_count=len(overrides), threshold=args.provisional_threshold,
                        llm_overrides_count=len(llm), llm_meta=llm_meta)
    path = write_report(text)
    out(f"wrote {len(frames)} CSVs and {path}")
    if text.startswith("> **PROVISIONAL"):
        out("PROVISIONAL: review labels_to_review.csv, fill label_overrides.csv, then re-run analyze")
    return EXIT_OK


def _parse_judges(text: str) -> list[str]:
    names = _runs(text)
    bad = [n for n in names if n not in config.LLM_JUDGES]
    if bad or not names:
        raise ConfigError(f"unknown judge(s) {bad or text!r}; choose from {', '.join(config.LLM_JUDGE_ORDER)}")
    return names


MISSING_RUN_WARNING = ("{missing} not collected yet. Labeling now means items first seen in {missing} need new votes "
                       "later, and some run1 items may get new evidence. Recommended: run llm-label after the last "
                       "collection run.")


def _prepare(args, out):
    """Item set (rules + human overrides only), evidence index and collected runs."""
    from brand_study import llm_evidence as ev
    from brand_study.labeling import Labeler, load_overrides, load_rules
    from brand_study.queries import load_queries
    runs = _runs(args.runs)
    qs = load_queries(args.queries, warn=False)
    labeler = Labeler(load_rules(args.rules), load_overrides(args.overrides))
    collected = ev.collected_runs(runs)
    missing = [r for r in runs if r not in collected]
    if not collected:
        return None
    items, frames = ev.build_items(qs, collected, labeler)
    idx = ev.build_index(qs, collected)
    return {"runs": runs, "collected": collected, "missing": missing, "items": items, "frames": frames, "idx": idx}


def _confirm(args, out, input_fn, isatty, what: str) -> bool:
    if args.yes:
        return True
    if not isatty():
        raise ConfigError(f"{what}: stdin is not a terminal; pass --yes to run non-interactively")
    return input_fn(f"{what}: proceed? [y/N] ").strip().lower() in ("y", "yes")


def _cmd_llm_label(args, out, input_fn, isatty, sleep_fn, hooks) -> int:
    from brand_study import llm_aggregate as la
    from brand_study import llm_evidence as ev
    from brand_study import llm_judges as lj
    from brand_study.cache import CollectLock
    from brand_study.ollama_lifecycle import _sigterm_as_exit
    judges = _parse_judges(args.judges)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be >= 1")
    prep = _prepare(args, out)
    if prep is None:
        out(f"nothing to label: no raw cache for runs {args.runs}")
        return EXIT_OK
    items = prep["items"]
    nd = sum(1 for i in items if i["key_type"] == "domain")
    out(f"{len(items)} items to label ({nd} domains, {len(items) - nd} entities); prompt {lj.PROMPT_VERSION}; "
        f"runs collected: {','.join(prep['collected'])}" + (f"; missing: {','.join(prep['missing'])}"
                                                             if prep["missing"] else ""))
    if prep["missing"]:
        out("WARNING: " + MISSING_RUN_WARNING.format(missing=",".join(prep["missing"])))
    selected = ev.order_items(items, limit=args.limit)
    sel_ids = [i["item_id"] for i in selected]
    previous = {r["item_id"]: r.get("evidence_sha") for r in ev.read_items()}

    def evidence():
        recs = ev.build_evidence(items, prep["idx"], prep["frames"]["aio_mentions"], prep["collected"],
                                 fetch_enabled=args.fetch)
        by_id = {r["item_id"]: r for r in recs}
        return recs, [by_id[i] for i in sel_ids]

    store = lj.VoteStore(log=out)
    clock = hooks.get("clock") or __import__("time").time
    claude_runner = hooks.get("claude_runner")

    probed = {}  # a Claude judge probed before the confirmation prompt, reused for the run

    def claude_judge():
        if probed.get("judge") is not None:
            return probed.pop("judge")
        binary = hooks.get("claude_binary") or lj.find_claude_binary(runner=claude_runner)
        model = config.LLM_JUDGES["claude"][1]
        return lj.ClaudeJudge("claude", model, binary, runner=claude_runner, log=out, sleep=sleep_fn, now=clock)

    if args.dry_run:
        records, sel_records = evidence()  # cached fetches only
        return _llm_dry_run(args, out, prep, records, sel_records, previous, judges, store, claude_judge)
    lock = config.llm_dir() / ".llm_label.lock"
    if args.aggregate_only:
        if args.limit is not None:
            raise ConfigError("--aggregate-only always aggregates the full item set; drop --limit")
        records, _ = evidence()
        with _sigterm_as_exit():
            with CollectLock(lock, what="llm-label"):
                ev.write_items(records)
                la.aggregate(records, store, policy=args.accept, reuse_stale=args.reuse_stale_votes, limited=False,
                             out_path=args.llm_overrides, runs=prep["collected"], log=out)
        return EXIT_OK
    if args.claude_diagnose and "claude" not in judges:
        raise ConfigError("--claude-diagnose sends a batch to Claude; include claude in --judges")
    if args.claude_diagnose and not sel_ids:
        out("nothing to diagnose: no items selected")
        return EXIT_OK

    # Plan and confirmation BEFORE any homepage request, using cached fetches only (hashes may not be final yet).
    _, plan_records = evidence()
    to_fetch = _n_to_fetch(args, plan_records)
    fetch_note = (f"up to {to_fetch} homepage requests first (one per domain"
                  + ("" if args.refetch else " not in the fetch cache") + ")" if to_fetch else "no homepage requests")
    need_claude = bool(args.claude_diagnose)
    if args.fetch_only:
        what = f"fetch-only: {fetch_note}"
    elif args.claude_diagnose:
        what = f"{fetch_note}, then send one batch to Claude (diagnostic)"
    else:
        for row in lj.dry_run_rows(plan_records, judges, store, args.batch_size,
                                   retry_abstentions=args.retry_abstentions,
                                   rejudge_on_digest_change=args.rejudge_on_digest_change):
            out(f"  {row.name} ({row.model}): {row.cached} cached, {row.pending} pending in {row.batches} batches"
                + (f" (hash not final until fetched: {row.not_final})" if row.not_final else "") + f"; {row.pull}")
            # pending + not-final: a fetch may change the hash, so Claude may still have work after fetching
            need_claude = need_claude or (row.name == "claude" and (row.pending or row.not_final))
        what = f"{fetch_note}, then run the LLM judges"
    if need_claude:
        # Resolve and probe the Claude CLI now, before any fetch or Ollama pull: a missing binary or a CLI that
        # cannot run without tools fails here instead of hours later, and the user sees the isolation flags.
        probed["judge"] = claude_judge()
        j = probed["judge"]
        out(f"  claude CLI {j.cli_version or '?'} ({j.binary}); flags: {' '.join(j.flag_names())}")
    try:
        return _llm_label_run(args, out, input_fn, isatty, sleep_fn, hooks, what, prep, judges, selected, evidence,
                              store, claude_judge, lock)
    finally:
        if probed.get("judge") is not None:  # not handed out (cancelled, failed, or Claude had nothing left)
            probed.pop("judge").close()


def _llm_label_run(args, out, input_fn, isatty, sleep_fn, hooks, what, prep, judges, selected, evidence, store,
                   claude_judge, lock) -> int:
    """Confirmation, then (under the SIGTERM handler and the run lock) fetch, items.jsonl, judges and aggregation."""
    import requests
    from brand_study import llm_aggregate as la
    from brand_study import llm_evidence as ev
    from brand_study import llm_judges as lj
    from brand_study.cache import CollectLock
    from brand_study.ollama_lifecycle import OllamaAPI, _sigterm_as_exit, ollama_model
    if not _confirm(args, out, input_fn, isatty, what):
        out("cancelled")
        return EXIT_OK

    api = hooks.get("ollama_api") or OllamaAPI()

    def make_judge(name):
        kind, model = config.LLM_JUDGES[name]
        return (lj.OllamaJudge(name, model, api, log=out, sleep=sleep_fn) if kind == "ollama"
                else claude_judge())

    def ollama_context(name, tag):
        if hooks.get("ollama_context"):
            return hooks["ollama_context"](name, tag)
        return ollama_model(tag, api=api, expected_size_gb=config.LLM_EXPECTED_SIZE_GB.get(name, 25.0),
                            required_gpu_gb=config.LLM_REQUIRED_GPU_GB.get(name, 24.0), log=out)

    # SIGTERM handling and the run lock cover fetch -> items.jsonl -> judges -> aggregate.
    with _sigterm_as_exit():
        with CollectLock(lock, what="llm-label"):
            if args.fetch:
                session = hooks.get("fetch_session") or requests.Session()
                res = ev.fetch_items(selected, prep["idx"], session,
                                     hooks.get("resolver") or __import__("socket").getaddrinfo,
                                     refetch=args.refetch, sleep=sleep_fn, log=out)
                out(f"homepages: {res['fetched']} fetched ({res['failed']} failed), {res['cached']} cached")
            records, sel_records = evidence()
            ev.write_items(records)
            if args.fetch_only:
                out(f"wrote {ev.items_path()}; stopping (--fetch-only)")
                return EXIT_OK
            if args.claude_diagnose:
                batch = lj.make_batches(sel_records, args.batch_size)[0]
                judge = claude_judge()
                try:
                    info = judge.diagnose(lj.SYSTEM_TEXT, batch.text)
                finally:
                    judge.close()
                out(f"claude diagnose (workdir {judge.workdir}, removed):")
                for k, v in info.items():
                    out(f"  {k}: {v}")
                return EXIT_OK
            lj.run_judges(sel_records, judges, store, batch_size=args.batch_size, make_judge=make_judge,
                          ollama_context=ollama_context, runs=prep["collected"],
                          retry_abstentions=args.retry_abstentions,
                          rejudge_on_digest_change=args.rejudge_on_digest_change, log=out)
            lj.warn_mixed(store, config.LLM_JUDGE_ORDER, out)
            limited = args.limit is not None
            la.aggregate(sel_records if limited else records, store, policy=args.accept,
                         reuse_stale=args.reuse_stale_votes, limited=limited, out_path=args.llm_overrides,
                         runs=prep["collected"], log=out)
    return EXIT_OK


def _n_to_fetch(args, sel_records) -> int:
    if not args.fetch:
        return 0
    if args.refetch:
        return sum(1 for r in sel_records if r["key_type"] == "domain")
    return sum(1 for r in sel_records if not r.get("hash_final", True))


def _llm_dry_run(args, out, prep, records, sel_records, previous, judges, store, claude_judge) -> int:
    from brand_study import llm_evidence as ev
    from brand_study import llm_judges as lj
    from brand_study.llm_aggregate import item_votes
    out(f"DRY RUN (no model, pull or fetch calls): {len(sel_records)} selected items, prompt {lj.PROMPT_VERSION}")
    new = sum(1 for r in records if r["item_id"] not in previous)
    changed = sum(1 for r in records if r["item_id"] in previous and previous[r["item_id"]] != r["evidence_sha"])
    out(f"  evidence vs last items.jsonl: {new} new items, {changed} with changed evidence"
        + ("" if previous else " (no previous items.jsonl)"))
    if args.fetch_only:
        n_fetch = _n_to_fetch(args, sel_records)
        out(f"  --fetch-only: would fetch up to {n_fetch} homepages and write {ev.items_path()}; "
            f"nothing written now, no model calls")
        return EXIT_OK
    if args.reuse_stale_votes:
        reuse = sum(1 for r in sel_records if item_votes(r, store, config.LLM_JUDGE_ORDER, True)[1])
        out(f"  --reuse-stale-votes: {reuse} items would reuse an earlier vote")
    for row in lj.dry_run_rows(sel_records, judges, store, args.batch_size, retry_abstentions=args.retry_abstentions,
                               rejudge_on_digest_change=args.rejudge_on_digest_change):
        out(f"  {row.name} ({row.model}): cached {row.cached}, pending {row.pending} "
            f"(would fetch, hash not final: {row.not_final}), batches {row.batches}; {row.pull}")
    if "claude" in judges:
        try:
            j = claude_judge()
            out(f"  claude CLI {j.cli_version or '?'}; flags: {' '.join(j.flag_names())}")
            for n in j.notes:
                out(f"  note: {n}")
            j.close()
        except ConfigError as exc:
            out(f"  claude: {exc}")
    batches = lj.make_batches(sel_records, args.batch_size)
    if batches:
        text = f"{lj.SYSTEM_TEXT}\n\n---- user message (batch 1) ----\n{batches[0].text}"
        path = ev.atomic_write_text(config.llm_dir() / "dry_run_prompt.txt", text)
        out(f"  example prompt: {len(text)} chars (~{len(text) // 4} tokens); wrote {path}")
    return EXIT_OK


def _cmd_llm_sample(args, out) -> int:
    from brand_study import llm_evidence as ev
    from brand_study import llm_judges as lj
    from brand_study import llm_validation as lv
    prep = _prepare(args, out)
    if prep is None:
        out("nothing to sample: no raw cache")
        return EXIT_OK
    if args.extend:
        if args.accept:
            raise ConfigError("--extend keeps the policy frozen in the snapshot; drop --accept")
        records = ev.build_evidence(prep["items"], prep["idx"], prep["frames"]["aio_mentions"], prep["collected"],
                                    fetch_enabled=False)
        lv.extend_sample(args.extend, records, log=out)
        return EXIT_OK
    policy = _sample_policy(args, out)
    records = ev.build_evidence(prep["items"], prep["idx"], prep["frames"]["aio_mentions"], prep["collected"],
                                fetch_enabled=False)
    lv.make_sample(records, lj.VoteStore(log=out), n=args.n, seed=args.seed, policy=policy,
                   runs=prep["collected"], force=args.force, allow_pending=args.allow_pending, log=out)
    return EXIT_OK


def _sample_policy(args, out) -> str:
    """The acceptance policy the validation scores: the one llm-label actually applied (accepted file header)."""
    from brand_study.labeling import read_llm_overrides_meta
    path = Path(args.llm_overrides) if args.llm_overrides else config.llm_overrides_path()
    if args.llm_overrides and not path.exists():
        raise ConfigError(f"--llm-overrides {args.llm_overrides}: file not found")
    used = read_llm_overrides_meta(path).get("policy", "")
    if args.accept and used and args.accept != used:
        raise ConfigError(f"--accept {args.accept} differs from the policy llm-label applied ({used}, in {path.name}); "
                          f"the validation must score the labels that were actually accepted. Re-run "
                          f"`llm-label --aggregate-only --accept {args.accept}` first, or drop --accept.")
    policy = args.accept or used or "unanimous"
    if not used:
        out(f"note: no accepted LLM labels file with a policy header ({path}); evaluating policy {policy}")
    else:
        out(f"acceptance policy: {policy} (from {path.name})")
    return policy


def _cmd_llm_agreement(args, out) -> int:
    from brand_study import llm_validation as lv
    lv.evaluate(args.sample, args.snapshot, args.out, args.snippet_out, log=out)
    return EXIT_OK


def _cmd_scan_leaks(args, out) -> int:
    from brand_study.leaks import scan_for_key
    key = os.environ.get("SERPAPI_API_KEY", "")
    paths = [Path(p) for p in (args.path or [str(config.output_dir())])]
    try:
        hits, n = scan_for_key(paths, key)
    except ValueError as exc:
        out(f"error: {exc}")
        return EXIT_CONFIG
    if not hits:
        out(f"clean ({n} files scanned)")
        return EXIT_OK
    out(f"LEAK in {len(hits)} files:")
    for h in hits:
        try:
            out(f"  {h.resolve().relative_to(config.PROJECT_ROOT)}")
        except ValueError:
            out(f"  {h}")
    return EXIT_LEAK


def main(argv: list[str] | None = None, load_env: bool = True, *, session_factory: Callable | None = None,
         out: Callable[[str], None] | None = None, input_fn: Callable[[str], str] = input,
         isatty: Callable[[], bool] | None = None, sleep_fn=None, llm_hooks: dict | None = None) -> int:
    """llm_hooks (tests): ollama_api, ollama_context, claude_runner, claude_binary, fetch_session, resolver, clock."""
    import time
    if load_env:
        load_dotenv(config.PROJECT_ROOT / ".env")
    out = out or (lambda s: print(s, flush=True))
    key = os.environ.get("SERPAPI_API_KEY", "")
    safe_out = (lambda s: out(redact(s, key))) if key else out
    args = _parser().parse_args(argv)
    session = session_factory() if session_factory else None
    isatty = isatty or (lambda: sys.stdin.isatty())
    sleep_fn = sleep_fn or time.sleep
    try:
        if args.cmd == "check-queries":
            return _cmd_check_queries(args, safe_out)
        if args.cmd == "account":
            return _cmd_account(args, safe_out, session)
        if args.cmd == "collect":
            return _cmd_collect(args, safe_out, session, input_fn, isatty, sleep_fn)
        if args.cmd == "analyze":
            return _cmd_analyze(args, safe_out)
        if args.cmd == "scan-leaks":
            return _cmd_scan_leaks(args, safe_out)
        if args.cmd == "llm-label":
            return _cmd_llm_label(args, safe_out, input_fn, isatty, sleep_fn, llm_hooks or {})
        if args.cmd == "llm-sample":
            return _cmd_llm_sample(args, safe_out)
        if args.cmd == "llm-agreement":
            return _cmd_llm_agreement(args, safe_out)
    except (bud.BudgetError, QuotaError) as exc:
        safe_out(f"REFUSED: {exc}")
        return EXIT_BUDGET
    except (ConfigError, AuthError, LockHeld) as exc:
        safe_out(f"error: {exc}")
        return EXIT_CONFIG
    except Exception as exc:
        from brand_study.llm_judges import JudgeError
        from brand_study.ollama_lifecycle import OllamaError
        if isinstance(exc, (JudgeError, OllamaError)):
            safe_out(f"model error: {exc}")
            return EXIT_MODEL
        raise
    return EXIT_CONFIG
