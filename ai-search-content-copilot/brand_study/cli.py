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

EXIT_OK, EXIT_BUDGET, EXIT_CONFIG, EXIT_LEAK = 0, 2, 3, 4


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


def _cmd_analyze(args, out) -> int:
    from brand_study.analyze import run_analysis
    from brand_study.labeling import Labeler, load_overrides, load_rules
    from brand_study.queries import load_queries
    from brand_study.report import build_report, write_report
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    qs = load_queries(args.queries, warn=False)
    overrides = load_overrides(args.overrides)
    labeler = Labeler(load_rules(args.rules), overrides)
    ledger = bud.read_ledger()
    frames = run_analysis(qs, runs, labeler, threshold=args.provisional_threshold, ledger_rows=ledger)
    text = build_report(frames, runs, queries_file=Path(args.queries).name, ledger_rows=ledger,
                        overrides_count=len(overrides), threshold=args.provisional_threshold)
    path = write_report(text)
    out(f"wrote {len(frames)} CSVs and {path}")
    if text.startswith("> **PROVISIONAL"):
        out("PROVISIONAL: review labels_to_review.csv, fill label_overrides.csv, then re-run analyze")
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
         isatty: Callable[[], bool] | None = None, sleep_fn=None) -> int:
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
    except (bud.BudgetError, QuotaError) as exc:
        safe_out(f"REFUSED: {exc}")
        return EXIT_BUDGET
    except (ConfigError, AuthError, LockHeld) as exc:
        safe_out(f"error: {exc}")
        return EXIT_CONFIG
    return EXIT_CONFIG
