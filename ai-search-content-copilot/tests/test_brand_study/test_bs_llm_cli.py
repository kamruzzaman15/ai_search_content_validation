from __future__ import annotations
import csv
import json
import os
import signal
from contextlib import contextmanager

import pytest

from bs_helpers import fx, write_wrap
from brand_study import cli, config
from brand_study.queries import load_queries
from test_bs_llm_judges import FakeClaude, answer_all
from test_bs_llm_evidence import FResp, public_resolver

HTML = b"<html><head><title>Home</title></head><body>Hello</body></html>"


class AnySession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return FResp(200, HTML)


class NoSession:
    def get(self, url, **kw):
        raise AssertionError(f"unexpected fetch {url}")


class OPost:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append(json["model"])
        user = json["messages"][1]["content"]
        content = answer_all("small_local")(None, {"input": user})[1]

        class R:
            status_code = 200
            text = ""

            def json(self_inner):
                return {"message": {"content": content}}
        return R()


class FakeOllamaAPI:
    base_url = "http://localhost:11434"

    def __init__(self):
        self.session = OPost()

    def model_digest(self, tag):
        return f"sha256:{tag}"


@pytest.fixture
def raw():
    qs = {x.id: x for x in load_queries(warn=False)}
    write_wrap(config.raw_dir(), "run1", qs["cof_l_01"], "search", fx("search_inline_aio.json"))
    write_wrap(config.raw_dir(), "run1", qs["cof_b_01"], "search", fx("search_no_aio.json"))
    return qs


def hooks(**kw):
    events = []

    @contextmanager
    def ctx(name, tag):
        events.append(("pull", name))
        yield tag
        events.append(("delete", name))
    h = {"claude_runner": FakeClaude(), "claude_binary": "/fake/claude", "fetch_session": AnySession(),
         "resolver": public_resolver, "ollama_api": FakeOllamaAPI(), "ollama_context": ctx, "clock": lambda: 1e9}
    h.update(kw)
    h["_events"] = events
    return h


def call(argv, h=None, isatty=False, answers=("y",)):
    lines = []
    it = iter(answers)
    code = cli.main(argv, load_env=False, out=lines.append, isatty=lambda: isatty, sleep_fn=lambda s: None,
                    input_fn=lambda prompt: next(it), llm_hooks=h or {})
    return code, "\n".join(lines)


def votes(judge):
    p = config.llm_dir() / f"votes_{judge}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_nothing_to_label():
    code, text = call(["llm-label", "--dry-run"])
    assert code == 0 and "nothing to label" in text


def test_dry_run_makes_no_calls(raw):
    h = hooks(fetch_session=NoSession())
    code, text = call(["llm-label", "--dry-run"], h)
    assert code == 0 and "DRY RUN" in text and "run2 not collected yet" in text
    assert h["claude_runner"].calls == [] and h["ollama_api"].session.posts == [] and h["_events"] == []
    assert "will pull" in text and '--tools ""' in text and "hash not final" in text
    assert (config.llm_dir() / "dry_run_prompt.txt").exists()
    assert not (config.llm_dir() / "items.jsonl").exists()


def test_claude_only_limit(raw, _isolated_llm):
    h = hooks()
    code, text = call(["llm-label", "--judges", "claude", "--limit", "2", "--yes"], h)
    assert code == 0, text
    assert len(votes("claude")) == 2 and votes("gemma") == [] and h["_events"] == []
    assert len(h["fetch_session"].calls) <= 2
    assert "trial run" in text and not _isolated_llm.exists()
    n_calls = len(h["claude_runner"].calls)
    code, text = call(["llm-label", "--judges", "claude", "--limit", "2", "--yes"], h)
    assert code == 0 and "nothing pending" in text and len(h["claude_runner"].calls) == n_calls
    v = votes("claude")[0]
    assert v["cli_version"].startswith("2.1.322") and v["temperature"] is None and v["runs"] == ["run1"]


def test_fetch_only_and_no_fetch(raw):
    h = hooks()
    code, text = call(["llm-label", "--fetch-only", "--limit", "1", "--yes"], h)
    assert code == 0 and len(h["fetch_session"].calls) <= 1 and h["claude_runner"].calls == []
    assert (config.llm_dir() / "items.jsonl").exists() and "--fetch-only" in text
    h2 = hooks(fetch_session=NoSession())
    code, _ = call(["llm-label", "--no-fetch", "--judges", "claude", "--limit", "1", "--yes"], h2)
    assert code == 0


def test_full_run_writes_redirected_accepted_file(raw, _isolated_llm):
    h = hooks()
    code, text = call(["llm-label", "--yes"], h)
    assert code == 0, text
    assert _isolated_llm.exists() and not config.DEFAULT_LLM_OVERRIDES.exists()
    assert h["_events"] == [("pull", "gemma"), ("delete", "gemma"), ("pull", "qwen"), ("delete", "qwen")]
    assert (config.results_dir() / "llm_disagreements.csv").exists()
    assert all(v["model_digest"] == "sha256:gemma4:31b" for v in votes("gemma"))
    # analyze picks the file up; --no-llm-labels ignores it
    code, text = call(["analyze", "--runs", "run1"])
    report = (config.results_dir() / "report.md").read_text()
    assert code == 0 and "**LLM labels:" in report
    code, _ = call(["analyze", "--runs", "run1", "--no-llm-labels"])
    report = (config.results_dir() / "report.md").read_text()
    assert "**LLM labels:" not in report and "LLM labels loaded (`llm_label_overrides.csv`): 0" in report
    # validation round trip through the CLI
    code, text = call(["llm-sample", "--runs", "run1"])
    assert code == 0, text
    sp = config.results_dir() / "llm_validation_sample.csv"
    lines = sp.read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    rows = list(csv.DictReader([ln for ln in lines if not ln.startswith("#")]))
    for r in rows:
        r["human_category"] = "small_local"
    with sp.open("w") as f:
        f.write("\n".join(head) + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    code, text = call(["llm-agreement"])
    assert code == 0, text
    assert (config.results_dir() / "llm_validation.md").exists()
    code, text = call(["llm-sample", "--runs", "run1"])
    assert code == 3 and "--extend" in text


def test_analyze_llm_overrides_path_and_runs_warning(raw, tmp_path):
    p = tmp_path / "custom.csv"
    p.write_text("# runs=run1,run2\n# policy=unanimous\nkey_type,key,category,note\n"
                 "domain,indiecafe.com,small_local,llm_unanimous; x\n")
    code, text = call(["analyze", "--llm-overrides", str(p)])
    report = (config.results_dir() / "report.md").read_text()
    assert code == 0 and "WARNING: the LLM labels were made from runs run1,run2" in text
    assert "LLM labels loaded (`llm_label_overrides.csv`): 1" in report


def test_exit_codes(raw):
    code, text = call(["llm-label", "--judges", "gpt", "--dry-run"])
    assert code == 3 and "unknown judge" in text
    h = hooks(claude_runner=FakeClaude([(2, "", "boom")]))
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--yes", "--no-fetch"], h)
    assert code == 5 and "exited 2" in text
    lock = config.llm_dir() / ".llm_label.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("{}")
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--yes", "--no-fetch"], hooks())
    assert code == 3 and str(lock) in text and "llm-label" in text


def test_confirmation(raw):
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--no-fetch"], hooks())
    assert code == 3 and "--yes" in text and "run2 not collected yet" in text
    h = hooks()
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--no-fetch"], h, isatty=True,
                      answers=("n",))
    assert code == 0 and "cancelled" in text and h["claude_runner"].calls == []


def test_claude_diagnose_writes_no_votes(raw):
    raw_json = json.dumps({"num_turns": 1, "usage": {"input_tokens": 3000}, "modelUsage": {"claude-opus-5-5": {}}})
    h = hooks(claude_runner=FakeClaude([(0, raw_json, "")]))
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--claude-diagnose", "--yes", "--no-fetch"],
                      h)
    assert code == 0 and "num_turns: 1" in text and votes("claude") == []


def test_sigterm_releases_lock(raw):
    def die(cmd, kw):
        os.kill(os.getpid(), signal.SIGTERM)
        return 0, "{}", ""
    h = hooks(claude_runner=FakeClaude([die]))
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as e:
        call(["llm-label", "--judges", "claude", "--limit", "1", "--yes", "--no-fetch"], h)
    assert e.value.code == 143
    assert not (config.llm_dir() / ".llm_label.lock").exists()
    assert signal.getsignal(signal.SIGTERM) == previous  # handler restored


def test_non_interactive_without_yes_fetches_nothing(raw):
    """Confirmation comes before any homepage request (live run and --fetch-only alike)."""
    for argv in (["llm-label", "--judges", "claude", "--limit", "2"], ["llm-label", "--fetch-only"]):
        h = hooks(fetch_session=NoSession())
        code, text = call(argv, h)
        assert code == 3 and "--yes" in text and "homepage requests" in text, text
        assert h["claude_runner"].calls == [] and not (config.llm_dir() / "items.jsonl").exists()


def test_interactive_cancel_fetches_nothing(raw):
    h = hooks(fetch_session=NoSession())
    code, text = call(["llm-label", "--judges", "claude", "--limit", "2"], h, isatty=True, answers=("n",))
    assert code == 0 and "cancelled" in text and "hash not final until fetched" in text


def test_fetch_runs_under_lock(raw):
    lock = config.llm_dir() / ".llm_label.lock"

    class LockCheckingSession(AnySession):
        def get(self, url, **kw):
            assert lock.exists(), "fetch outside the run lock"
            return super().get(url, **kw)
    h = hooks(fetch_session=LockCheckingSession())
    code, text = call(["llm-label", "--fetch-only", "--limit", "2", "--yes"], h)
    assert code == 0, text
    assert h["fetch_session"].calls and not lock.exists()


def test_dry_run_fetch_only_writes_nothing(raw):
    h = hooks(fetch_session=NoSession())
    code, text = call(["llm-label", "--dry-run", "--fetch-only"], h)
    assert code == 0 and "would fetch" in text
    assert "wrote" not in text
    assert not (config.llm_dir() / "items.jsonl").exists()


def test_claude_diagnose_guards(raw):
    code, text = call(["llm-label", "--judges", "gemma", "--claude-diagnose", "--yes", "--no-fetch"], hooks())
    assert code == 3 and "include claude" in text


def test_claude_diagnose_nothing_selected(raw, monkeypatch):
    from brand_study import llm_evidence as ev
    monkeypatch.setattr(ev, "order_items", lambda items, limit=None: [])
    h = hooks()
    code, text = call(["llm-label", "--judges", "claude", "--claude-diagnose", "--yes", "--no-fetch"], h)
    assert code == 0 and "nothing to diagnose" in text and h["claude_runner"].calls == []


def test_analyze_missing_explicit_llm_overrides(raw, tmp_path):
    code, text = call(["analyze", "--runs", "run1", "--llm-overrides", str(tmp_path / "typo.csv")])
    assert code == 3 and "file not found" in text
    code, _ = call(["analyze", "--runs", "run1"])  # missing default path is fine
    assert code == 0


def test_aggregate_only_rejects_limit(raw):
    code, text = call(["llm-label", "--aggregate-only", "--limit", "3", "--yes"], hooks())
    assert code == 3 and "drop --limit" in text


def test_claude_probed_before_confirmation_fetch_and_pulls(raw):
    from test_bs_llm_judges import HELP, drop_lines
    no_tools = FakeClaude(help_text=drop_lines(HELP, "--tools", "--disallowedTools", "--allowedTools"))
    h = hooks(claude_runner=no_tools, fetch_session=NoSession())
    code, text = call(["llm-label", "--yes"], h)
    assert code == 3 and "neither --tools nor --disallowedTools" in text
    assert h["_events"] == [] and h["ollama_api"].session.posts == []
    assert not (config.llm_dir() / "items.jsonl").exists()
    # a working CLI: flags are shown in the plan before the prompt, and the probed judge is reused (one probe)
    probes = []

    class Counting(FakeClaude):
        def __call__(self, cmd, **kw):
            if cmd[1:] == ["--help"]:
                probes.append(1)
            return super().__call__(cmd, **kw)
    h = hooks(claude_runner=Counting())
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--no-fetch"], h, isatty=True,
                      answers=("n",))
    assert code == 0 and "cancelled" in text
    assert text.index('flags: --tools ""') < text.index("cancelled") and h["claude_runner"].calls == []
    code, text = call(["llm-label", "--judges", "claude", "--limit", "1", "--no-fetch", "--yes"], h)
    assert code == 0, text
    assert len(probes) == 2 and len(votes("claude")) == 1  # one probe per invocation


def test_llm_sample_policy_follows_accepted_file(raw, _isolated_llm):
    code, text = call(["llm-label", "--accept", "majority", "--yes"], hooks())
    assert code == 0, text
    assert "# policy=majority" in _isolated_llm.read_text()
    code, text = call(["llm-sample", "--runs", "run1", "--accept", "unanimous"])
    assert code == 3 and "differs from the policy llm-label applied (majority" in text
    assert not (config.results_dir() / "llm_validation_sample.csv").exists()
    code, text = call(["llm-sample", "--runs", "run1"])
    assert code == 0 and "acceptance policy: majority" in text, text
    snap = json.loads((config.llm_dir() / "validation_snapshot.json").read_text())
    assert snap["policy"] == "majority"
    code, text = call(["llm-sample", "--runs", "run1", "--extend", "2", "--accept", "majority"])
    assert code == 3 and "frozen in the snapshot" in text


def test_llm_sample_policy_without_accepted_file(raw, _isolated_llm, tmp_path):
    code, _ = call(["llm-label", "--judges", "claude", "--yes"], hooks())  # incomplete: no accepted file
    assert code == 0 and not _isolated_llm.exists()
    code, text = call(["llm-sample", "--runs", "run1", "--allow-pending"])
    assert "no accepted LLM labels file" in text and "evaluating policy unanimous" in text
    code, text = call(["llm-sample", "--runs", "run1", "--llm-overrides", str(tmp_path / "nope.csv"), "--force"])
    assert code == 3 and "file not found" in text
