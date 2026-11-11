"""Plan -> review -> implement -> review loop driven by the Claude Code and Codex CLIs.

Usage:
    python agent_workflow/orchestrate.py --task-file task.md
    python agent_workflow/orchestrate.py --task "Add an SEO checker module" --workdir ai-search-content-copilot
    python agent_workflow/orchestrate.py --task-file task.md --config agent_workflow/roles.example.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROLES = ["planner", "plan_reviewer", "implementer", "code_reviewer"]
ROLE_LABELS = {
    "planner": "Planner (writes the plan)",
    "plan_reviewer": "Plan reviewer",
    "implementer": "Implementer (writes the code)",
    "code_reviewer": "Code reviewer",
}
EFFORTS = {
    "claude": ["low", "medium", "high", "xhigh", "max"],
    "codex": ["minimal", "low", "medium", "high", "xhigh"],
}
MODEL_HINTS = {
    "claude": "blank = default; or an alias such as opus, sonnet, fable, haiku",
    "codex": "blank = your Codex config default; or a model name",
}
VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVED|CHANGES_REQUESTED)", re.IGNORECASE)
MAX_DIFF_CHARS = 60_000
LIMIT_RE = re.compile(r"usage limit|rate limit|hit your [\w -]{0,20}limit|limit reached|quota|too many requests|\b429\b",
                      re.IGNORECASE)
LIMIT_POLL_SECONDS = 15 * 60
MAX_LIMIT_WAIT_SECONDS = 6 * 3600
MAX_LIMIT_WAITS = 700  # per step: about a week of 15-minute polls, enough for a weekly limit


class UsageLimitError(RuntimeError):
    def __init__(self, message: str, wait_seconds: int):
        super().__init__(message)
        self.wait_seconds = wait_seconds


@dataclass
class RoleConfig:
    tool: str
    effort: str
    model: str = ""


# --- CLI discovery ---------------------------------------------------------

def _works(path: str) -> bool:
    try:
        return subprocess.run([path, "--version"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_binary(tool: str) -> str:
    env = os.environ.get(f"{tool.upper()}_BIN")
    if env:
        return env
    on_path = shutil.which(tool)
    if on_path and _works(on_path):
        return on_path
    # VS Code ships its own copies; the PATH ones can be missing or broken (e.g. old Node for codex).
    patterns = {
        "claude": "~/.vscode*/extensions/anthropic.claude-code-*/resources/native-binary/claude",
        "codex": "~/.vscode*/extensions/openai.chatgpt-*/bin/linux-x86_64/codex",
    }
    for candidate in sorted(glob.glob(os.path.expanduser(patterns[tool])), reverse=True):
        if _works(candidate):
            return candidate
    sys.exit(f"Could not find a working '{tool}' CLI. Set {tool.upper()}_BIN to its path.")


# --- Usage limits ----------------------------------------------------------

def _seconds_until_reset(text: str) -> int:
    """Best-effort read of the reset time from a limit message; falls back to polling."""
    epoch = re.search(r"\|(\d{10})\b", text)
    if epoch:
        return max(60, int(epoch.group(1)) - int(time.time()) + 60)
    rel = re.search(r"(?:try again|resets?) in\s+(?:(\d+)\s*h\w*)?\s*(?:(\d+)\s*m\w*)?", text, re.IGNORECASE)
    if rel and (rel.group(1) or rel.group(2)):
        return int(rel.group(1) or 0) * 3600 + int(rel.group(2) or 0) * 60 + 60
    clock = re.search(r"(?:resets?|try again at)\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.IGNORECASE)
    if clock:
        hour = int(clock.group(1)) % 12 + (12 if clock.group(3).lower() == "pm" else 0)
        now = time.localtime()
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, int(clock.group(2) or 0), 0, 0, 0, -1))
        if target <= time.time():
            target += 24 * 3600
        return int(target - time.time()) + 60
    return LIMIT_POLL_SECONDS


def _check(proc: subprocess.CompletedProcess, tool: str) -> None:
    if proc.returncode == 0:
        return
    # Only the tail: a long agent transcript may mention "quota" or "429" for unrelated reasons.
    text = f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    if LIMIT_RE.search(text):
        # Capped so a misread reset time cannot stall the run; a still-active limit just waits again.
        raise UsageLimitError(f"{tool} usage limit: {text.strip()[-500:]}",
                              min(_seconds_until_reset(text), MAX_LIMIT_WAIT_SECONDS))
    raise RuntimeError(f"{tool} exited {proc.returncode}: {text.strip()[-2000:]}")


# --- Agent calls -----------------------------------------------------------

def _run_claude(binary: str, cfg: RoleConfig, prompt: str, workdir: Path, write: bool,
                timeout: int, impl_permission: str) -> str:
    cmd = [binary, "-p", "--output-format", "text", "--effort", cfg.effort]
    if cfg.model:
        cmd += ["--model", cfg.model]
    if write:
        cmd += ["--permission-mode", impl_permission]
    else:
        # Read-only: anything outside these tools is denied without prompting.
        cmd += ["--permission-mode", "dontAsk", "--allowedTools", "Read,Grep,Glob"]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=workdir, timeout=timeout)
    _check(proc, "claude")
    return proc.stdout.strip()


def _run_codex(binary: str, cfg: RoleConfig, prompt: str, workdir: Path, write: bool,
               timeout: int, out_file: Path) -> str:
    cmd = [binary, "exec", "-C", str(workdir), "--skip-git-repo-check",
           "-s", "workspace-write" if write else "read-only",
           "-c", f'model_reasoning_effort="{cfg.effort}"',
           "-o", str(out_file)]
    if cfg.model:
        cmd += ["-m", cfg.model]
    cmd.append("-")
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    _check(proc, "codex")
    return out_file.read_text().strip() if out_file.exists() else proc.stdout.strip()


class Runner:
    def __init__(self, roles: dict[str, RoleConfig], workdir: Path, run_dir: Path,
                 timeout: int, impl_permission: str):
        self.roles = roles
        self.workdir = workdir
        self.run_dir = run_dir
        self.timeout = timeout
        self.impl_permission = impl_permission
        self.bins = {tool: _find_binary(tool) for tool in {r.tool for r in roles.values()}}
        self.step = 0

    def call(self, role: str, prompt: str, name: str, write: bool = False) -> str:
        cfg = self.roles[role]
        self.step += 1
        stem = f"{self.step:02d}_{name}"
        (self.run_dir / f"{stem}.prompt.md").write_text(prompt)
        print(f"  -> {ROLE_LABELS[role]} [{cfg.tool}, effort={cfg.effort}"
              f"{', model=' + cfg.model if cfg.model else ''}] ...", flush=True)
        started = time.time()
        limit_waits = 0
        while True:
            try:
                if cfg.tool == "claude":
                    out = _run_claude(self.bins["claude"], cfg, prompt, self.workdir, write,
                                      self.timeout, self.impl_permission)
                else:
                    out = _run_codex(self.bins["codex"], cfg, prompt, self.workdir, write,
                                     self.timeout, self.run_dir / f"{stem}.codex_last.md")
                break
            except UsageLimitError as err:
                limit_waits += 1
                if limit_waits > MAX_LIMIT_WAITS:
                    raise
                # Wait for the limit to reset, then rerun the same step. A write step reruns
                # on top of whatever it had already changed, which the prompt tolerates.
                resume = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + err.wait_seconds))
                print(f"     {cfg.tool} hit its usage limit; waiting until about {resume}, then retrying.",
                      flush=True)
                with open(self.run_dir / "usage_limits.log", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stem}: {err} (retry at {resume})\n")
                time.sleep(err.wait_seconds)
        (self.run_dir / f"{stem}.output.md").write_text(out)
        print(f"     done in {time.time() - started:.0f}s -> {stem}.output.md", flush=True)
        return out


# --- Prompts ---------------------------------------------------------------

def _verdict(text: str) -> str:
    found = VERDICT_RE.findall(text)
    # A missing verdict counts as a request for changes, so the loop never silently passes.
    return found[-1].upper() if found else "CHANGES_REQUESTED"


def _plan_prompt(task: str, plan: str = "", review: str = "") -> str:
    if not plan:
        return f"""You are the PLANNER. Read the repository as needed and write a detailed, step-by-step
implementation plan for the task below. Do not modify any files.

Cover: goal, files to create or change, the approach for each step, data and interfaces,
edge cases, how to test or verify the result, and anything out of scope.
Reply with the plan in Markdown only.

## Task
{task}"""
    return f"""You are the PLANNER. A reviewer has commented on your plan. Revise it.
Accept feedback you agree with. Where you disagree, keep your approach and explain why
under a final "## Response to review" section, so the reviewer can reconsider.
Do not modify any files. Reply with the full revised plan in Markdown.

## Task
{task}

## Your previous plan
{plan}

## Reviewer feedback
{review}"""


def _plan_review_prompt(task: str, plan: str, round_no: int, max_rounds: int) -> str:
    return f"""You are the PLAN REVIEWER (round {round_no} of {max_rounds}). Read the repository as needed
and critically review the plan below against the task. Do not modify any files.

Check correctness, completeness, feasibility in this codebase, risks, and test strategy.
If the planner pushed back on earlier feedback, judge their reasoning fairly.
List concrete, numbered issues. Do not request changes for minor style preferences.

End your reply with exactly one line:
VERDICT: APPROVED
or
VERDICT: CHANGES_REQUESTED

## Task
{task}

## Plan
{plan}"""


def _implement_prompt(task: str, plan: str, summary: str = "", review: str = "") -> str:
    if not review:
        return f"""You are the IMPLEMENTER. Implement the approved plan below in this repository.
Follow the existing code style. Run relevant tests or checks if they exist.
Do not commit or push. Do not put secrets in files.
When finished, reply with a short summary: files changed, what you did, how you verified it,
and anything you could not do.

## Task
{task}

## Approved plan
{plan}"""
    return f"""You are the IMPLEMENTER. A code reviewer has commented on your implementation.
Fix the issues you agree with. Where you disagree, leave the code and explain why under
"## Response to review". Do not commit or push.
Reply with a short summary of what you changed and how you verified it.

## Task
{task}

## Approved plan
{plan}

## Your previous summary
{summary}

## Reviewer feedback
{review}"""


def _code_review_prompt(task: str, plan: str, summary: str, diff: str,
                        round_no: int, max_rounds: int) -> str:
    return f"""You are the CODE REVIEWER (round {round_no} of {max_rounds}). Review the implementation
below against the approved plan. Read the changed files in the repository as needed.
Do not modify any files.

Check correctness, bugs, missing pieces from the plan, tests, security (no secrets
committed), and code quality. If the implementer pushed back, judge their reasoning fairly.
List concrete, numbered issues with file and line where possible.

End your reply with exactly one line:
VERDICT: APPROVED
or
VERDICT: CHANGES_REQUESTED

## Task
{task}

## Approved plan
{plan}

## Implementer summary
{summary}

## Changes (git diff against the starting commit)
{diff}"""


# --- Git helpers -----------------------------------------------------------

def _git(workdir: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=workdir, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


def _diff_since(workdir: Path, base: str) -> str:
    diff = _git(workdir, "diff", base) if base else _git(workdir, "diff")
    untracked = _git(workdir, "ls-files", "--others", "--exclude-standard").strip()
    if untracked:
        diff += "\n# New untracked files (read them directly):\n" + untracked
    if not diff.strip():
        return "(no changes detected)"
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated; read the files directly)"
    return diff


# --- Setup -----------------------------------------------------------------

def _ask(question: str, choices: list[str], default: str) -> str:
    while True:
        answer = input(f"  {question} [{'/'.join(choices)}] (default {default}): ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        print(f"    Please choose one of: {', '.join(choices)}")


def _ask_roles() -> dict[str, RoleConfig]:
    print("\nChoose a tool, model and thinking effort for each agent.\n")
    roles = {}
    for i, role in enumerate(ROLES):
        print(ROLE_LABELS[role])
        tool = _ask("Tool", ["claude", "codex"], "claude" if i % 2 == 0 else "codex")
        effort = _ask("Thinking effort", EFFORTS[tool], "high")
        model = input(f"  Model ({MODEL_HINTS[tool]}): ").strip()
        roles[role] = RoleConfig(tool, effort, model)
        print()
    return roles


def _load_roles(path: str) -> dict[str, RoleConfig]:
    data = json.loads(Path(path).read_text())
    roles = {}
    for role in ROLES:
        if role not in data:
            sys.exit(f"Config is missing role '{role}'.")
        cfg = RoleConfig(**data[role])
        if cfg.tool not in EFFORTS or cfg.effort not in EFFORTS[cfg.tool]:
            sys.exit(f"Invalid tool/effort for '{role}': {cfg.tool}/{cfg.effort}")
        roles[role] = cfg
    return roles


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


# --- Main loop -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", help="Task description text")
    parser.add_argument("--task-file", help="File containing the task description")
    parser.add_argument("--workdir", default=".", help="Repository folder the agents work in")
    parser.add_argument("--config", help="JSON file with role settings (skips the questions)")
    parser.add_argument("--max-plan-rounds", type=int, default=5)
    parser.add_argument("--max-code-rounds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per agent call")
    parser.add_argument("--claude-impl-permission", default="auto",
                        choices=["auto", "acceptEdits", "bypassPermissions"],
                        help="Permission mode for a Claude implementer")
    parser.add_argument("--plan-file", help="Skip planning and implement this already-approved plan")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    if args.task_file:
        task = Path(args.task_file).read_text().strip()
    elif args.task:
        task = args.task.strip()
    else:
        print("Describe the task (finish with an empty line):")
        task = "\n".join(iter(input, "")).strip()
    if not task:
        sys.exit("No task given.")

    workdir = Path(args.workdir).resolve()
    roles = _load_roles(args.config) if args.config else _ask_roles()

    print("Agent setup:")
    for role in ROLES:
        cfg = roles[role]
        print(f"  {ROLE_LABELS[role]:32} {cfg.tool:7} effort={cfg.effort:8} model={cfg.model or 'default'}")
    print(f"  Working folder: {workdir}")
    if _git(workdir, "status", "--porcelain").strip():
        print("  Note: the working folder already has uncommitted changes; reviewers will see them too.")
    if not _confirm("Start?", args.yes):
        sys.exit("Cancelled.")

    run_dir = Path(__file__).resolve().parent / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "task.md").write_text(task)
    (run_dir / "config.json").write_text(json.dumps({r: asdict(c) for r, c in roles.items()}, indent=2))
    runner = Runner(roles, workdir, run_dir, args.timeout, args.claude_impl_permission)
    print(f"\nLogs: {run_dir}\n")

    # Phase 1: plan and review (skipped when resuming from an approved plan).
    if args.plan_file:
        plan = Path(args.plan_file).read_text().strip()
        approved = True
        print(f"=== Phase 1 skipped: using approved plan {args.plan_file} ===")
    else:
        print("=== Phase 1: planning ===")
        plan = runner.call("planner", _plan_prompt(task), "plan_v1")
        approved = False
    for round_no in range(1, 0 if approved else args.max_plan_rounds + 1):
        review = runner.call("plan_reviewer",
                             _plan_review_prompt(task, plan, round_no, args.max_plan_rounds),
                             f"plan_review_r{round_no}")
        if _verdict(review) == "APPROVED":
            approved = True
            print(f"  Plan approved in round {round_no}.")
            break
        print(f"  Reviewer requested changes (round {round_no}).")
        if round_no < args.max_plan_rounds:
            plan = runner.call("planner", _plan_prompt(task, plan, review), f"plan_v{round_no + 1}")
    (run_dir / "final_plan.md").write_text(plan)
    if not approved and not _confirm(
            f"Plan not approved after {args.max_plan_rounds} rounds. Implement the latest plan anyway?", args.yes):
        sys.exit(f"Stopped. Latest plan: {run_dir / 'final_plan.md'}")

    # Phase 2: implement and review.
    print("\n=== Phase 2: implementation ===")
    base = _git(workdir, "rev-parse", "HEAD").strip()
    prompt = _implement_prompt(task, plan)
    if args.plan_file:
        prompt += ("\n\nNote: an earlier run was interrupted while implementing this plan. Some files may already "
                   "exist or be half-written. Inspect the working tree first, then finish and fix the implementation.")
    summary = runner.call("implementer", prompt, "implement_v1", write=True)
    approved = False
    for round_no in range(1, args.max_code_rounds + 1):
        review = runner.call("code_reviewer",
                             _code_review_prompt(task, plan, summary, _diff_since(workdir, base),
                                                 round_no, args.max_code_rounds),
                             f"code_review_r{round_no}")
        if _verdict(review) == "APPROVED":
            approved = True
            print(f"  Code approved in round {round_no}.")
            break
        print(f"  Reviewer requested changes (round {round_no}).")
        if round_no < args.max_code_rounds:
            summary = runner.call("implementer", _implement_prompt(task, plan, summary, review),
                                  f"implement_v{round_no + 1}", write=True)

    status = "APPROVED" if approved else f"NOT APPROVED after {args.max_code_rounds} rounds"
    (run_dir / "result.md").write_text(f"# Result: {status}\n\n## Implementer summary\n{summary}\n")
    print(f"\nFinished: {status}. Changes are uncommitted in {workdir}.")
    print(f"Review them with: git -C {workdir} diff    Logs: {run_dir}")


if __name__ == "__main__":
    main()
