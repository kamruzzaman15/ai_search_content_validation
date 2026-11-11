# Agent workflow: plan → review → implement → review

`orchestrate.py` drives the Claude Code and Codex CLIs through four roles:

```
Planner ──plan──▶ Plan reviewer      (up to 5 rounds until VERDICT: APPROVED)
                       │
                 approved plan
                       ▼
Implementer ──code──▶ Code reviewer  (up to 5 rounds until VERDICT: APPROVED)
```

- Before it starts, it asks which tool (`claude` or `codex`), thinking effort, and model each role uses.
- Planners and reviewers run read-only. Only the implementer can edit files.
- When a reviewer asks for changes, the planner or implementer can either fix the issue or push back under "Response to review". The reviewer then judges that reasoning in the next round.
- If the plan is still not approved after 5 rounds, it asks whether to implement the latest plan anyway.
- Nothing is committed or pushed. The changes stay in the working folder for you to check.

## Run

From the repository root:

```bash
# Interactive: asks for the task and each role's settings
python3 agent_workflow/orchestrate.py --workdir ai-search-content-copilot

# Task from a file, role settings from JSON (no questions)
python3 agent_workflow/orchestrate.py --task-file my_task.md \
    --workdir ai-search-content-copilot --config agent_workflow/roles.example.json
```

Thinking effort options:

| Tool | Effort levels |
|---|---|
| claude | low, medium, high, xhigh, max |
| codex | minimal, low, medium, high, xhigh |

Other options: `--max-plan-rounds`, `--max-code-rounds`, `--timeout` (seconds per call, default 3600),
`--claude-impl-permission` (`auto` by default), and `--yes` to skip confirmations.

## Output

Every run writes to `agent_workflow/runs/<timestamp>/`, which Git ignores. It contains each prompt and reply, `final_plan.md`, and `result.md`.

## CLI discovery

The script uses `claude` and `codex` from `PATH` if they work. Otherwise it falls back to the copies bundled with the VS Code extensions. To choose a specific binary, set `CLAUDE_BIN` or `CODEX_BIN`.
