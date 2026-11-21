---
name: agentpace
description: Maintain a coding-agent deliverable plan, answer ETA questions from current task state, record machine timings, and connect vigil completions to plan tasks. Use for ETA to APK, PR, build or another concrete coding deliverable, including mid-run ETA updates.
license: MIT
---

# agentpace

**Use when** the owner asks for an ETA, scope changes, or a run or machine task starts or finishes.
**Produces** a persistent plan and a current commit / P50 / slip P80, with assumptions and owner decisions recorded.

You are the primary caller. The owner sees the answer you relay. Use `--json`
to consume state; relay the short text answer and any material limitations. Give clock times in
the owner's local time zone first, then UTC.
For a deliverable, never give a single number without its median and slip line.
Do not install or publish anything to obtain an estimate.

## Keep one plan per deliverable

Choose an explicit plan identity and location such as
`<project>.research/plans/<deliverable>.json`. Reuse it through the deliverable,
including follow-up review rounds and scope changes. An unrelated plan in the
folder is not evidence that it describes the request.

```sh
agentpace plan init <plan> --from <declared-plan.json> --json
agentpace eta <plan> --json
```

A declared plan has `id`, `cwd`, optional `deliverable`, and `tasks`. Each task
has `id`, `type` (`agent`, `machine`, or `wait`), and `depends_on` for sequencing.
Agent tasks name `model`, `kind`, and optionally `brief`, `human_minutes`,
`effort`, and `verifiable`. Human minutes is an explicit difficulty guess,
not a conversion of agent estimates. Do not invent a historical human guess.
Machine/wait tasks use measured or stated `seconds` plus `basis`, or a recorded
machine `what` key and `host`. Keep queue/remote transfer/review rounds explicit.
The configured agent cap defaults to one; dependencies alone do not authorize
extra agents.

Before dispatch, record a task start. Dispatch, then register the hold with the
estimator's own flags and the id of the run you just started. The task's run P80
is vigil's `--expect-s`; the deliverable commitment cut is a different number
and must not be used as the overdue alarm.

```sh
agentpace plan start <plan> <task> --json
# dispatch (delegate_task, Agent, codex); keep its taskId (delegate_task or t3_thread_send result; a childThreadId also works but is windowed; Agent: agentId; Codex: session id)
vigil hold add <hold> --for none --label <run> \
  $(agentpace estimate-run --model sol --kind implementation --brief <brief> --vigil-meta) \
  --meta run=<taskId> --meta task=<plan-id>/<task-id>
agentpace plan done <plan> <task> --took <measured-seconds> --json
```

`--vigil-meta` prints `--expect-s <P80> --meta p50=<s> --meta p80=<s> --meta est=agentpace-<version>
--meta model=<m> --meta kind=<k>`. `learn` scores the `p50`/`p80` as the run's estimate and finds the
run by `run`, exactly, not by label and time. Without `run` a hold is linked through the
orchestrator's transcript, which cannot see a `$(...)` flag. A hand-typed `--expect-s` with no
`p50`/`p80` is only a ceiling: calibration reports it apart and never as P80 coverage.

Vigil completed ledger items close linked tasks automatically on `eta`, or
explicitly through `plan sync`. A transcript-backed run end can still differ
from the orchestrator's hold-done time; inspect the label-source flag when
judging forecast quality. Do not mark an incomplete or paused task done to
improve the score.

## Answer the remainder, and record changes

```sh
agentpace plan add <plan> <new-task> --model opus --kind review --depends-on <previous> --json
agentpace plan scope <plan> --remove <task> --reason '<actual scope decision>' --json
agentpace eta <plan> --json
```

Done tasks retain their actuals. Running tasks condition on `started_at` and
elapsed time. Todo tasks use fresh distributions. Every issued ETA records
the effective plan snapshot. `extrapolated=true` means the running task has
outlasted empirical support; tell the owner rather than hiding the weak tail.
`what_would_change_it` names uncertainties worth checking before relaying ETA.
The short text command is suitable for the owner; state every assumed
trade-off and its ETA saving, added risk and spending. A miss of `target_s`
remains a miss unless the owner changes scope or authorizes a policy change.

After the deliverable actually lands, close or explicitly remove remaining
tasks, then run `agentpace plan deliver <plan> --json`. This scores every ETA
issued for that deliverable against its actual remaining time.

## Owner decisions, risk and spending

Appetite layers are user default, inferred kind/type, project/deliverable,
plan, and task. Commit follows task cuts through the plan; P50 and P80 are
simulated whole-plan remainders. Inspect each task's resolved policy.
A request above configured appetite returns `needs_owner: true` and keeps the
configured cut. Ask the owner and wait for the answer before applying it.

High-cost accelerators are suggestions only. They include fast/ultra modes,
extra agents beyond the cap, large or additional remote sandboxes and idle sandbox
time. Catalogue `cost` states `class` (`none`, `low`, `high`) and spend such as
`tokens`, `usage_window_pct`, or `sandbox_hours`. Their output includes
`needs_owner: true`, ETA gain and `available_with`. A flag alone is insufficient:
use `--allow-spend <name> --owner-grant '<verbatim owner authorization>'` only
when that authorization was actually given, or use an owner-edited config
grant. Never invent, paraphrase into permission, or self-author a grant.
The quote and timestamp are logged. This is an auditable attestation, not
cryptographic proof of the speaker; local files remain in the owner's trust
boundary. Keep an authorized shortcut explicit in `--apply` as well.

Max reasoning effort is not a speed lever. It is excluded from the time-saving
catalogue. Its observed duration effect is confounded by task selection and
must not be interpreted as causal acceleration.

## Record machine time, not another TIMINGS estimate

```sh
agentpace record start --project <cwd> --what suite:full --host <host-class> --json
agentpace record stop --id <returned-id> --json
agentpace record --project <cwd> --what build:apk --host <host-class> --took <seconds> --json
agentpace timings show --project <cwd> --json
```

Capture `--n-tests`, `--n-files`, `--contention` and `--note` when known.
Records live in `<project>.research/timings.jsonl` with a local index. Import
old notes with `agentpace timings import <TIMINGS.md> --json`; keep originals,
inspect parse diagnostics, and do not pool unknown host/cache conditions as
though they were one measured distribution. Rows needing review are excluded
from automatic machine lookup. Human waits must have a stated basis; do not
derive an unmeasured human part by subtracting agent time from a guess.

## Optional local models

Use only locally fitted earlier labels; no private model or run history ships.
Set AGENTPACE_HOME for a separate store, AGENTPACE_T3_DB for a different read-only
T3 database, and AGENTPACE_VIGIL_LEDGER for the ledger path. Quantiles require a
locally trained model and remain experimental. Use a time-split held-out test
before claiming accuracy. Interpolated support is not historical run counts.

When making an explicit human-minutes guess, carry the literal
`human_minutes: N` into the actual delegated brief before dispatch. A guess
stored only in a plan or CLI argument is not automatically present in T3's run
message and therefore will not become a historical training feature.

## Keep history fresh and review the error

`agentpace learn` ingests runs finished since its last cursor into a local observation store, so
`estimate-run --learned` sees recent history without a full replay. Run it daily or before a batch
of estimates; a second run adds nothing. `agentpace calibration --since 7d` shows where dispatch
estimates missed by model and kind, with pointers to the runs. It is evidence for a human or a
reviewing model to read; do not retune rules from it automatically. A hold is scored only when
learn can tie it to one run: `--meta run=<id>` does that exactly. Without it, register each hold right after
its own dispatch call, not several holds after a batch of dispatches.

## Separate usage-window waits from the work forecast

Read `separate_risks` on run JSON and `tasks[].separate_risks` on ETA JSON.
Relay the usage-reset line separately from commit/P50/slip. The main labels
exclude only evidenced usage-window waits; owner questions, approvals and other
pauses remain in the duration. Do not subtract a quiet gap merely because a
shorter label would improve the score.

Pass `--budget-remaining-pct` on estimate-run, or `budget_remaining_pct` on the
plan task, only when that dispatch snapshot was recorded. Missing history uses
an earlier base rate. Check `severity_identified`: null conditional/expected
wait length means it is unknown. Without positive observations, wait length is unknown. State that rather than
inventing a reset time. Optional wait models must not block the main ETA.
The over_12h risk is per run, not a deliverable-wide probability.

## Research has no reliable duration signal yet

For kind=research, read `reliable: false`, `research_history` and the warning.
State the historical P50/P90, and say that research is low-confidence. No reliable general duration signal has been established.

If an ETA lists `unbounded_tasks`, its totals are illustrative.
`commit_is_actionable: false` means you must not promise that numeric cut to the
owner. State which research remains, keep its historical range visible, and
update the plan as it completes. Done/removed research does not make the
remaining timeline unbounded. Preserve the warning when relaying the ETA;
never silently convert it into a confident deadline. vigil's P80 integer is
still an overdue alarm, not a research commitment.
