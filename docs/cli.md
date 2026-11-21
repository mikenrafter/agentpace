# Command reference

Every `agentpace` command, its flags, its input and output formats, its exit codes and the files it keeps. `agentpace <command> --help` prints the parser's own summary of the same flags. For what the numbers mean and how they are tested, see the [README](../README.md) and [DESIGN](DESIGN.md).

```text
agentpace [--version] <command> [options]
```

| Command | What it does |
|---|---|
| [`ingest`](#ingest) | Read local agent history into rows |
| [`runs`](#runs) | List recent local runs and their references |
| [`label`](#label) | Correct the kind or risk of one run |
| [`estimate`](#estimate) | P50/P80/P90 for a plan of tasks |
| [`score`](#score) | Score a recorded forecast, or show how your forecasts held up |
| [`turns`](#turns) | Measure how long you wait between agent turns, kept local |
| [`backtest`](#backtest) | Evaluate on held-out runs and write the published report |
| [`share`](#share) | Preview your rows, or write a file to contribute |
| [`validate`](#validate) | Check data files |
| [`site`](#site) | Refresh the data inside the static page |
| [`claims`](#claims) | Tie numbers in documents to the data |
| [`learn`](#learn) | Ingest runs finished since the last cursor into a local observation store |
| [`calibration`](#calibration) | Estimates against actuals per model and kind, for a weekly review |

## Output and exit codes

**stdout carries the answer, stderr carries everything else.** Results, listings and the rows `share --dry-run` previews go to stdout, so they can be piped or redirected. Progress reports, notes, warnings about the run itself and error messages go to stderr. Closing the pipe early, as `agentpace share | head` does, stops the command quietly.

| Exit code | Meaning |
|---|---|
| 0 | Success. `ingest --hook` always exits 0, whatever happened. |
| 1 | A check found a problem: `validate` found invalid rows, or `backtest --check`, `site --check` or `claims --check` found published files out of date. |
| 2 | The command refused its input. Either the arguments are wrong (an unknown command, flag or choice; argparse prints the usage), or the input is (a missing path, a plan path that is not a file, an invalid plan, an unknown run reference, or a store whose idle cap does not allow the command). The message is one line on stderr, starting `agentpace:`. |

Any other failure ends with a Python traceback and exit code 1. None is known; please report one in an [issue](https://github.com/mohamedzhioua/agentpace/issues).

## Files: `AGENTPACE_HOME`

agentpace keeps its local files in `~/.agentpace`, or in the folder the `AGENTPACE_HOME` environment variable names. Nothing in it is sent anywhere. Only `share --write` writes rows meant for publication, and it writes them outside this folder.

| File | Written by | Holds |
|---|---|---|
| `secret` | the first command that needs it | A random key, created on first use (owner-only permissions where the system supports them). The `contributor`, `batch` and `project` pseudonyms are keyed by it, so a new secret gives new rows new pseudonyms. Never share it. |
| `runs.jsonl` | `ingest`, `label` | Your rows, one JSON object per line, in the [run schema](../schema/run.schema.json). `estimate` pools them with the seed by default. |
| `runs.jsonl.clock.json` | `ingest` | The idle cap the store was filled with, written before its first rows. Local only, never shared. See [The idle cap](#the-idle-cap). |
| `runs.jsonl.lock` | `ingest`, `label` | Present only while a command is writing the store, so a hook and a manual run never interleave. One older than 15 minutes is taken to be left by a crash. |
| `labels.json` | `label` | Your kind and risk corrections, by run reference. `ingest` re-applies them to fresh rows. |
| `cache-<harness>.json` | `ingest` at the default idle cap | One per harness, such as `cache-claude-code.json`. Parsed runs by transcript path, so a second ingest reads only what changed. It holds transcript paths, so project folder names sit in it raw, as they do in the transcripts. Local only, never shared. |
| `turns.jsonl` | `turns` | Your waits between agent turns. Local only, never shared. |
| `forecasts.jsonl` | `estimate --record`, `score` | The forecast ledger: recorded forecasts and their scores. Local only, never shared. |

`ingest --out` and `turns --out` move their own output file elsewhere. The secret, labels and caches stay in the home folder.

## ingest

```text
agentpace ingest [path] [--source NAME] [--out FILE] [--idle-cap SECONDS] [--kind-map NAME=KIND]... [--hook]
```

Reads a harness's local history and merges the rows into the store. Only timestamps, model ids, token and tool counts and a few flags are copied out of it, never content; DESIGN section 3 lists what each adapter copies.

| Option | Default | Meaning |
|---|---|---|
| `path` | the harness's own history folder | Where to read from. It must exist. |
| `--source` | `claude-code` | One of `claude-code`, `cline`, `codex`, `gemini-cli`, `opencode`. `gemini-cli`, `opencode` and `cline` are experimental: built from each tool's source, not yet verified on real sessions, and `validate` rejects their rows. See [verify-adapter.md](verify-adapter.md). |
| `--out` | `~/.agentpace/runs.jsonl` | The rows file to merge into. |
| `--idle-cap` | `900` | Seconds an idle gap may contribute, inside a run's duration and between runs on the session clock. A non-default cap neither reads nor writes the cache. A rows file keeps one cap; see [The idle cap](#the-idle-cap). |
| `--kind-map NAME=KIND` | none | Map one of your agent types to a task kind. Repeatable. |
| `--hook` | off | Read a Claude Code `SubagentStop` payload on stdin and ingest that session quietly. Prints nothing and always exits 0. Gives up after 5 s if the store is busy. Because it cannot report, run `agentpace ingest` by hand when runs stop appearing: it prints what the hook could not, such as a store line that is not valid JSON. |

stdout: nothing. stderr: a note when the source is experimental, then, when the idle cap is not the default, a warning:

```text
warning: idle cap is 600 s, not the default 900 s. The published dataset uses 900 s, so these rows cannot be shared, and an estimate that pools them with the bundled seed mixes two clocks.
```

then the report (files seen, read from cache, rows produced, empty runs, unreadable files, unparsed lines, zero-duration, still running, and skipped files or runs when there are any), then `store now holds:  N rows`. Last, on an ingest by hand into a store whose idle cap was never recorded:

```text
note: runs.jsonl has no recorded idle cap: it was filled before agentpace kept one. share refuses such a store. To record the cap, rebuild it: https://github.com/mohamedzhioua/agentpace/blob/main/docs/cli.md#the-idle-cap
```

### The idle cap

`--idle-cap` sets how many seconds an idle gap may contribute, inside a run and between runs on the session clock. It changes a run's `duration_s` and `offset_s`, so the same run ingested under two caps would be stored twice. A rows file therefore keeps one cap.

- The first ingest into a new or empty rows file writes `<file>.clock.json` beside it, holding `{"version": 1, "idle_cap_s": N}`. For the default store that is `runs.jsonl.clock.json`. Keep `--out` files outside `data/runs`: `validate` rejects any file there that is not `.jsonl`.
- A later ingest with another `--idle-cap` is refused with exit 2 and ingests nothing. For another cap, pass `--out` with a new file.
- With a cap other than the default, `ingest` warns: the published dataset uses 900 s, so those rows cannot be shared.
- `share` runs only when the default store's cap is known to be 900 s.
- `estimate` prints an `Idle cap:` line under the total whenever your own rows or `--data` take part.

**A store from before 0.13.0** has rows and no record. If every row in it is a published row, it counts as 900 s and the next ingest writes the record. Otherwise its cap is not recorded: `ingest` still adds to it at the default cap and says so, `estimate` still uses it and says so, and `share` refuses it. To record the cap, rebuild the store:

1. Rename `runs.jsonl` in your agentpace folder, for example to `runs-before-0.13.jsonl`. agentpace no longer reads it, and nothing is lost.
2. Run `agentpace ingest`, once for each harness you log (`--source codex`, and so on). It rebuilds `runs.jsonl` at the default cap from the history your harness still has, and writes the record.

Runs whose transcripts the harness has deleted are not rebuilt; they stay in the renamed file, which `estimate --data` can still read. Labels are re-applied by run reference, and a label made on a row from another cap may no longer match its run, so check `agentpace runs` afterwards. Do the same to repair a store that was filled under two caps. The record cannot see a write by an agentpace older than 0.13.0, which ignores it.

## runs

```text
agentpace runs [--last N]
```

Lists local runs in order, ending with the most recent. `--last` sets how many (default `20`).

stdout: one line per run: its reference (`<batch>:<offset_s>`), week, model, kind, risk, size, duration in seconds and outcome. The reference is what `label` and `score --session` take.

**Chronology.** A row carries its ISO week but no clock time, so the order within a week comes from your local ingest caches, never from the row. Runs are sorted by week, then by their session's earliest cached start. A session with no cached start sorts first within its week, so `--last` prefers sessions whose time is known, and one line on stderr says how many of the listed runs' sessions have no local chronology. Sessions have no cached start when their rows came from an `ingest --idle-cap` run (a non-default cap never feeds the cache), when the cache was rebuilt after the transcripts were deleted, or when the secret was replaced. `agentpace ingest` refreshes the cache for transcripts that still exist.

## label

```text
agentpace label REF [--kind KIND] [--risk RISK]
```

Corrects one run's kind or risk, in `labels.json` and in the store. Give `--kind`, `--risk` or both.

| Option | Meaning |
|---|---|
| `REF` | A reference from `agentpace runs`, or `last`. |
| `--kind` | One of `feature`, `bugfix`, `refactor`, `migration`, `tests`, `docs`, `review`, `other`. |
| `--risk` | One of `low`, `normal`, `high`. |

`last` is the latest run of the session with the latest cached start. It refuses, with exit 2, rather than guess: when no session has a cached start, or when a session without one has rows in that session's latest week or later. The message counts the sessions without a cached start that could be the later one; when no session has one, that is all of them. Then run `agentpace ingest`, or give a reference from `agentpace runs`. `label` holds the store's lock while it writes, and says so if another ingest is writing.

stdout: `labeled <ref>: <kind>, <risk> risk`.

## estimate

```text
agentpace estimate (PLAN | - | --example) [--data PATH]... [--method NAME] [--use-size] [--use-risk] [--draws N] [--seed N] [--json] [--turns FILE] [--record] [--quoted TEXT] [--review-capacity TEXT]
```

Simulates a plan of tasks and prints P50, P80 and P90 per task (agent run time) and for the plan (its span on the session clock).

| Option | Default | Meaning |
|---|---|---|
| `PLAN` | none | A plan file in the format below, or `-` to read it from stdin. |
| `--example` | off | Estimate the example plan that ships with agentpace. Not together with a plan. |
| `--data PATH` | the seed data plus `~/.agentpace/runs.jsonl` if it exists | A rows file or folder. Repeatable. When given, only these rows are used. |
| `--method` | `shrunk` | One of `global`, `kind`, `backoff`, `shrunk`, `weighted`. The default is the backtest's pre-registered winner. |
| `--use-size` | off | Use `size`. An oracle track, because stored size is measured after a run. |
| `--use-risk` | off | Look each task's `risk` up as written, as a recorded category (review and research agents `low`, migration agents `high`, all else `normal`), even where no recorded run has that kind at that risk. By default such a task takes its kind's recorded risk; see `risk` under Plan JSON. |
| `--draws` | `20000` | Simulations. At least 100. |
| `--seed` | `20261107` | Random seed. The same plan, data and seed give the same result. |
| `--json` | off | Print the result as JSON instead of text. |
| `--turns FILE` | `~/.agentpace/turns.jsonl` | Waits measured by `agentpace turns`, for the human part line. The file must exist when given. |
| `--record` | off | Keep this forecast in `~/.agentpace/forecasts.jsonl`, to score later. |
| `--quoted TEXT` | none | A timeline someone already gave, such as `"3 weeks"`. Shown beside the range as quoted, and stored with the forecast when `--record` is given. |
| `--review-capacity TEXT` | none | Your own review capacity, such as `"4 PRs a day"`. Shown verbatim on its own line under the human part line, as stated by you: not measured and never applied to the range. Nothing is parsed or computed from it. It must be one line of at most 80 characters once stripped; otherwise the command exits 2. Stored with the forecast when `--record` is given. |

stdout: the result, as text or JSON. stderr, with `--record`: `recorded forecast <id> in <path>`, and with `--quoted`, whether the quote was parsed as calendar time or stored as text.

**Experimental rows.** Without `--data`, the estimate pools the seed data with your local store, so rows you ingested from an experimental adapter (`gemini-cli`, `opencode`, `cline`) take part in it. When any do, one line on stderr says so: `note: N of M rows come from experimental adapters (<names>); they take part in this estimate because they are in your local store. Pass --data to choose the rows.` stdout, the JSON and any recorded forecast are the same with or without it. The seed data holds no experimental rows.

The text form starts with the total, the clock note and the human part:

```text
Plan total on the session clock: P50 …, P80 …, P90 …
The session clock counts agent runs and the gaps between them, each gap up to the idle cap (900 s by default), so waiting on you between runs sits inside the total up to that cap.
Human part (review, testing, decisions, waiting between agent turns): not measured. No allowance for it is added to the total above. State it separately; do not derive it from the agent time.
```

The clock note gives the default cap. When rows other than the bundled seed take part, your local store or `--data`, one more line sits above it, under the total and under the `Quoted:` line when there is one. It names the idle cap each source was clipped at, from the record `ingest` keeps beside a rows file.

```text
Idle cap: 900 s (bundled seed, your local store).
Idle cap: 900 s (bundled seed), 600 s (your local store). This range pools rows on different clocks.
Idle cap: 900 s (bundled seed), not recorded (your local store). The rows with no recorded cap may be on another clock.
Idle cap: 600 s (--data rows).
Idle cap: not recorded (--data rows).
```

A rows file with no record counts as 900 s when every row in it is a published row, and as not recorded otherwise. For one clock, pass `--data` with rows ingested under it. With only the bundled seed, whose rows are at 900 s, the line is absent.

With `--quoted`, one line sits directly under the total. It holds the quote verbatim and is never converted into agent time or compared with the range:

```text
Quoted: "3 weeks" (calendar time, as quoted; unverified, not converted)
```

Then a table of tasks with their P50, P80, P90, how often each was on the critical path and the data cell it drew from, then the critical path, the effective parallelism, the method and the number of runs used, the runs that did not complete by harness, and any warnings. Durations print to two significant figures: seconds under 90 s, minutes under 90 min, hours under 48 h, days beyond. The line after the method counts every loaded run, never per cell: `Not completed, by harness: <harness> N of M (S%; F failed, U unknown), ...`, then says these are recorded harness outcomes, not task acceptance, left out of the fit, and, when there are any, how many completed rows with no duration are left out too.

### Plan JSON

A plan is a JSON object. Keys not listed here are ignored.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `tasks` | list of task objects | required | At least one task. |
| `agent_cap` | integer, 1 or more | `1` | How many agents run at once. |
| `model` | string or `null` | `null` | The model for every task that does not name its own. `null` means no model. |

Each task:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | string or integer | required | Unique within the plan. An integer is read as its string form. |
| `kind` | string | required | One of `feature`, `bugfix`, `refactor`, `migration`, `tests`, `docs`, `review`, `other`. |
| `risk` | string or `null` | the kind's recorded risk: `low` for `review`, `high` for `migration`, else `normal` | One of `low`, `normal`, `high`. In the rows, risk is the agent's role, not a rating of the work. A written value is used when the rows can record that kind at that risk (`review` low, `migration` high, `other` low for research or normal, all else normal); any other value takes the kind's recorded risk and a warning names the task; `--use-risk` looks every written value up as given. |
| `size` | string or `null` | `null` | One of `S`, `M`, `L`, `XL`, `na`. Ignored, with a warning, unless `--use-size` is given. |
| `model` | string or `null` | the plan's `model` | Overrides the plan's model for this task. An explicit `null` means no model for this task; it does not inherit the plan's `model`. |
| `depends_on` | list of task ids | `[]` | Tasks that must finish before this one starts. Each must be in the plan, and the graph must have no cycle. |

The plan holds no durations: every number comes from the data. A task with a model draws from the model × kind × risk cell, and one without from kind × risk. A model name the data has never seen falls back to the broader cell with a wider interval and a warning, so use the model family as it appears in `agentpace runs`, such as `claude-opus-5-5`. A cell with no recorded run gets a warning that names the level that had none (kind, risk or model), which of the task's levels were therefore not used, where the mean and the residuals came from, and what is recorded under the broader cell. A cell with too few runs to stand alone gets one that says its mean was shrunk toward broader cells and names the cell its residuals were pooled from. A written risk the rows cannot record at that kind gets one that names each such task with its risk as written and the risk used.

A plan that breaks these rules is refused with exit 2. A field problem is reported by the task's position and the rule (`task #2: ...`), never by the plan's text. A dependency on an id that is not in the plan is reported with both ids; a duplicate id or a cycle is reported as such. `model` must be a string or `null`, at plan and task level.

```json
{
  "agent_cap": 2,
  "model": "claude-opus-5-5",
  "tasks": [
    { "id": "schema", "kind": "feature" },
    { "id": "research", "kind": "other", "risk": "low" },
    { "id": "api", "kind": "feature", "depends_on": ["schema"] },
    { "id": "docs", "kind": "docs", "model": null, "depends_on": ["api"] }
  ]
}
```

### `--json` output

One JSON object. Every duration is in seconds, as a number.

| Key | Type | Meaning |
|---|---|---|
| `p50`, `p80`, `p90` | number | The plan's total on the session clock: agent runs plus the gaps between them, each gap up to the idle cap. |
| `tasks` | list | One object per task, in plan order: `id` (string), `p50`, `p80`, `p90` (seconds), `cell` (the data cell it drew from, such as `feature x normal x claude-opus-5-5`, or `all`), `n` (runs in that cell, 0 when the cell was never seen), `n_eff` (the cell's effective size), `criticality` (share of simulations in which the task was on the longest chain, 0 to 1). |
| `critical_path` | list of strings | Task ids of the chain that was longest most often. |
| `critical_path_share` | number, 0 to 1 | Share of simulations in which that chain was the longest. |
| `agent_cap` | integer | From the plan. |
| `effective_parallelism` | number or `null` | The median of total work over elapsed span in past multi-run sessions, grouped by peak concurrency and taken from the nearest group at or below the cap, never above the cap. `null` when there are no multi-run sessions to learn from; one at a time is then assumed. |
| `parallelism_sessions` | integer | Sessions `effective_parallelism` was learned from. 0 when it is `null`. |
| `throughput_bound_share` | number, 0 to 1 | Share of simulations in which total work over parallelism, rather than the critical path, set the total. |
| `method` | string | The method used. |
| `size_used` | boolean | Whether `--use-size` was given. |
| `rows_used` | integer | Completed runs the estimate learned from. |
| `rows_excluded` | integer | Runs left out of the fit: those that did not complete (failed or unknown), and completed runs with no duration. |
| `outcomes` | object | Recorded harness outcomes of every row the estimate loaded, after duplicates are dropped, and never per cell: `rows`, `completed`, `failed`, `unknown`, `completed_without_duration` (completed rows with no positive duration), and `by_harness`, a list sorted by harness name of objects with `harness`, `rows`, `completed`, `failed` and `unknown`. An outcome is how the run ended in its harness, not task acceptance, and `unknown` is not counted as a failure. The counts are shown, never used to change the range. `rows` is `completed` + `failed` + `unknown`, `rows_used` is `completed` - `completed_without_duration`, and `rows_excluded` is `failed` + `unknown` + `completed_without_duration`. |
| `draws` | integer | Simulations run. |
| `seed` | integer | The random seed. |
| `warnings` | list of strings | Thin or unseen cells, ignored sizes, a written risk the estimate did not use, an assumed parallelism. Pass them on. |
| `human` | object | The human part. `status` is `not measured` or `measured`, and `included_in_total` is always `false`: no allowance for the human part is added to the total (waits shorter than the idle cap sit inside the session clock already). When measured, it also holds `source`, `per_turn_p50`, `per_turn_p80`, `per_turn_p90` (seconds per human turn), `gaps` and `sessions`. With `--review-capacity`, it holds `review_capacity_stated`, your text as given, stripped. Keys without a value are left out. |
| `idle_cap` | object | Present only when rows other than the bundled seed take part, under the same rule as the `Idle cap:` line, such as `{"seconds": 900, "sources": [{"source": "bundled seed", "seconds": 900}, {"source": "your local store", "seconds": 900}]}`. `seconds` is the one idle cap every source shares, else `null`. `sources` has one entry per source label and cap, in the line's order: `source` is `bundled seed`, `your local store` or `--data rows`, and its `seconds` is `null` when the cap is not recorded. |

The JSON output does not change with `--record` or `--quoted`.

## score

```text
agentpace score [ID] [--session REF | --actual DURATION]
```

Without an id, prints how your recorded forecasts held up: how many are scored, how often the actual fell within each quantile, and, for quoted timelines, how often the quote was closer than the P50. With an id from `estimate --record`, scores that forecast and records the score in the ledger. Give exactly one of:

| Option | Meaning |
|---|---|
| `--session REF` | A session, or any run reference from `runs` (its session is used). The actual is the span of that session's completed local runs on its own clock. It is refused when the session has fewer than 2 completed runs, ran before the forecast was issued, or has no cached start to prove it began after the forecast. |
| `--actual DURATION` | Your own figure, attested by you, such as `"2h"` or `"3 days"`: a number and one of `min`, `h`, `d`, `w`, `mo`, or their long forms. 1 d = 86,400 s, 1 w = 7 d, 1 mo = 30 d. |

stdout: the forecast, the actual and whether it fell within each quantile, plus the quote's ratio to the actual when one was recorded. stderr: a count of damaged ledger lines, if any were skipped. With `--session`, a note when your store's idle cap was never recorded:

```text
note: runs.jsonl has no recorded idle cap, so this session's span may not be on the forecast's clock.
```

or when it was recorded at a cap other than the default:

```text
note: runs.jsonl was filled with a 600 s idle cap, so this session's span is on that clock, which may not be the forecast's.
```

## turns

```text
agentpace turns [path] [--out FILE]
```

Measures how long you wait between agent turns, from the timestamps of your Claude Code main sessions. Kept local, never shared, and never added to an estimate's total.

| Option | Default | Meaning |
|---|---|---|
| `path` | Claude Code's projects folder | Where to read from. It must exist. |
| `--out` | `~/.agentpace/turns.jsonl` | The turns file. |

stdout: the number of gaps and sessions, the wait per human turn at P50, P80 and P90, and the median share of a session spent waiting on you. stderr: the read report.

## backtest

```text
agentpace backtest [--data PATH]... [--out FOLDER] [--seed N] [--check]
```

Runs the published evaluation on held-out runs. Run it from the repository root.

| Option | Default | Meaning |
|---|---|---|
| `--data PATH` | the seed data | A rows file or folder. Repeatable. Your local store is not used unless you pass it. |
| `--out` | `docs` | Where `backtest.md`, `backtest.json` and the two charts under `assets/` go. |
| `--seed` | `20261107` | Random seed. |
| `--check` | off | Write nothing; exit 1 if the results differ from the published `backtest.json`, or `backtest.md` is missing or differs from what they render. |

stdout: what was written and the winning method, or that the numbers match. stderr: the mismatch under `--check`, and a note when the winner is not the shipped default.

## share

```text
agentpace share [--dry-run | --write] [--out-dir FOLDER] [--since YYYY-Www] [--drop-project]
                [--drop-session REF]...
```

Prepares your local rows for a data pull request. It never sends anything.

| Option | Default | Meaning |
|---|---|---|
| `--dry-run` | on | Print the exact rows `--write` would write, and write nothing. |
| `--write` | off | Write them. |
| `--out-dir` | `data/runs` | The dataset folder. Rows already present anywhere in it are left out. |
| `--since YYYY-Www` | none | Only rows from this ISO week onward. |
| `--drop-project` | off | Leave out the `project` field. |
| `--drop-session REF` | none | Leave out a whole session, in the dry run and in the file. A session, or any run reference from `runs` (its session is used). Repeatable. Its rows are left out before anything else, so they count neither as published nor in the file name's weeks. A `REF` whose session has no local row is refused with exit 2, and nothing is written. |

`share` runs only on a store whose idle cap is known to be 900 s (see [The idle cap](#the-idle-cap)). In both modes, before it reads `--out-dir` or applies any option, it refuses with exit 2, prints nothing on stdout and writes nothing when the store was filled with another cap:

```text
agentpace: runs.jsonl was filled with a 600 s idle cap, and the published dataset uses 900 s, so its rows cannot be shared. Nothing was written. To contribute, rename it and run agentpace ingest at the default cap
```

or when its cap was never recorded:

```text
agentpace: runs.jsonl has no recorded idle cap, so share cannot tell whether its rows are on the published 900 s clock. Nothing was written. Rebuild the store first: https://github.com/mohamedzhioua/agentpace/blob/main/docs/cli.md#the-idle-cap
```

`--write` adds or extends one file, `<contributor>-<first week>-<last week>.jsonl`, in `--out-dir`. If that file exists, the new rows are appended and its existing bytes are left as they are. A file holding another contributor's rows is refused with exit 2. The dry run says which of the two `--write` would do.

stdout, in a dry run: the rows, one canonical JSON object per line. stderr: what the rows reveal, how many rows the dropped sessions held (`N rows from M dropped sessions left out.`), how many were already published, a warning when any come from an experimental adapter (`validate` rejects those, so a pull request with them stays red until they are reconciled), and after `--write`, what was written and the commands to submit it, starting with `agentpace validate`. See [CONTRIBUTING.md](../CONTRIBUTING.md#contribute-your-runs).

## validate

```text
agentpace validate [PATH]...
```

Checks data files: the schema, duplicates across files, implausible values, unknown model names, experimental harnesses and anything shaped like personal data. Findings name a file, a line and a rule, never the offending value.

| Option | Default | Meaning |
|---|---|---|
| `PATH` | the seed data shipped with agentpace | Files or folders. Contributors pass `data/runs`: `agentpace validate data/runs`. |

stdout: one `file:line: rule` line per finding, or `all rows valid`. stderr: `N problems`. Exit 1 when there is any finding.

## site

```text
agentpace site [--data PATH]... [--out FILE] [--check]
```

Refreshes the data and the sentences inside the static page from the rows and `backtest.json` beside it. Run `backtest` first.

| Option | Default | Meaning |
|---|---|---|
| `--data PATH` | the seed data | A rows file or folder. Repeatable. |
| `--out` | `docs/index.html` | The page. |
| `--check` | off | Write nothing; exit 1 if the page is out of date. |

## claims

```text
agentpace claims [FILE]... [--data PATH]... [--report FILE] [--check]
```

Rewrites every number between claim markers (`<!-- claim:name -->...<!-- /claim -->`) from the data and the backtest report. Run it from the repository root, because it estimates `examples/plan.json` for the README's example output and `examples/quote-plan.json`, with `--quoted "3 months"`, for its quoted-timeline example.

| Option | Default | Meaning |
|---|---|---|
| `FILE` | `README.md` and `skills/agentpace/assets/portable-prompt.md` | Documents to update. |
| `--data PATH` | the seed data | A rows file or folder. Repeatable. |
| `--report` | `docs/backtest.json` | The backtest report the numbers come from. |
| `--check` | off | Write nothing; exit 1 if any number is stale. |

## estimate-run

Agent-facing run distribution, used internally by deliverable ETAs and vigil.
Call before dispatch; mid-run pass the measured elapsed seconds.

```text
agentpace estimate-run --model sol --kind implementation --brief task.md --cwd /repo --json
agentpace estimate-run --model sol --kind implementation --elapsed 300 --json
agentpace estimate-run --model sol --kind implementation --expect-s
agentpace estimate-run --model sol --kind implementation --vigil-meta
```

`--expect-s` prints only one integer, run P80 seconds, for vigil's overdue alarm.
`--vigil-meta` prints one line to append to `vigil hold add`:
`--expect-s <P80> --meta p50=<s> --meta p80=<s> --meta est=agentpace-<version> --meta model=<m> --meta kind=<k>`
(model and kind as learn stores them: `gemini-4-neon` is `neon`, `impl` is `implementation`).
A value with whitespace or a glob character is refused (exit 2): the line is expanded unquoted, where quotes do not help.
Dispatch first, then register the hold with that line and `--meta run=<id>`. For a T3 delegate the id is its
`taskId` (what `delegate_task` or `t3_thread_send` returned: exact, one run). The `childRunId` is exact too; the
`childThreadId` also works but names every run on the thread, so the hold's window picks the run (see `learn`).
For a Claude background agent it is the `agentId`; for Codex, the session id:

```sh
vigil hold add "$HOLD" --for none --label sol-fix $(agentpace estimate-run --model sol --kind implementation --vigil-meta) --meta run=<taskId>
```

`learn` scores the `p50`/`p80` the hold carries as the run's estimate and finds the run by `run`, not by
label and time. It cannot be combined with `--json` or `--expect-s`.
`--json` returns P50/P80/P90, method, derived features, machine term, row count
and elapsed time. `--elapsed SECONDS` conditions on survival, not subtraction
from an unconditional median. `support` counts historical survivors;
`extrapolated: true` marks elapsed times beyond empirical support. Tell the
owner when that weakens the estimate.

`--expect SECONDS` supplies an estimate actually issued before the run, for
calibration against the reference class. Never invent one from the answer
being predicted. `--human-minutes` is an explicit human-difficulty guess, not a
renamed agent estimate. `--effort`, `--session`, `--host`, `--machine`, `--data`
and `--timings` supply their corresponding context. `--method` is for declared
experiments; live defaults use the train-selected `~/.agentpace/model-policy.json`.
The rows default to `~/.agentpace/local-runs.jsonl`. No public seed rows enter.

## eta

The primary user is the agent. Answer the owner's ETA to APK/PR from the
persistent deliverable plan, not a new guessed task list on every ask.

```text
agentpace eta /project.research/plans/pilot.json --json
agentpace eta /project.research/plans/pilot.json
```

Tasks have `status` of `todo`, `running`, `done`, or `removed`. Running tasks
need `started_at`; done tasks retain `actual_s` and start/end. Scope events keep
added/removed task identities and timestamps. Todo tasks use normal forecasts;
running tasks use conditional remaining distributions; completed work is fixed.
`commit`, `p50`, `p80` are remaining seconds from issue time. `total_p50` and
`total_p80` include elapsed wall time since the plan started; `actual_work_s`
sums known done task durations, which can overlap. JSON includes resolved
per-task appetite, `needs_owner`, tail support, assumed shortcuts,
`available_spend`, configured `agent_cap` and `what_would_change_it`.

Every ETA is logged with a snapshot and timestamp. `eta` automatically syncs
vigil ledger completions first. Older ad-hoc plan files get a companion
`.state.json`; use that file or explicit `plan init` for subsequent state changes.
`--now` supplies an explicit replay timestamp; omit for live work. `--ledger`
overrides the completed-items ledger for replay/testing. `--json` is stable
structured output; the text form is short enough to relay to the owner.

Policy layers are user default, inferred kind/type, configured project,
configured deliverable, plan, task. Settings live in
`~/.config/agentpace/config.json`, with `risk_appetite`, `commit_quantile`,
`agent_cap`, `projects` keyed by absolute cwd, `deliverables` keyed by type,
and an optional owner-edited `allow_spend` list. Conservative/balanced/aggressive
map to P50/P40/P30. A numeric cut must be above zero and at most 0.5. The composed
commit follows task cuts through dependencies; it is not a whole-plan quantile.
Whole-plan P50/P80 use shared residual ranks and list scheduling under the cap.

`--risk-appetite`, `--commit-quantile`, and `--task-risk ID=APPETITE` request
overrides. A cut more aggressive than the configured/inferred layer requires
an owner grant; otherwise JSON marks `needs_owner: true` and retains the baseline.
`--owner-grant '<verbatim owner quote>'` logs the authorization and timestamp.
The agent must not invent a quote or treat a vague reply as permission. Local
quote logging is an auditable attestation, not cryptographic speaker proof.

Catalogue entries declare `id`, `task`, replacement `seconds`, `savings_s`,
`basis`, added `risk`, optional `protects`, and `cost`. A cost object has `class`
(`none`, `low`, `high`) and concrete spend such as `tokens`, `usage_window_pct`
or `sandbox_hours`. Entries without cost default to none for old local plans.
`--apply ID` selects a declared shortcut. `--keep NAME` and `--no-defer TASK`
block corresponding shortcuts; no undeclared work is removed. `target_s`
reports a miss when the unchanged plan cannot fit.

High-cost options stay suggestions unless requested with `--allow-spend NAME`
and authorized by an actual owner quote or owner-edited config. A flag alone
is not authority. JSON marks suggestions `needs_owner: true`, gives their ETA
gain and cost, and text shows `available with --allow-spend NAME`. Fast/ultra
modes, extra parallel agents beyond the cap, large or extra remote sandboxes and idle
sandbox time must be high cost. Extra-agent entries use
`mechanism: extra-agents` and `agent_cap`; they remain suggestable by default.
Max reasoning effort is not a speed lever and is rejected as an accelerator.
Historical effort effects are descriptive, confounded comparisons.

Parallel machine tasks need actual resources; the cap governs agent tasks,
not host scheduling. Model uncertainty, changing scope and unknown machine
contention remain limitations. Relay every applied trade-off and spending
suggestion to the owner. Never silently apply one to make a promised ETA fit.

## plan

Maintain one file per deliverable as work progresses. Every subcommand has
`--json`; `--at` supplies an explicit epoch or ISO timestamp for replay.

```text
agentpace plan init PLAN --from DECLARED_PLAN.json --json
agentpace plan add PLAN review2 --model opus --kind review --depends-on implement --json
agentpace plan start PLAN implement --json
agentpace plan done PLAN implement --took 120 --json
agentpace plan scope PLAN --remove stretch --reason 'Owner deferred to next build' --json
agentpace plan sync PLAN --json
agentpace plan deliver PLAN --json
agentpace plan show PLAN --json
```

`init` accepts `--cwd` without a source file. `add` accepts `--task-json` or task
flags including `--type`, `--seconds`, `--basis`, and repeated `--depends-on`.
`scope --add-json` adds a declared task; removal preserves its identity and
records the timestamp and reason. `start` refuses unfinished dependencies.
`done` needs a prior start or measured `--took`. Completed tasks are idempotent.

Put `--meta task=<plan-id>/<task-id>` on the vigil item. `sync` and `eta` close
linked successful items from `~/.local/state/vigil/items.jsonl`, preserving
ledger times and avoiding duplicate events. Unknown plan/task links are ignored.
`--ledger` on `sync` selects an explicit replay fixture. Agent and machine rows
still retain whether actuals came from a transcript or hold completion.

`deliver` refuses open tasks. Once closed/removed, it records delivery and scores
every issued ETA against actual remaining time, including coverage and pinball
loss. Forecasts retain their issued scope snapshot even if scope later changed.
The outcome is immutable on repeated `deliver` calls.

## record

Record observed machine work instead of writing a new free-text TIMINGS row.
All forms have `--json`.

```text
agentpace record start --project /repo --what suite:full --host laptop --json
agentpace record stop --id RETURNED_ID --json
agentpace record --project /repo --what build:apk --host remote:4cpu --took 120 --json
```

Capture normalized `--what`, `--host` or sandbox class, optional `--n-tests`,
`--n-files`, `--contention`, and `--note`. One-shot `--took` is measured seconds.
Records include start/end and live in `<project>.research/timings.jsonl`, indexed
by `~/.agentpace/timings-index.json`. `--out` overrides the project store.
Unknown timing ids are refused; start/stop state lives locally. Machine lookup
uses uncontended, reviewed project/key/host observations, not estimate prose.

## timings

Import old notes once and inspect parse diagnostics before trusting their
machine classes. Every form has `--json`.

```text
agentpace timings import /repo.research/TIMINGS.md --json
agentpace timings show --project /repo --json
```

`import` accepts `--project` and `--out`, is idempotent by source-line identity,
and leaves the original Markdown untouched. Only observed Took/Actual columns
are imported; ranges, killed jobs, unsupported prose and mixed agent/machine
tasks are flagged in `timings-import.json`. Unknown-host rows need review and
are excluded from automatic machine lookup. `show` groups by project, normalized
key, host, contention and review flag, showing n and P50/P80/P90 seconds.

### Optional models and local configuration

Local-estimator files honor `AGENTPACE_HOME` (default `~/.agentpace`).
`--data`, `--timings`, `--config` and `--ledger` override individual inputs.
`AGENTPACE_T3_DB` defaults to `~/.t3/userdata/statev2.sqlite`; the database is
always opened read-only. Session readers accept root arguments; defaults can
also be set with `AGENTPACE_CODEX_SESSIONS` and `AGENTPACE_CLAUDE_PROJECTS`.
`AGENTPACE_VIGIL_LEDGER` defaults to `~/.local/state/vigil/items.jsonl`.
`AGENTPACE_TIMEZONE` defaults to UTC for diagnostic overnight-gap classification.
Other named zones require a system zoneinfo database or Python tzdata.

No local labels or fitted models ship. Use `experiments/local_pilot.py --pairs
<export.json> --timing <TIMINGS.md> --report <report.json>` to derive local rows;
`--timing` is repeatable and no sibling projects are scanned automatically.
`experiments/deliverable_replay.py --input <private-asks.json> --rows <rows.jsonl>
--out <report.json>` loads replay inputs externally. Each input has asked_at,
actual_s and the pre-ask plan. Results only use earlier complete source sessions.
Private reports and replay inputs must stay outside this repository.

Direct quantiles are experimental: set model_file in model-policy.json to a
locally fitted artifact (default quantile-model.json), and select --method
quantile. Fit via agentpace.quantiles.fit on your earlier derived rows and save
the returned object as JSON. Retain a time-split holdout for evaluation.
The default method is ridge unless a local model-policy.json specifies another
method. Empty local history is an error, not a fallback to an invented ETA.
Interpolated quantile support is not a count of historical surviving runs.

### Wait risks and research reliability

Optional usage-model.json and long-risk.json contain separately fitted local
risk models. Missing or invalid optional models never block the main forecast.
Read separate_risks on run JSON and tasks[].separate_risks on ETA JSON. Unknown
conditional wait durations are null; never infer a zero-hour reset from no
observations. Supply --budget-remaining-pct only for a measured dispatch snapshot.
Only evidenced usage resets are trimmed from main labels; other pauses remain.

Research forecasts return reliable=false and their historical P50/P90. Plans
with remaining research list unbounded_tasks and commit_is_actionable=false;
quantiles remain illustrative and deadline_met is unknown. Relay that warning
to the owner. Done/removed research does not lower remaining-plan confidence.
No general research-duration reliability has been established.

## usage-gaps

```text
agentpace usage-gaps [--t3-db PATH | --no-t3] [--claude-dir PATH | --no-claude] [--codex-dir PATH | --no-codex] [--max-gap-s SECONDS] [--strong-only] [--no-collapse] [--json]
```

Measures between-run waits after a run ends on a provider usage limit and a later
run resumes the same work. Reads T3 `statev2.sqlite` strictly read-only (`mode=ro`,
`PRAGMA query_only=ON`) plus local Claude and Codex transcripts, without feeding
gaps into runtime estimates or persisting prompts or workspace paths.

Pairing precedence: `explicit-continuation` (`usageLimitContinuationOfRunId`) >
`same-t3-thread` > `same-session` (all strong) > `same-project-provider` (weak,
excluded by `--strong-only`). Intermediate zero-progress limit retries collapse
into one episode unless `--no-collapse` is passed. Gaps above `--max-gap-s`
(default `43200`, 12 hours) are classified as `owner-away` and excluded from
usage-wait P50/P80/P90 quantiles.

## learn

```text
agentpace learn [--state DIR] [--t3-db PATH] [--vigil-ledger PATH] [--claude-projects DIR] [--codex-sessions DIR] [--plans DIR]... [--json]
```

Reads runs finished since the last cursor from every source and appends one observation per run to `observations.jsonl` in the state directory (`AGENTPACE_STATE`, else `$XDG_STATE_HOME/agentpace`, else `~/.local/state/agentpace`). Running it twice adds nothing. `cursor.json` holds a finished-time watermark for T3 and the ledger (re-read with a 6 h overlap) and per-file state (`mtime_ns`, size) for transcripts and plans; a file is recorded only after it parsed. The cursor moves only after the rows are written, so a crash loses and doubles nothing. A row whose content changed (a resumed subagent) is appended again and the later line wins. A store written by an older schema is moved aside (`observations.v1.jsonl`) and rebuilt from the sources; so is a cursor whose store has no rows. A subagent resumed by its parent (a coordinator message or a system notification after a finished turn) starts a new span, and its actual is the sum of its spans, never the idle time between them.

| Source | Default location (override) | Read as |
|---|---|---|
| T3 database | `~/.t3/userdata/statev2.sqlite` (`AGENTPACE_T3_DB`) | Completed runs of every provider, Antigravity included, and runs that ended on a usage limit, with the estimator's features (effort, ordinal, concurrency, queue wait). Filtered by the cursor in SQL; opened `mode=ro`. A main-thread (non-delegated) turn is stored as `orchestrator-turn`. |
| vigil item ledger | `$XDG_STATE_HOME/vigil/items.jsonl` (`AGENTPACE_VIGIL_LEDGER`) | Completed items, linked to the run the hold watched through the orchestrator's transcript (the hold registration, the dispatch before it, the run or subagent its result names). A hold with `--meta run=<id>` is linked to that run by id alone (a T3 run id, child thread id, task id or follow-up task id, url-encoded or not; a Claude background-agent id; or a Codex session id) before any transcript, label or time is looked at, and the label never makes it a check-in or machine job. A thread id names every run on the thread: the hold claims the one run that overlaps it and started no earlier than 15 minutes before it, however many runs the thread has; a cancelled, failed or interrupted run that the hold also watched makes it `multi-run`. Any other id is exact: it names one run, and the hold may have opened late (a hold that opened after its run ended is `target-ended-first` and claims nothing). A taskId names the run its `delegate_task` or `t3_thread_send` call started, never a later run T3 started on its own. An id that matches nothing is `link-failed`, not guessed; a hold that found its run but carries no estimate or ceiling is `linked-no-estimate`. A hold carrying the estimator's `--meta p50`/`p80` is scored as that estimate (`estimate_source` `agentpace`); a hand-set `--expect-s` without them is a `ceiling`, kept apart and never scored as a P80. Either is scored against the run only when the hold watched exactly one run: a thread id that spans two is `multi-run`, an exact id is that run. Check-ins and owner waits are `checkin`; holds that name no model are `machine`. |
| Claude subagent transcripts | `~/.claude/projects` (`AGENTPACE_CLAUDE_PROJECTS`) | One observation per subagent transcript; the actual is the sum of its turns, and API-error stubs are not work. A subagent inside a delegated T3 run is linked to it and not a second sample. |
| Codex transcripts | `~/.codex/sessions` (`AGENTPACE_CODEX_SESSIONS`) | One run per completed turn; turns that ended on a usage limit are kept. |
| Plan files | every `*.research/plans` beside the current directory's project (`AGENTPACE_PLAN_DIRS`, `os.pathsep` separated) | Done agent tasks; `forecast_s` is kept as the P50 estimate. |

A missing source is reported as `absent`; a source that fails is reported as `error: <type>` and the others are still read; a bad file or row is skipped and counted. An observation holds the estimate at dispatch when one existed, the actual, model, kind, how the kind and model were labelled (`kind_source`, `model_source`), how the actual was measured (`label_source`), a category (`orchestrator-turn`, `machine`, `checkin`, then `usage-wait`, `over-12h`, `research`, `ordinary`, in that precedence), and pointers (thread id, transcript path, vigil hold and label, plan task). It never holds prompt text. The same run can appear in two sources; readers call `learn.collapse`, which drops a Codex transcript row that belongs to a T3 run, merges a vigil item with the plan task closed from it, moves a hold's estimate onto the one run it watched, and links subagents to their parent turn.

`estimate-run --learned` adds learned agent-run samples that finished after the last row of `--data`, so fresh history reaches an estimate without replaying the whole history. Orchestrator turns, holds and nested subagents are never added.

## calibration

```text
agentpace calibration [--since 7d] [--json | --md] [--out FILE] [--top N] [--state DIR] [--at TIME]
```

Reads the observation store and compares runs finished in the last `--since` window (`7d`, `12h`, `2w`) with everything earlier, per model and kind and per model overall: mean pinball loss of the estimates made at dispatch, coverage of P50/P80/P90 (the share of actuals at or under each estimate; targets 50/80/90), models and kinds with little history, the largest misses with pointers, and drift flags (`p80-undercoverage`, `loss-up`). Usage-wait runs, orchestrator turns, machine jobs and check-ins are counted and not scored, and so are vigil holds that could not be tied to exactly one run, or whose run had ended before the hold started (the report lists them by reason, `target-cancelled` apart from `link-failed`). A group with no estimate shows no loss, not zero. A hand-set ceiling (a `--expect-s` with no estimator output behind it) is never in those figures: the report counts it (`estimates.ceiling`) and its own section says how often runs finished within it (`ceiling` in each group's stats), because a loose alarm that every run meets is not a calibrated P80. `--top` must be at least 1 and `--at` a finite time. The default output is Markdown; `--out` writes it to a file, owner-only (0600).

The report is evidence for a reader. It never changes an estimator, and its flags are thresholds, not rules.
