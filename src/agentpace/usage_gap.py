"""Measure between-run waits after a run ends on a provider usage limit.

When a coding-agent provider hits an account usage window limit, the active run usually
terminates and a later run resumes the same work. In-run decomposition (`pause_clock`,
`activity_history`) cannot see those inter-run gaps. This module detects runs that ended
on a usage limit, pairs each one with the run that resumed the same work, classifies gaps
beyond `DEFAULT_MAX_GAP_S` as owner-away rather than a usage wait, and summarizes P50/P80/P90
wait distributions by provider and match strength.

Invariants:
- Read-only boundaries: T3 SQLite is opened with `mode=ro` and `PRAGMA query_only=ON`.
- Privacy: loaders never persist prompts, thread titles, or raw filesystem paths; workspace
  paths are hashed through `local_runs.identity`.
- Determinism: pairing sorts runs chronologically and is invariant to input ordering.
- Honest absence: empty groups report `None` quantiles rather than fabricated zeros.

Mating parts:
- `agentpace.activity_history.LIMIT` and `parse_log` supply authoritative transcript markers.
- `agentpace.estimate.quantile` supplies Hyndman-Fan type 7 quantiles matching `turns.py`.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agentpace import activity_history as ah
from agentpace import estimate as est
from agentpace import local_runs as lr

ProviderKind = Literal["claude", "codex", "other"]
MatchTier = Literal[
    "explicit-continuation",
    "same-t3-thread",
    "same-session",
    "same-project-provider",
]
MatchStrength = Literal["strong", "weak"]
GapClassification = Literal["usage-wait", "owner-away"]

# Matches the 12h boundary of `pause_clock.category`; covers 5h session resets and
# overnight 7-day continuations while separating multi-day absences.
DEFAULT_MAX_GAP_S: float = 43_200.0
DEFAULT_TAIL_TOLERANCE_S: float = 5.0

_WORK_TURN_ITEM_TYPES = frozenset(
    {
        "command_execution",
        "file_change",
        "file_search",
        "dynamic_tool",
        "web_search",
        "subagent",
        "reasoning",
    }
)


@dataclass(frozen=True)
class RunRecord:
    """Normalized run/turn observation at the boundary of usage-gap pairing.

    Raises `ValueError` during construction via `parse_run` if timestamps are non-finite
    or `end < start`.
    """

    run_id: str
    start: float
    end: float | None
    provider: ProviderKind
    raw_provider: str = ""
    thread_id: str | None = None
    session_id: str | None = None
    project: str | None = None
    ordinal: int | None = None
    status: str = "completed"
    source: str = "synthetic"
    usage_limit_end: bool = False
    usage_limit_source: str | None = None
    usage_limit_continuation_of: str | None = None
    made_progress: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UsageGapPair:
    """One paired usage-limit end and the subsequent run that resumed the same work."""

    limit_run_id: str
    resume_run_id: str
    provider: ProviderKind
    limit_end: float
    resume_start: float
    gap_s: float
    last_attempt_gap_s: float
    match_tier: MatchTier
    strength: MatchStrength
    classification: GapClassification
    limit_source: str
    rejected_retries: int = 0
    rejected_retry_ids: tuple[str, ...] = ()
    thread_id: str | None = None
    session_id: str | None = None
    project: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rejected_retry_ids"] = list(self.rejected_retry_ids)
        return data


@dataclass(frozen=True)
class PairingResult:
    """All paired gaps (split into usage-wait vs owner-away) and unresumed limit ends."""

    pairs: tuple[UsageGapPair, ...]
    unresumed_run_ids: tuple[str, ...]
    max_gap_s: float
    include_weak: bool
    collapse_rejected_retries: bool

    @property
    def usage_waits(self) -> tuple[UsageGapPair, ...]:
        return tuple(pair for pair in self.pairs if pair.classification == "usage-wait")

    @property
    def owner_away(self) -> tuple[UsageGapPair, ...]:
        return tuple(pair for pair in self.pairs if pair.classification == "owner-away")

    @property
    def strong_usage_waits(self) -> tuple[UsageGapPair, ...]:
        return tuple(
            pair
            for pair in self.pairs
            if pair.classification == "usage-wait" and pair.strength == "strong"
        )

    @property
    def weak_usage_waits(self) -> tuple[UsageGapPair, ...]:
        return tuple(
            pair
            for pair in self.pairs
            if pair.classification == "usage-wait" and pair.strength == "weak"
        )


@dataclass(frozen=True)
class DistributionStats:
    """Uncapped quantile summary for a group of gap seconds; None when count == 0."""

    count: int
    p50_s: float | None
    p80_s: float | None
    p90_s: float | None
    min_s: float | None
    max_s: float | None
    mean_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GapSummary:
    """Aggregated usage-wait gap statistics overall, by provider, by strength, and by tier."""

    max_gap_s: float
    overall: DistributionStats
    by_provider: dict[ProviderKind, DistributionStats]
    by_strength: dict[MatchStrength, DistributionStats]
    by_tier: dict[MatchTier, DistributionStats]
    owner_away_count: int
    unresumed_count: int
    total_rejected_retries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_gap_s": self.max_gap_s,
            "overall": self.overall.to_dict(),
            "by_provider": {k: v.to_dict() for k, v in self.by_provider.items()},
            "by_strength": {k: v.to_dict() for k, v in self.by_strength.items()},
            "by_tier": {k: v.to_dict() for k, v in self.by_tier.items()},
            "owner_away_count": self.owner_away_count,
            "unresumed_count": self.unresumed_count,
            "total_rejected_retries": self.total_rejected_retries,
        }


def normalize_provider(raw: str | None, model: str | None = None) -> ProviderKind:
    """Normalize provider/model identifiers into Claude, Codex, or other."""
    haystack = f"{raw or ''} {model or ''}".casefold()
    if any(token in haystack for token in ("claude", "anthropic", "opus", "sonnet", "haiku")):
        return "claude"
    if any(token in haystack for token in ("codex", "openai", "gpt", "sol")):
        return "codex"
    return "other"


def is_usage_limit_text(text: str) -> bool:
    """True when authoritative platform error text indicates an account usage-window limit.

    Excludes transient capacity errors ("Selected model is at capacity") and generic 5xx errors.
    """
    # ↳ CLAIM(limit-marker-channels)
    if not text:
        return False
    lowered = text.casefold()
    if not ah.LIMIT.search(text):
        return False
    return any(keyword in lowered for keyword in ("usage", "reset", "quota"))


def run_ended_on_usage_limit(
    run: Mapping[str, Any],
    trace: Mapping[str, Any] | None = None,
    *,
    tail_tolerance_s: float = DEFAULT_TAIL_TOLERANCE_S,
) -> tuple[bool, str | None]:
    """Determine whether a run terminated due to a usage limit (not an in-run recovered wait).

    CLAIM(usage-limit-end-terminal): a transcript usage-limit marker marks a usage-limit END
    only when no subsequent non-limit model progress event occurred before the run ended
    [tests/test_usage_gap.py:test_in_run_usage_wait_with_later_progress_is_not_a_usage_limit_end].
    """
    if tail_tolerance_s < 0 or not math.isfinite(tail_tolerance_s):
        raise ValueError("tail_tolerance_s must be finite and nonnegative")

    if run.get("usage_limit_end"):
        source = run.get("usage_limit_source")
        return True, str(source) if source else "explicit-flag"

    if run.get("failure_class") == "usage_limit" or run.get("failure_code") == "api_error_429":
        return True, "t3-error-item"

    if run.get("usage_limit_notice"):
        return True, "t3-system-notice"

    start_raw = run.get("run_start", run.get("start"))
    end_raw = run.get("run_end", run.get("end"))
    if trace is not None and start_raw is not None and end_raw is not None:
        start = lr.epoch(start_raw)
        end = lr.epoch(end_raw)
        markers = [
            float(m["at"])
            for m in trace.get("markers") or ()
            if m.get("kind") == "usage-limit" and start <= float(m["at"]) <= end + tail_tolerance_s
        ]
        if markers:
            last_marker = max(markers)
            # ↳ CLAIM(usage-limit-end-terminal)
            progress_after = any(
                last_marker < float(t) <= end for t in (trace.get("progress") or ())
            )
            if not progress_after:
                return True, "transcript-limit-marker"
            return False, None

    if int(run.get("usage_limit_markers") or 0) > 0 and not run.get("resumed_in_run"):
        return True, "run-usage-limit-marker"

    return False, None


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_run(raw: RunRecord | Mapping[str, Any]) -> RunRecord:
    """Validate and coerce a mapping or `RunRecord` into a checked `RunRecord`."""
    if isinstance(raw, RunRecord):
        if not math.isfinite(raw.start):
            raise ValueError("run start must be finite")
        if raw.end is not None and (not math.isfinite(raw.end) or raw.end < raw.start):
            raise ValueError("run end must be finite and >= start")
        return raw

    start_val = raw.get("run_start", raw.get("start"))
    if start_val is None:
        raise ValueError("run requires a start timestamp")
    start = lr.epoch(start_val)
    if not math.isfinite(start):
        raise ValueError("run start must be finite")

    end_val = raw.get("run_end", raw.get("end"))
    end: float | None = None
    if end_val is not None:
        end = lr.epoch(end_val)
        if not math.isfinite(end) or end < start:
            raise ValueError("run end must be finite and >= start")

    ended_on_limit, limit_source = run_ended_on_usage_limit(raw)
    raw_provider = str(raw.get("raw_provider") or raw.get("provider") or "")
    provider = normalize_provider(raw_provider, str(raw.get("model") or ""))

    ordinal_raw = raw.get("ordinal")
    ordinal = int(ordinal_raw) if ordinal_raw is not None else None
    made_progress_raw = raw.get("made_progress")
    made_progress = None if made_progress_raw is None else bool(made_progress_raw)

    run_id = _clean_optional_str(raw.get("run_id"))
    if not run_id:
        raise ValueError("run requires a non-empty run_id")

    return RunRecord(
        run_id=run_id,
        start=start,
        end=end,
        provider=provider,
        raw_provider=raw_provider,
        thread_id=_clean_optional_str(raw.get("thread_id")),
        session_id=_clean_optional_str(raw.get("session_id") or raw.get("session")),
        project=_clean_optional_str(raw.get("project")),
        ordinal=ordinal,
        status=str(raw.get("status") or "completed"),
        source=str(raw.get("source") or "synthetic"),
        usage_limit_end=ended_on_limit,
        usage_limit_source=limit_source,
        usage_limit_continuation_of=_clean_optional_str(raw.get("usage_limit_continuation_of")),
        made_progress=made_progress,
    )


def _sort_key(run: RunRecord) -> tuple[float, float, int, str]:
    return (
        run.start,
        run.end if run.end is not None else float("inf"),
        run.ordinal if run.ordinal is not None else 0,
        run.run_id,
    )


def _is_rejected_retry(run: RunRecord, is_limit: bool) -> bool:
    return is_limit and run.made_progress is False


def _follows_chronologically(prev: RunRecord, nxt: RunRecord) -> bool:
    assert prev.end is not None
    if nxt.start < prev.end:
        return False
    if prev.thread_id and prev.thread_id == nxt.thread_id:
        if prev.ordinal is not None and nxt.ordinal is not None:
            if nxt.ordinal < prev.ordinal:
                return False
            if nxt.ordinal == prev.ordinal and nxt.start == prev.end:
                return False
    return True


def pair_usage_gaps(
    runs: Sequence[RunRecord | Mapping[str, Any]],
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    include_weak: bool = True,
    collapse_rejected_retries: bool = True,
) -> PairingResult:
    """Pair each run that ended on a usage limit with the run that resumed the same work.

    Error detail:
        Raises `ValueError` if `max_gap_s` is non-finite or <= 0, or if any run has
        invalid timestamps (`end < start` or non-finite) or a missing `run_id`.

    Pairing rules (in strict precedence order):
        1. `explicit-continuation` (strong): successor run sets
           `usage_limit_continuation_of` to the limit-ended run (or the tail of its
           collapsed rejected-retry chain).
        2. `same-t3-thread` (strong): next chronological run sharing `thread_id`.
        3. `same-session` (strong): next chronological run sharing `session_id`.
        4. `same-project-provider` (weak, only if `include_weak=True`): earliest unclaimed
           subsequent run sharing `(project, provider)`, matched 1-to-1.

    Retry-chain collapsing (`collapse_rejected_retries=True`):
        Intermediate same-work runs that also end on a usage limit with
        `made_progress is False` are absorbed into the outage episode as rejected retries
        so one limit wait is not fragmented into short failed attempts. A subsequent run
        with `made_progress is not False` closes the episode (and opens a new episode if
        it later hits a fresh limit).

    Owner-away cap:
        Gaps `<= max_gap_s` are classified `"usage-wait"`; gaps `> max_gap_s` are
        classified `"owner-away"`.

    Claims:
        CLAIM(gap-definition): gap_s == resume.start - limit.end [test_same_t3_thread_...].
        CLAIM(tier-precedence): explicit-continuation > same-t3-thread > same-session >
            same-project-provider [test_tier_precedence_...].
        CLAIM(owner-away-cap): gap_s <= max_gap_s is usage-wait; > max_gap_s is owner-away
            [test_owner_away_upper_bound_is_inclusive_at_exact_cap].
        CLAIM(retry-chain-collapse): zero-progress limit retries collapse into one episode
            [test_rejected_retry_chain_collapses_into_one_outage_episode].
        CLAIM(one-to-one-weak-claim): each resume run is claimed by at most one weak match
            [test_weak_match_enforces_one_to_one_claiming_for_parallel_workers].
        CLAIM(deterministic-order): output is invariant to input list ordering
            [test_input_order_does_not_affect_pairing_results].
    """
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")

    # ↳ CLAIM(deterministic-order)
    parsed = sorted((parse_run(item) for item in runs), key=_sort_key)
    by_id: dict[str, RunRecord] = {run.run_id: run for run in parsed}

    # Explicit continuation links also certify that their predecessor ended on a usage limit.
    continuation_targets: dict[str, list[RunRecord]] = defaultdict(list)
    for run in parsed:
        if run.usage_limit_continuation_of and run.usage_limit_continuation_of in by_id:
            continuation_targets[run.usage_limit_continuation_of].append(run)

    limit_sources: dict[str, str] = {}
    for run in parsed:
        if run.end is None:
            continue
        if run.usage_limit_end:
            limit_sources[run.run_id] = run.usage_limit_source or "explicit-flag"
        elif run.run_id in continuation_targets:
            limit_sources[run.run_id] = "t3-continuation-link"

    by_thread: dict[str, list[RunRecord]] = defaultdict(list)
    by_session: dict[str, list[RunRecord]] = defaultdict(list)
    by_proj_provider: dict[tuple[str, ProviderKind], list[RunRecord]] = defaultdict(list)
    for run in parsed:
        if run.thread_id:
            by_thread[run.thread_id].append(run)
        if run.session_id:
            by_session[run.session_id].append(run)
        if run.project:
            by_proj_provider[(run.project, run.provider)].append(run)

    def _find_strong_candidate(
        anchor: RunRecord,
        *,
        skip_ids: set[str],
    ) -> tuple[RunRecord | None, MatchTier | None]:
        # ↳ CLAIM(tier-precedence)
        for cand in continuation_targets.get(anchor.run_id, ()):
            if (
                cand.run_id not in skip_ids
                and cand.run_id not in claimed_resume_ids
                and _follows_chronologically(anchor, cand)
            ):
                return cand, "explicit-continuation"
        if anchor.thread_id:
            for cand in by_thread[anchor.thread_id]:
                if (
                    cand.run_id in skip_ids
                    or cand.run_id in claimed_resume_ids
                    or cand.run_id == anchor.run_id
                ):
                    continue
                if _follows_chronologically(anchor, cand):
                    return cand, "same-t3-thread"
        if anchor.session_id:
            for cand in by_session[anchor.session_id]:
                if (
                    cand.run_id in skip_ids
                    or cand.run_id in claimed_resume_ids
                    or cand.run_id == anchor.run_id
                ):
                    continue
                if _follows_chronologically(anchor, cand):
                    return cand, "same-session"
        return None, None

    pairs: list[UsageGapPair] = []
    unresumed: list[str] = []
    absorbed_ids: set[str] = set()
    claimed_resume_ids: set[str] = set()
    weak_pending: list[tuple[RunRecord, RunRecord, tuple[str, ...]]] = []

    # Pass 1: resolve all strong matches first so weak matches never steal a strong resume.
    for run in parsed:
        if run.run_id not in limit_sources or run.run_id in absorbed_ids:
            continue
        assert run.end is not None

        tail = run
        rejected_ids: list[str] = []
        visited: set[str] = {run.run_id}
        matched_resume: RunRecord | None = None
        matched_tier: MatchTier | None = None

        while True:
            cand, tier = _find_strong_candidate(tail, skip_ids=visited)
            if cand is None or tier is None:
                break
            visited.add(cand.run_id)
            cand_is_limit = cand.run_id in limit_sources
            # ↳ CLAIM(retry-chain-collapse)
            if collapse_rejected_retries and _is_rejected_retry(cand, cand_is_limit):
                absorbed_ids.add(cand.run_id)
                rejected_ids.append(cand.run_id)
                tail = cand
                continue
            matched_resume = cand
            matched_tier = tier
            break

        if matched_resume is not None and matched_tier is not None:
            claimed_resume_ids.add(matched_resume.run_id)
            assert tail.end is not None
            # ↳ CLAIM(gap-definition)
            gap_s = matched_resume.start - run.end
            last_gap_s = matched_resume.start - tail.end
            # ↳ CLAIM(owner-away-cap)
            classification: GapClassification = (
                "usage-wait" if gap_s <= max_gap_s else "owner-away"
            )
            pairs.append(
                UsageGapPair(
                    limit_run_id=run.run_id,
                    resume_run_id=matched_resume.run_id,
                    provider=run.provider,
                    limit_end=run.end,
                    resume_start=matched_resume.start,
                    gap_s=gap_s,
                    last_attempt_gap_s=last_gap_s,
                    match_tier=matched_tier,
                    strength="strong",
                    classification=classification,
                    limit_source=limit_sources[run.run_id],
                    rejected_retries=len(rejected_ids),
                    rejected_retry_ids=tuple(rejected_ids),
                    thread_id=run.thread_id,
                    session_id=run.session_id,
                    project=run.project,
                )
            )
        else:
            weak_pending.append((run, tail, tuple(rejected_ids)))

    # Pass 2: resolve weak same-project-provider matches 1-to-1 on remaining episodes.
    for run, tail, rejected_ids in weak_pending:
        assert run.end is not None
        assert tail.end is not None
        matched_weak: RunRecord | None = None
        if include_weak and run.project:
            for cand in by_proj_provider[(run.project, run.provider)]:
                if cand.run_id == run.run_id or cand.run_id in absorbed_ids:
                    continue
                # ↳ CLAIM(one-to-one-weak-claim)
                if cand.run_id in claimed_resume_ids:
                    continue
                if cand.start < tail.end:
                    continue
                # Do not pair weak match onto another zero-progress limit rejection when collapsing.
                if collapse_rejected_retries and _is_rejected_retry(
                    cand, cand.run_id in limit_sources
                ):
                    continue
                matched_weak = cand
                break

        if matched_weak is not None:
            claimed_resume_ids.add(matched_weak.run_id)
            gap_s = matched_weak.start - run.end
            last_gap_s = matched_weak.start - tail.end
            classification = "usage-wait" if gap_s <= max_gap_s else "owner-away"
            pairs.append(
                UsageGapPair(
                    limit_run_id=run.run_id,
                    resume_run_id=matched_weak.run_id,
                    provider=run.provider,
                    limit_end=run.end,
                    resume_start=matched_weak.start,
                    gap_s=gap_s,
                    last_attempt_gap_s=last_gap_s,
                    match_tier="same-project-provider",
                    strength="weak",
                    classification=classification,
                    limit_source=limit_sources[run.run_id],
                    rejected_retries=len(rejected_ids),
                    rejected_retry_ids=rejected_ids,
                    thread_id=run.thread_id,
                    session_id=run.session_id,
                    project=run.project,
                )
            )
        else:
            unresumed.append(run.run_id)

    pairs.sort(key=lambda p: (p.limit_end, p.resume_start, p.limit_run_id, p.resume_run_id))
    return PairingResult(
        pairs=tuple(pairs),
        unresumed_run_ids=tuple(unresumed),
        max_gap_s=max_gap_s,
        include_weak=include_weak,
        collapse_rejected_retries=collapse_rejected_retries,
    )


def _distribution(values: Sequence[float]) -> DistributionStats:
    if not values:
        return DistributionStats(
            count=0,
            p50_s=None,
            p80_s=None,
            p90_s=None,
            min_s=None,
            max_s=None,
            mean_s=None,
        )
    ordered = sorted(float(v) for v in values)
    return DistributionStats(
        count=len(ordered),
        p50_s=est.quantile(ordered, 0.5),
        p80_s=est.quantile(ordered, 0.8),
        p90_s=est.quantile(ordered, 0.9),
        min_s=ordered[0],
        max_s=ordered[-1],
        mean_s=statistics.mean(ordered),
    )


def summarize_gaps(result: PairingResult) -> GapSummary:
    """Summarize usage-wait gaps overall, by provider, by match strength, and by match tier.

    Owner-away gaps (`gap_s > max_gap_s`) are counted in `owner_away_count` and excluded from
    usage-wait quantile distributions.
    """
    waits = result.usage_waits
    providers: tuple[ProviderKind, ...] = ("claude", "codex", "other")
    strengths: tuple[MatchStrength, ...] = ("strong", "weak")
    tiers: tuple[MatchTier, ...] = (
        "explicit-continuation",
        "same-t3-thread",
        "same-session",
        "same-project-provider",
    )
    return GapSummary(
        max_gap_s=result.max_gap_s,
        overall=_distribution([p.gap_s for p in waits]),
        by_provider={
            provider: _distribution([p.gap_s for p in waits if p.provider == provider])
            for provider in providers
        },
        by_strength={
            strength: _distribution([p.gap_s for p in waits if p.strength == strength])
            for strength in strengths
        },
        by_tier={
            tier: _distribution([p.gap_s for p in waits if p.match_tier == tier]) for tier in tiers
        },
        owner_away_count=len(result.owner_away),
        unresumed_count=len(result.unresumed_run_ids),
        total_rejected_retries=sum(p.rejected_retries for p in waits),
    )


_RUN_SINCE = (
    " WHERE r.completed_at IS NOT NULL AND julianday(r.completed_at) >= julianday(?, 'unixepoch')"
)


def _json_object(text: Any) -> dict[str, Any] | None:
    """A JSON object from a database column, or None for a corrupt or non-object value."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def load_t3_runs(db_path: Path | str, *, since: float | None = None) -> list[RunRecord]:
    """Read T3 v2 runs and structured usage-limit signals in strict read-only SQLite mode.

    With `since` (epoch), only runs that finished at or after it are read, and only their turn
    items, provider threads and attempts are scanned (incremental callers such as `learn`).

    CLAIM(t3-readonly-loader): opens SQLite with URI `mode=ro` and `PRAGMA query_only=ON`,
    hashes workspace roots via `local_runs.identity`, and never reads prompt prose
    [tests/test_usage_gap.py:test_load_t3_runs_is_read_only_hashes_project_and_reads_limit_signals].
    """
    # ↳ CLAIM(t3-readonly-loader)
    path = Path(db_path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        run_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT r.run_id, r.thread_id, r.ordinal, r.provider, r.status,
                       r.requested_at, r.completed_at, r.payload_json, p.workspace_root
                FROM orchestration_v2_projection_runs r
                LEFT JOIN orchestration_v2_projection_threads t ON t.thread_id = r.thread_id
                LEFT JOIN projection_projects p ON p.project_id = t.project_id
                """
                + (_RUN_SINCE if since is not None else ""),
                (since,) if since is not None else (),
            )
        ]
        provider_threads = {}
        for provider_thread_id, payload_text in connection.execute(
            "SELECT provider_thread_id, payload_json "
            "FROM orchestration_v2_projection_provider_threads"
            + (
                " WHERE provider_thread_id IN (SELECT json_extract(CASE WHEN json_valid(r.payload_json) THEN r.payload_json END, '$.providerThreadId') "
                "FROM orchestration_v2_projection_runs r" + _RUN_SINCE + ")"
                if since is not None
                else ""
            ),
            (since,) if since is not None else (),
        ):
            loaded = _json_object(payload_text)
            if loaded is not None:
                provider_threads[provider_thread_id] = loaded
        attempt_native: dict[str, str] = {}
        for (payload_text,) in connection.execute(
            "SELECT payload_json FROM orchestration_v2_projection_run_attempts"
            + (
                " WHERE json_extract(CASE WHEN json_valid(payload_json) THEN payload_json END, '$.runId') IN "
                "(SELECT r.run_id FROM orchestration_v2_projection_runs r" + _RUN_SINCE + ")"
                if since is not None
                else ""
            ),
            (since,) if since is not None else (),
        ):
            attempt = _json_object(payload_text)
            if attempt and attempt.get("runId") and attempt.get("nativeThreadId"):
                attempt_native[attempt["runId"]] = str(attempt["nativeThreadId"])

        item_filter = (
            " AND run_id IN (SELECT r.run_id FROM orchestration_v2_projection_runs r" + _RUN_SINCE + ")"
            if since is not None
            else ""
        )
        item_params = (since,) if since is not None else ()
        error_limit_runs: set[str] = set()
        for run_id, payload_text in connection.execute(
            "SELECT run_id, payload_json FROM orchestration_v2_projection_turn_items "
            "WHERE type = 'error' AND status = 'failed'" + item_filter,
            item_params,
        ):
            payload = _json_object(payload_text)
            if payload is None:
                continue
            failure = payload.get("failure") or {}
            if (
                failure.get("class") == "usage_limit"
                or failure.get("code") == "api_error_429"
                or is_usage_limit_text(str(failure.get("message") or ""))
            ):
                error_limit_runs.add(run_id)

        notice_limit_runs: set[str] = set()
        # Filter mirrors orchestration_v2_projection_turn_items_conversation_idx so SQLite
        # uses the partial index instead of scanning ~215k tool/reasoning rows.
        for run_id, payload_text in connection.execute(
            "SELECT run_id, payload_json FROM orchestration_v2_projection_turn_items "
            "WHERE type = 'system_notice' "
            "AND type NOT IN ("
            "'command_execution', 'reasoning', 'file_change', "
            "'file_search', 'web_search', 'dynamic_tool'"
            ")" + item_filter,
            item_params,
        ):
            if "limit" not in str(payload_text).casefold():
                continue
            payload = _json_object(payload_text)
            if payload is None:
                continue
            text = str(payload.get("text") or payload.get("message") or "")
            if is_usage_limit_text(text):
                notice_limit_runs.add(run_id)

        flagged_ids = sorted(error_limit_runs | notice_limit_runs)
        work_counts: dict[str, int] = {}
        if flagged_ids:
            placeholders = ",".join("?" for _ in flagged_ids)
            for run_id, item_type in connection.execute(
                f"SELECT run_id, type FROM orchestration_v2_projection_turn_items "
                f"WHERE run_id IN ({placeholders})",
                flagged_ids,
            ):
                if item_type in _WORK_TURN_ITEM_TYPES:
                    work_counts[run_id] = work_counts.get(run_id, 0) + 1
    finally:
        connection.close()

    records: list[RunRecord] = []
    for row in run_rows:
        payload = _json_object(row["payload_json"])
        if payload is None:
            continue
        start_raw = payload.get("startedAt") or row["requested_at"]
        end_raw = payload.get("completedAt") or row["completed_at"]
        if not start_raw:
            continue
        try:
            start = lr.epoch(start_raw)
            end = lr.epoch(end_raw) if end_raw else None
        except (ValueError, TypeError, AttributeError):
            continue
        if end is not None and end < start:
            continue

        run_id = str(row["run_id"])
        pt = provider_threads.get(payload.get("providerThreadId"), {})
        native_id = attempt_native.get(run_id) or (pt.get("nativeThreadRef") or {}).get("nativeId")
        workspace = row.get("workspace_root") or ""
        project_hash = lr.identity(workspace) if workspace else None
        model = str((payload.get("modelSelection") or {}).get("model") or "")

        is_error_limit = run_id in error_limit_runs
        is_notice_limit = run_id in notice_limit_runs
        ended_on_limit = is_error_limit or is_notice_limit
        limit_source: str | None = None
        if is_error_limit:
            limit_source = "t3-error-item"
        elif is_notice_limit:
            limit_source = "t3-system-notice"

        made_progress: bool | None
        if ended_on_limit:
            made_progress = work_counts.get(run_id, 0) > 0
        else:
            made_progress = True

        records.append(
            RunRecord(
                run_id=run_id,
                start=start,
                end=end,
                provider=normalize_provider(row.get("provider"), model),
                raw_provider=str(row.get("provider") or ""),
                thread_id=_clean_optional_str(row.get("thread_id")),
                session_id=_clean_optional_str(native_id),
                project=project_hash,
                ordinal=int(row["ordinal"]) if row.get("ordinal") is not None else None,
                status=str(row.get("status") or "completed"),
                source="t3-database",
                usage_limit_end=ended_on_limit,
                usage_limit_source=limit_source,
                usage_limit_continuation_of=_clean_optional_str(
                    payload.get("usageLimitContinuationOfRunId")
                ),
                made_progress=made_progress,
            )
        )
    return records


def load_claude_transcript_runs(
    projects_dir: Path | str,
    *,
    exclude_sessions: set[str] | frozenset[str] = frozenset(),
) -> list[RunRecord]:
    """Extract turn-level runs from Claude Code transcripts (main and subagent JSONL files).

    CLAIM(claude-turn-progress): unanswered queued user records and non-limit `<synthetic>`
    stubs with zero real model progress are skipped so they cannot fabricate false resumes
    [tests/test_usage_gap.py:test_load_claude_transcript_runs_ignores_unanswered_queued_user_records;
     tests/test_usage_gap.py:test_claude_synthetic_no_response_turn_does_not_count_as_resume].
    CLAIM(claude-turn-end-execution-only): turn `end` advances only on execution records
    (`assistant` and `tool_result`), never on `queue-operation` or `attachment` records
    [tests/test_usage_gap.py:test_claude_queue_operations_do_not_stretch_limit_turn_end].
    CLAIM(claude-dedupe-compacted-turns): duplicate turns replayed by compaction at the same
    `(start, end, ended_on_limit)` are deduplicated within a session
    [tests/test_usage_gap.py:test_claude_compaction_duplicate_limit_turn_is_not_double_counted].
    """
    root = Path(projects_dir)
    if not root.exists():
        return []

    records: list[RunRecord] = []
    for path in sorted(root.rglob("*.jsonl")):
        session_id = path.stem
        if session_id in exclude_sessions:
            continue

        is_subagent = "subagents" in path.parts
        cwd = ""
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for rec in lr.json_lines(path):
            ts = rec.get("timestamp")
            if ts is None:
                continue
            try:
                at = lr.epoch(ts)
            except (ValueError, TypeError, AttributeError):
                continue

            if not cwd and rec.get("cwd"):
                cwd = str(rec["cwd"])

            typ = rec.get("type")
            msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            content = msg.get("content")

            is_tool_result = typ == "user" and (
                isinstance(content, list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
            )
            is_user_prompt = typ == "user" and not is_tool_result

            if is_user_prompt:
                if current is not None:
                    turns.append(current)
                current = {
                    "start": at,
                    "end": at,
                    "made_progress": False,
                    "limit_at": None,
                    "last_progress_at": None,
                }
                continue

            if current is None:
                continue

            # ↳ CLAIM(claude-turn-end-execution-only)
            if is_tool_result and at > current["end"]:
                current["end"] = at
                continue

            if typ == "assistant":
                if at > current["end"]:
                    current["end"] = at
                is_synthetic_error = bool(rec.get("isApiErrorMessage")) or (
                    msg.get("model") == "<synthetic>"
                )
                text = ah.text_of(rec)
                if is_synthetic_error and is_usage_limit_text(text):
                    current["limit_at"] = at
                elif not is_synthetic_error:
                    current["made_progress"] = True
                    current["last_progress_at"] = at

        if current is not None:
            turns.append(current)

        project_hash = lr.identity(cwd) if cwd else lr.identity(str(path.parent))
        seen_signatures: set[tuple[float, float, bool]] = set()
        turn_index = 0
        for turn in turns:
            limit_at = turn["limit_at"]
            last_progress_at = turn["last_progress_at"]
            ended_on_limit = limit_at is not None and (
                last_progress_at is None or last_progress_at <= limit_at
            )
            # ↳ CLAIM(claude-turn-progress)
            if not turn["made_progress"] and not ended_on_limit:
                continue
            # ↳ CLAIM(claude-dedupe-compacted-turns)
            signature = (float(turn["start"]), float(turn["end"]), ended_on_limit)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            turn_index += 1
            records.append(
                RunRecord(
                    run_id=f"claude:{session_id}:turn:{turn_index}",
                    start=float(turn["start"]),
                    end=float(turn["end"]),
                    provider="claude",
                    raw_provider="claude-subagent" if is_subagent else "claude-code",
                    thread_id=None,
                    session_id=session_id,
                    project=project_hash,
                    ordinal=turn_index,
                    status="failed" if ended_on_limit else "completed",
                    source="claude-subagent" if is_subagent else "claude-transcript",
                    usage_limit_end=ended_on_limit,
                    usage_limit_source="transcript-limit-marker" if ended_on_limit else None,
                    made_progress=bool(turn["made_progress"]),
                )
            )
    return records


def load_codex_transcript_runs(
    sessions_dir: Path | str,
    *,
    exclude_sessions: set[str] | frozenset[str] = frozenset(),
) -> list[RunRecord]:
    """Extract turn-level runs from Codex rollout JSONL transcripts, skipping forks.

    CLAIM(codex-turn-extraction): forked Codex sessions (`forked_from_id`) are skipped to
    prevent duplicating parent usage-limit turns, and `task_complete` / error limit messages
    mark usage-limit ends
    [tests/test_usage_gap.py:test_load_codex_transcript_runs_extracts_limit_turns_and_skips_forks].
    """
    root = Path(sessions_dir)
    if not root.exists():
        return []

    records: list[RunRecord] = []
    for path in sorted(root.rglob("rollout-*.jsonl")):
        session_id = path.stem[-36:] if len(path.stem) >= 36 else path.stem
        cwd = ""
        is_fork = False
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for rec in lr.json_lines(path):
            ts = rec.get("timestamp")
            at: float | None = None
            if ts is not None:
                try:
                    at = lr.epoch(ts)
                except (ValueError, TypeError, AttributeError):
                    at = None

            typ = rec.get("type")
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            sub = payload.get("type")

            if typ == "session_meta":
                # ↳ CLAIM(codex-turn-extraction)
                if payload.get("forked_from_id"):
                    is_fork = True
                    break
                cwd = str(payload.get("cwd") or cwd)
                if payload.get("id"):
                    session_id = str(payload["id"])
                continue

            if typ == "event_msg" and sub == "task_started":
                start_raw = payload.get("started_at") or ts
                if start_raw is None:
                    continue
                try:
                    start_at = lr.epoch(start_raw)
                except (ValueError, TypeError, AttributeError):
                    continue
                current = {
                    "start": start_at,
                    "end": None,
                    "turn_id": payload.get("turn_id"),
                    "made_progress": False,
                    "limit_at": None,
                    "last_progress_at": None,
                    "aborted": False,
                }
                continue

            if current is None:
                continue

            if typ == "event_msg" and sub in ("task_complete", "turn_aborted"):
                if payload.get("turn_id") and current["turn_id"] and payload["turn_id"] != current["turn_id"]:
                    continue
                end_raw = payload.get("completed_at") or ts
                if end_raw is not None:
                    try:
                        end_at = lr.epoch(end_raw)
                    except (ValueError, TypeError, AttributeError):
                        end_at = current["start"]
                else:
                    end_at = current["start"]
                current["end"] = max(float(current["start"]), float(end_at))
                current["aborted"] = sub == "turn_aborted"
                text = ah.text_of(rec) or str(payload.get("last_agent_message") or "")
                if is_usage_limit_text(text):
                    current["limit_at"] = current["end"]
                turns.append(current)
                current = None
                continue

            if at is None:
                continue

            if typ == "event_msg" and sub in ("error", "warning", "rate_limit"):
                text = ah.text_of(rec)
                if is_usage_limit_text(text):
                    current["limit_at"] = at
            elif typ == "response_item" and sub in (
                "reasoning",
                "function_call",
                "custom_tool_call",
                "local_shell_call",
                "message",
            ):
                if payload.get("role") not in ("user", "developer", "system"):
                    current["made_progress"] = True
                    current["last_progress_at"] = at
            elif typ == "event_msg" and sub in ("agent_message", "agent_reasoning"):
                current["made_progress"] = True
                current["last_progress_at"] = at

        if is_fork or session_id in exclude_sessions:
            continue

        project_hash = lr.identity(cwd) if cwd else None
        for idx, turn in enumerate(turns, 1):
            if turn["end"] is None:
                continue
            limit_at = turn["limit_at"]
            last_progress_at = turn["last_progress_at"]
            ended_on_limit = limit_at is not None and (
                last_progress_at is None or last_progress_at <= limit_at
            )
            status = "failed" if ended_on_limit else ("interrupted" if turn["aborted"] else "completed")
            records.append(
                RunRecord(
                    run_id=f"codex:{session_id}:turn:{idx}",
                    start=float(turn["start"]),
                    end=float(turn["end"]),
                    provider="codex",
                    raw_provider="codex",
                    thread_id=None,
                    session_id=session_id,
                    project=project_hash,
                    ordinal=idx,
                    status=status,
                    source="codex-transcript",
                    usage_limit_end=ended_on_limit,
                    usage_limit_source="transcript-limit-marker" if ended_on_limit else None,
                    made_progress=bool(turn["made_progress"]),
                )
            )
    return records


def _format_stats_line(label: str, stats: DistributionStats) -> str:
    if stats.count == 0 or stats.p50_s is None or stats.p80_s is None or stats.p90_s is None:
        return f"{label:<22s} n=0"
    return (
        f"{label:<22s} n={stats.count:<3d} "
        f"P50={stats.p50_s:8.1f}s ({stats.p50_s/3600:4.2f}h)  "
        f"P80={stats.p80_s:8.1f}s ({stats.p80_s/3600:4.2f}h)  "
        f"P90={stats.p90_s:8.1f}s ({stats.p90_s/3600:4.2f}h)"
    )


def cmd_usage_gaps(args: Any) -> int:
    """CLI handler for `agentpace usage-gaps` (read-only; never feeds into estimates)."""
    # ↳ CLAIM(cli-usage-gaps)
    runs: list[RunRecord] = []
    t3_sessions: set[str] = set()

    if not args.no_t3:
        t3_path = Path(args.t3_db) if args.t3_db else Path.home() / ".t3/userdata/statev2.sqlite"
        if args.t3_db and not t3_path.exists():
            raise ValueError(f"T3 database does not exist: {t3_path}")
        if t3_path.exists():
            t3_runs = load_t3_runs(t3_path)
            runs.extend(t3_runs)
            t3_sessions = {r.session_id for r in t3_runs if r.session_id}

    if not args.no_claude:
        claude_dir = Path(args.claude_dir) if args.claude_dir else Path.home() / ".claude/projects"
        if args.claude_dir and not claude_dir.exists():
            raise ValueError(f"Claude projects directory does not exist: {claude_dir}")
        if claude_dir.exists():
            runs.extend(load_claude_transcript_runs(claude_dir, exclude_sessions=t3_sessions))

    if not args.no_codex:
        codex_dir = Path(args.codex_dir) if args.codex_dir else Path.home() / ".codex/sessions"
        if args.codex_dir and not codex_dir.exists():
            raise ValueError(f"Codex sessions directory does not exist: {codex_dir}")
        if codex_dir.exists():
            runs.extend(load_codex_transcript_runs(codex_dir, exclude_sessions=t3_sessions))

    paired = pair_usage_gaps(
        runs,
        max_gap_s=args.max_gap_s,
        include_weak=not args.strong_only,
        collapse_rejected_retries=not args.no_collapse,
    )
    summary = summarize_gaps(paired)

    if args.json:
        payload = {
            "summary": summary.to_dict(),
            "pairs": [p.to_dict() for p in paired.pairs],
            "unresumed_run_ids": list(paired.unresumed_run_ids),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Between-run usage-limit waits (cap <= {paired.max_gap_s:.0f}s):")
    print("  " + _format_stats_line("overall", summary.overall))
    for provider in ("claude", "codex", "other"):
        print("  " + _format_stats_line(f"provider:{provider}", summary.by_provider[provider]))
    for strength in ("strong", "weak"):
        print("  " + _format_stats_line(f"strength:{strength}", summary.by_strength[strength]))
    print(
        f"  owner_away={summary.owner_away_count}  "
        f"unresumed={summary.unresumed_count}  "
        f"collapsed_retries={summary.total_rejected_retries}"
    )
    return 0


def add_command(commands: Any) -> None:
    """Register the `usage-gaps` subcommand on the CLI parser."""
    parser = commands.add_parser(
        "usage-gaps",
        help="measure between-run waits after runs end on a provider usage limit",
    )
    parser.add_argument("--t3-db", help="path to T3 statev2.sqlite (default: ~/.t3/userdata/statev2.sqlite)")
    parser.add_argument("--no-t3", action="store_true", help="skip reading T3 database")
    parser.add_argument("--claude-dir", help="path to Claude projects dir (default: ~/.claude/projects)")
    parser.add_argument("--no-claude", action="store_true", help="skip reading Claude transcripts")
    parser.add_argument("--codex-dir", help="path to Codex sessions dir (default: ~/.codex/sessions)")
    parser.add_argument("--no-codex", action="store_true", help="skip reading Codex transcripts")
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=DEFAULT_MAX_GAP_S,
        help="upper bound in seconds before classifying a gap as owner-away (default: %(default)s)",
    )
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="exclude weak same-project+provider matches",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="pair adjacent limit attempts without collapsing zero-progress rejected retries",
    )
    parser.add_argument("--json", action="store_true", help="print JSON summary and pairs")
    parser.set_defaults(handler=cmd_usage_gaps)
