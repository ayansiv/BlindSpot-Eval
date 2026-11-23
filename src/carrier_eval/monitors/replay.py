"""Replay a monitor config over recorded traces and record a score timeline per trace.

Rollouts are generated once. Every arm is then a replay over the same
sanitized traces, so arm comparisons are paired by construction. A timeline
holds the score at each decision point, keyed by `seq`. The results layer
counts a positive as caught only if a flag lands at or before its release
point R.
"""

from __future__ import annotations

import asyncio
import json
from typing import Iterable

from pydantic import BaseModel, Field

from carrier_eval.monitors.config import MonitorConfig
from carrier_eval.monitors.scorers import NotesUpdater, Scorer, make_notes_updater, make_scorer
from carrier_eval.schema import Context, Step, TraceTree

ALL_CONTEXTS = "*"


class ViewUnit(BaseModel):
    seq: float
    ctx: str
    text: str


class ScorePoint(BaseModel):
    seq: int
    score: float


class ScoreTimeline(BaseModel):
    trace_id: str
    arm: str
    repeat: int = 0
    points: list[ScorePoint] = Field(default_factory=list)
    tokens_read: int = 0
    calls: int = 0
    policy_tokens: int = 0


def est_tokens(text: str) -> int:
    return len(text) // 4 + (1 if text else 0)


def _ctx_label(ctx: Context) -> str:
    return ctx.ctx_id if ctx.parent is None else f"{ctx.ctx_id} <- {ctx.parent}"


def render_step(step: Step, label: str, channels: set[str]) -> str:
    if step.kind == "meta" and "meta" not in channels:
        return ""
    if step.kind.startswith("attachment") and "attachments" not in channels:
        return ""
    if step.role == "user":
        return f"[{label}] USER: {step.text}" if "user" in channels and step.text else ""
    if step.role == "assistant":
        parts = []
        if "assistant" in channels and step.text:
            parts.append(f"[{label}] ASSISTANT: {step.text}")
        if "tool_args" in channels:
            parts += [f"[{label}] CALL {c.name} {json.dumps(c.args, ensure_ascii=False)}" for c in step.tool_calls]
        return "\n".join(parts)
    if step.role == "tool":
        return f"[{label}] TOOL RESULT ({step.tool_name}): {step.text}" if "tool_results" in channels else ""
    if step.role == "system":
        return f"[{label}] SYSTEM: {step.text}" if "system" in channels else ""
    return ""


def render_units(trace: TraceTree, channels: Iterable[str]) -> list[ViewUnit]:
    channels = set(channels)
    units: list[ViewUnit] = []
    if "context_files" in channels:
        for cf in trace.context_files:
            seq = trace.seq_of(cf.loaded_at) + 0.5 if cf.loaded_at else -1.0
            units.append(ViewUnit(seq=seq, ctx=ALL_CONTEXTS, text=f"[context file: {cf.path}]\n{cf.content}"))
    for ctx in trace.contexts:
        label = _ctx_label(ctx)
        if "system" in channels and ctx.system_prompt and ctx.steps:
            units.append(ViewUnit(seq=ctx.steps[0].seq - 0.5, ctx=ctx.ctx_id, text=f"[{label}] SYSTEM: {ctx.system_prompt}"))
        for s in ctx.steps:
            text = render_step(s, label, channels)
            if text:
                units.append(ViewUnit(seq=s.seq, ctx=ctx.ctx_id, text=text))
    return sorted(units, key=lambda u: u.seq)


def decision_points(trace: TraceTree, cfg: MonitorConfig) -> list[tuple[Context, Step]]:
    ordered = trace.steps_in_order()
    if not ordered:
        return []
    if cfg.cadence == "end":
        return [ordered[-1]]
    roles = {"assistant"} if cfg.scope == "message" else {"user", "assistant"}
    points = [(c, s) for c, s in ordered if s.role in roles]
    if not points or points[-1][1].seq != ordered[-1][1].seq:
        points.append(ordered[-1])
    return points


class _Stream:
    def __init__(self) -> None:
        self.last_seq = float("-inf")
        self.notes = ""
        self.buffer: list[tuple[float, str]] = []
        self.last_score: float | None = None


class _AccountCarry:
    """Earlier sessions on the same account, as already-seen units."""

    def __init__(self) -> None:
        self.units: list[ViewUnit] = []
        self.stream = _Stream()
        self.sessions = 0


def _join(units: list[ViewUnit]) -> str:
    return "\n".join(u.text for u in units)


async def replay_trace(
    trace: TraceTree,
    cfg: MonitorConfig,
    scorer: Scorer,
    notes: NotesUpdater,
    carry: _AccountCarry | None = None,
    repeat: int = 0,
) -> ScoreTimeline:
    policy = cfg.policy_text()
    units = render_units(trace, cfg.channels)
    tl = ScoreTimeline(trace_id=trace.trace_id, arm=cfg.name, repeat=repeat)
    tl.policy_tokens = sum(
        est_tokens(s.text) + sum(est_tokens(json.dumps(c.args)) for c in s.tool_calls)
        for _, s in trace.steps_in_order()
        if s.role == "assistant"
    )
    streams: dict[str, _Stream] = {}
    last_assistant: dict[str, float] = {}

    for ctx, step in decision_points(trace, cfg):
        if cfg.scope == "message":
            lo = last_assistant.get(ctx.ctx_id, float("-inf"))
            window = [u for u in units if lo < u.seq <= step.seq and u.ctx in (ctx.ctx_id, ALL_CONTEXTS)]
            if step.role == "assistant":
                last_assistant[ctx.ctx_id] = step.seq
            view = _join(window)
            if not view:
                continue
            score = await scorer.score(policy, view)
            _count(tl, policy, view)
            tl.points.append(ScorePoint(seq=step.seq, score=score))
            continue

        if cfg.scope == "context":
            key, visible = ctx.ctx_id, [u for u in units if u.ctx in (ctx.ctx_id, ALL_CONTEXTS)]
        else:
            key, visible = "tree", units
        stream = carry.stream if carry is not None else streams.setdefault(key, _Stream())
        prior = [u for u in visible if u.seq <= stream.last_seq]
        delta = [u for u in visible if stream.last_seq < u.seq <= step.seq]
        if carry is not None:
            prior = carry.units + prior
        stream.last_seq = step.seq
        if not delta and stream.last_score is not None:
            tl.points.append(ScorePoint(seq=step.seq, score=stream.last_score))
            continue

        view = _history_view(cfg, stream, prior, delta)
        if not view:
            continue
        score = await scorer.score(policy, view)
        _count(tl, policy, view)
        tl.points.append(ScorePoint(seq=step.seq, score=score))
        stream.last_score = score
        if cfg.history == "notes":
            stream.notes = await notes.update(stream.notes, _join(delta), cfg.notes_max_chars)
        elif cfg.history == "buffer":
            stream.buffer += [(score, u.text) for u in delta]
            stream.buffer = sorted(stream.buffer, key=lambda x: -x[0])[: cfg.buffer_m]

    if carry is not None:
        carry.sessions += 1
        header = ViewUnit(seq=-1, ctx=ALL_CONTEXTS, text=f"--- earlier session {carry.sessions} ---")
        carry.units += [header] + units
        carry.stream.last_seq = float("-inf")
    return tl


def _history_view(cfg: MonitorConfig, stream: _Stream, prior: list[ViewUnit], delta: list[ViewUnit]) -> str:
    h = cfg.history
    if h == "full":
        return _join(prior + delta)
    if h == "last_tokens":
        return _join(prior + delta)[-cfg.window_tokens * 4 :]
    if h == "delta":
        return _join(delta)
    if h == "notes":
        return f"NOTES SO FAR:\n{stream.notes or '(none)'}\n\nNEW ACTIVITY:\n{_join(delta)}"
    if h == "buffer":
        earlier = "\n".join(t for _, t in stream.buffer)
        return f"EARLIER HIGHEST-SCORING ACTIVITY:\n{earlier or '(none)'}\n\nNEW ACTIVITY:\n{_join(delta)}"
    raise NotImplementedError(f"history={h!r} is planned (design doc A5) but not implemented")


def _count(tl: ScoreTimeline, policy: str, view: str) -> None:
    tl.calls += 1
    tl.tokens_read += est_tokens(policy) + est_tokens(view)


async def replay(
    traces: list[TraceTree],
    cfg: MonitorConfig,
    scorer: Scorer | None = None,
    notes: NotesUpdater | None = None,
    concurrency: int = 8,
    repeat: int = 0,
) -> list[ScoreTimeline]:
    if cfg.history in ("retrieval", "product_memory"):
        raise NotImplementedError(f"history={cfg.history!r} is planned (design doc A5) but not implemented")
    scorer = scorer or make_scorer(cfg.scorer)
    notes = notes or make_notes_updater(cfg.notes_model, cfg.scorer)
    sem = asyncio.Semaphore(concurrency)

    if cfg.scope != "account":
        async def one(t: TraceTree) -> ScoreTimeline:
            async with sem:
                return await replay_trace(t, cfg, scorer, notes, repeat=repeat)

        return list(await asyncio.gather(*(one(t) for t in traces)))

    groups: dict[str, list[TraceTree]] = {}
    for t in traces:
        groups.setdefault(t.account_id or f"solo:{t.trace_id}", []).append(t)

    async def run_group(group: list[TraceTree]) -> list[ScoreTimeline]:
        if all(t.started_at for t in group):
            group = sorted(group, key=lambda t: t.started_at)
        carry = _AccountCarry()
        out = []
        async with sem:
            for t in group:
                out.append(await replay_trace(t, cfg, scorer, notes, carry=carry, repeat=repeat))
        return out

    results = await asyncio.gather(*(run_group(g) for g in groups.values()))
    by_id = {tl.trace_id: tl for group in results for tl in group}
    return [by_id[t.trace_id] for t in traces]


def replay_sync(traces: list[TraceTree], cfg: MonitorConfig, **kw) -> list[ScoreTimeline]:
    return asyncio.run(replay(traces, cfg, **kw))
