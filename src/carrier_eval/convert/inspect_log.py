"""Convert Inspect AI eval logs into trace trees.

Each Inspect agent span becomes a context. A subagent's parent is the nearest
enclosing agent span, and the tool call that started it (the ToolEvent whose
`agent_span_id` names the subagent's span) is its spawn point. The parent's
request becomes the subagent's user turn, because that is what the subagent's
model sees: from inside the subagent, the orchestrator is the user.

Inspect writes the spawning ToolEvent before the subagent's own events. Its
result is only emitted once the subagent's span ends, so `seq` reflects when
the parent actually received the answer.

Sample metadata conventions (set by the rollout harness):
  context_files: [{"path", "content"}]  files the harness loaded into context
  _labels:        generator-side ground truth (see schema.Labels)
  account_id, session_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.event import ModelEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_ai.log import EvalSample, read_eval_log

from carrier_eval.convert.builder import TraceBuilder, file_write_from_call
from carrier_eval.schema import Context, Labels, StepRef, ToolCall, TraceTree

ROOT = "main"


def convert_eval_log(path: str | Path) -> list[TraceTree]:
    log = read_eval_log(str(path), resolve_attachments=True)
    eval_id = log.eval.eval_id
    return [convert_sample(s, eval_id=eval_id) for s in (log.samples or [])]


def convert_sample(sample: EvalSample, eval_id: str = "") -> TraceTree:
    b = TraceBuilder()
    events = sample.events
    spans = {e.id: e for e in events if isinstance(e, SpanBeginEvent)}
    spawner = {
        e.agent_span_id: e
        for e in events
        if isinstance(e, ToolEvent) and getattr(e, "agent_span_id", None)
    }
    ctx_for_span: dict[str, Context] = {}
    deferred: dict[str, tuple[Context, ToolEvent]] = {}
    n_sub = 0

    def agent_of(span_id: str | None) -> str | None:
        while span_id:
            sp = spans.get(span_id)
            if sp is None:
                return None
            if sp.type == "agent":
                return sp.id
            span_id = sp.parent_id
        return None

    def root() -> Context:
        if ROOT not in b.contexts:
            ctx = b.new_context(ROOT)
            _add_sample_input(b, ctx, sample.input)
        return b.contexts[ROOT]

    def ctx_of(span_id: str | None) -> Context:
        agent = agent_of(span_id)
        return ctx_for_span.get(agent) if agent in ctx_for_span else root()

    for ev in events:
        if isinstance(ev, SpanBeginEvent) and ev.type == "agent":
            parent_agent = agent_of(ev.parent_id)
            if parent_agent is None:
                ctx = root()
                ctx.agent_name = ctx.agent_name or ev.name
            else:
                n_sub += 1
                parent = ctx_for_span[parent_agent]
                spawn = spawner.get(ev.id)
                ctx = b.new_context(
                    f"sub-{n_sub}",
                    parent=parent.ctx_id,
                    agent_name=ev.name,
                    spawned_at=b.call_ref.get(spawn.id) if spawn else None,
                )
                if spawn is not None:
                    b.add(ctx, "user", _render_args(spawn.arguments))
            ctx_for_span[ev.id] = ctx

        elif isinstance(ev, ModelEvent):
            ctx = ctx_of(ev.span_id)
            if ctx.system_prompt is None:
                sys_msgs = [m for m in ev.input if m.role == "system"]
                if sys_msgs:
                    ctx.system_prompt = "\n\n".join(m.text for m in sys_msgs)
            msg = ev.output.message
            calls = [
                ToolCall(id=tc.id, name=tc.function, args=dict(tc.arguments or {}))
                for tc in (msg.tool_calls or [])
            ]
            b.add(ctx, "assistant", msg.text, tool_calls=calls)

        elif isinstance(ev, ToolEvent):
            ctx = ctx_of(ev.span_id)
            if getattr(ev, "agent_span_id", None):
                deferred[ev.agent_span_id] = (ctx, ev)
            else:
                _add_tool_result(b, ctx, ev)

        elif isinstance(ev, SpanEndEvent) and ev.id in deferred:
            ctx, tev = deferred.pop(ev.id)
            _add_tool_result(b, ctx, tev)

    for ctx, tev in deferred.values():
        _add_tool_result(b, ctx, tev)
    root()

    meta = sample.metadata or {}
    for cf in meta.get("context_files") or []:
        b.add_context_file(cf["path"], cf.get("content", ""), loaded_at=None)
    labels = meta.get("_labels")
    return b.build(
        trace_id=f"{eval_id}:{sample.id}:{sample.epoch}",
        session_id=str(meta.get("session_id") or f"{eval_id}:{sample.id}:{sample.epoch}"),
        account_id=meta.get("account_id"),
        started_at=str(sample.started_at) if getattr(sample, "started_at", None) else None,
        labels=Labels.model_validate(labels) if labels else None,
        meta={"source": "inspect", "sample_id": str(sample.id), "epoch": sample.epoch},
    )


def _add_sample_input(b: TraceBuilder, ctx: Context, sample_input: Any) -> None:
    if isinstance(sample_input, str):
        b.add(ctx, "user", sample_input)
        return
    for m in sample_input:
        if m.role == "system":
            ctx.system_prompt = (ctx.system_prompt + "\n\n" if ctx.system_prompt else "") + m.text
        else:
            b.add(ctx, m.role, m.text)


def _add_tool_result(b: TraceBuilder, ctx: Context, ev: ToolEvent) -> None:
    error = getattr(ev, "error", None)
    text = f"ERROR: {getattr(error, 'message', error)}" if error else _result_text(ev.result)
    step = b.add(
        ctx, "tool", text, tool_call_id=ev.id, tool_name=ev.function, is_error=bool(error)
    )
    if not error:
        w = file_write_from_call(ev.function, dict(ev.arguments or {}))
        if w:
            b.add_write(StepRef(ctx=ctx.ctx_id, i=step.i), *w)


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(getattr(c, "text", None) or str(c) for c in result)
    return str(result)


def _render_args(args: dict[str, Any] | None) -> str:
    args = dict(args or {})
    if set(args) == {"input"} and isinstance(args["input"], str):
        return args["input"]
    return json.dumps(args, ensure_ascii=False)
