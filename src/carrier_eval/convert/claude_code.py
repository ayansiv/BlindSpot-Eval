"""Convert native Claude Code session transcripts into trace trees.

Layout under ~/.claude/projects/<project>/:
  <session>.jsonl                                  the main conversation
  <session>/subagents/**/agent-<agentId>.jsonl     one file per subagent

A subagent file is linked to the tool call that spawned it in one of two ways.
Either the tool result carries the `agentId` (the Task/Agent tool), or the
subagent lives under workflows/<runId>/ and the Workflow tool's result carries
that `runId`. Anything unmatched still becomes a child of the main context,
with `spawned_at` left empty.

Records from all files are merged by timestamp so `seq` follows real time.
Claude Code logs each content block of one API response as its own record;
blocks sharing a message id are merged back into one step. Thinking blocks are
dropped: monitors in this eval read actions and messages, not reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carrier_eval.convert.builder import TraceBuilder, file_write_from_call
from carrier_eval.schema import Context, StepRef, ToolCall, TraceTree

ROOT = "main"
CONTEXT_FILE_ATTACHMENTS = {"nested_memory", "file"}
"""Attachments that load file content into context (CLAUDE.md-style memory, @-mentions)."""
TEXT_ATTACHMENTS = {"hook_additional_context", "task_reminder", "skill_listing"}
"""Attachments whose text the model sees; recorded as kind="attachment:<type>" steps."""


def convert_dir(root: str | Path) -> list[TraceTree]:
    """Convert every main session file under a projects directory."""
    root = Path(root)
    mains = [p for p in sorted(root.rglob("*.jsonl")) if "subagents" not in p.parts]
    return [convert_session(p) for p in mains]


def convert_session(path: str | Path, account_id: str | None = None) -> TraceTree:
    path = Path(path)
    sources: list[tuple[str, Path]] = [(ROOT, path)]
    sub_dir = path.with_suffix("") / "subagents"
    agent_files = sorted(sub_dir.rglob("agent-*.jsonl")) if sub_dir.is_dir() else []
    for i, f in enumerate(agent_files, 1):
        sources.append((f"sub-{i}", f))

    merged: list[tuple[str, int, int, str, dict[str, Any]]] = []
    for order, (ctx_id, f) in enumerate(sources):
        for idx, rec in enumerate(_read_records(f)):
            merged.append((rec.get("timestamp") or "", order, idx, ctx_id, rec))
    # Records without a timestamp inherit the previous one within their file.
    last_ts: dict[str, str] = {}
    fixed = []
    for ts, order, idx, ctx_id, rec in merged:
        ts = ts or last_ts.get(ctx_id, "")
        last_ts[ctx_id] = ts
        fixed.append((ts, order, idx, ctx_id, rec))
    fixed.sort(key=lambda r: (r[0], r[1], r[2]))

    b = TraceBuilder()
    main = b.new_context(ROOT)
    for ctx_id, f in sources[1:]:
        b.new_context(ctx_id, parent=ROOT, agent_name=f.stem)

    st = {cid: _CtxState() for cid in b.contexts}
    spawn_links: dict[str, StepRef] = {}
    skipped_sidechain = 0
    session_id = None
    started_at = fixed[0][0] if fixed else None

    for _, _, _, ctx_id, rec in fixed:
        if ctx_id == ROOT and rec.get("isSidechain"):
            skipped_sidechain += 1  # pre-2025 layout kept subagent turns inline
            continue
        session_id = session_id or rec.get("sessionId")
        ctx = b.contexts[ctx_id]
        _apply_record(b, ctx, st[ctx_id], rec, spawn_links)

    for ctx_id, f in sources[1:]:
        agent_id = f.stem.removeprefix("agent-")
        ref = spawn_links.get(agent_id)
        if ref is None and "workflows" in f.parts:
            ref = spawn_links.get(f.parent.name)
        b.contexts[ctx_id].spawned_at = ref

    return b.build(
        trace_id=path.stem,
        session_id=session_id or path.stem,
        account_id=account_id,
        started_at=started_at or None,
        meta={"source": "claude_code", "skipped_sidechain_records": skipped_sidechain},
    )


class _CtxState:
    def __init__(self) -> None:
        self.last_msg_id: str | None = None
        self.assistant_step = None
        self.call_names: dict[str, str] = {}
        self.pending_writes: dict[str, tuple[str, str, str]] = {}


def _apply_record(b: TraceBuilder, ctx: Context, st: _CtxState, rec: dict, links: dict) -> None:
    rtype = rec.get("type")
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}

    if rtype == "assistant":
        mid = msg.get("id") or rec.get("requestId") or rec.get("uuid")
        if mid != st.last_msg_id or st.assistant_step is None:
            st.assistant_step = b.add(ctx, "assistant", "")
            st.last_msg_id = mid
        step = st.assistant_step
        for block in _blocks(msg.get("content")):
            btype = block.get("type")
            if btype == "text":
                step.text = (step.text + "\n" if step.text else "") + block.get("text", "")
            elif btype == "tool_use":
                call = ToolCall(id=block["id"], name=block.get("name", ""), args=block.get("input") or {})
                b.add_tool_call(ctx, step, call)
                st.call_names[call.id] = call.name
                w = file_write_from_call(call.name, call.args)
                if w:
                    st.pending_writes[call.id] = w
        return

    if rtype == "user":
        st.last_msg_id, st.assistant_step = None, None
        content = msg.get("content")
        kind = "meta" if rec.get("isMeta") else "message"
        if isinstance(content, str):
            b.add(ctx, "user", content, kind=kind)
            return
        for block in _blocks(content):
            btype = block.get("type")
            if btype == "tool_result":
                call_id = block.get("tool_use_id", "")
                is_error = bool(block.get("is_error"))
                step = b.add(
                    ctx,
                    "tool",
                    _block_text(block.get("content")),
                    tool_call_id=call_id,
                    tool_name=st.call_names.get(call_id),
                    is_error=is_error,
                )
                w = st.pending_writes.pop(call_id, None)
                if w and not is_error:
                    b.add_write(StepRef(ctx=ctx.ctx_id, i=step.i), *w)
                result = rec.get("toolUseResult")
                spawn_ref = b.call_ref.get(call_id)
                if isinstance(result, dict) and spawn_ref is not None:
                    for key in ("agentId", "runId"):
                        if result.get(key):
                            links[str(result[key])] = spawn_ref
            elif btype == "text":
                b.add(ctx, "user", block.get("text", ""), kind=kind)
            elif btype == "image":
                b.add(ctx, "user", "[image]", kind=kind)
        return

    if rtype == "attachment":
        st.last_msg_id, st.assistant_step = None, None
        att = rec.get("attachment") if isinstance(rec.get("attachment"), dict) else {}
        atype = att.get("type")
        content = att.get("content")
        if atype in CONTEXT_FILE_ATTACHMENTS:
            loaded = _attachment_file(att)
            if loaded:
                b.add_context_file(*loaded, loaded_at=b.last_ref(ctx))
        elif atype in TEXT_ATTACHMENTS and isinstance(content, str) and content:
            b.add(ctx, "user", content, kind=f"attachment:{atype}")


def _attachment_file(att: dict[str, Any]) -> tuple[str, str] | None:
    """(path, content) from a nested_memory or file attachment.

    nested_memory carries content as a string. file nests it one level down,
    at content.file.content, with the path in content.file.filePath.
    """
    content = att.get("content")
    path = att.get("path") or att.get("filename") or att.get("displayPath")
    if isinstance(content, dict) and isinstance(content.get("file"), dict):
        inner = content["file"]
        path = inner.get("filePath") or path
        content = inner.get("content")
    if not isinstance(content, str):
        return None
    return (path or "?", content)


def _read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [c for c in content if isinstance(c, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _block_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or ("[image]" if c.get("type") == "image" else ""))
            else:
                parts.append(str(c))
        return "\n".join(p for p in parts if p)
    return str(content)
