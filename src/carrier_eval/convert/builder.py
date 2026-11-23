"""Incremental trace builder shared by the converters."""

from __future__ import annotations

import hashlib
from typing import Any

from carrier_eval.schema import (
    Context,
    ContextFile,
    FileWrite,
    Step,
    StepRef,
    ToolCall,
    TraceTree,
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TraceBuilder:
    def __init__(self) -> None:
        self.contexts: dict[str, Context] = {}
        self.context_files: list[ContextFile] = []
        self.writes: list[FileWrite] = []
        self.call_ref: dict[str, StepRef] = {}
        """Tool-call id -> the assistant step that made the call."""
        self._seq = 0

    def new_context(
        self,
        ctx_id: str,
        parent: str | None = None,
        agent_name: str | None = None,
        spawned_at: StepRef | None = None,
    ) -> Context:
        ctx = Context(ctx_id=ctx_id, parent=parent, agent_name=agent_name, spawned_at=spawned_at)
        self.contexts[ctx_id] = ctx
        return ctx

    def add(self, ctx: Context, role: str, text: str = "", **fields: Any) -> Step:
        step = Step(i=len(ctx.steps), seq=self._seq, role=role, text=text or "", **fields)
        self._seq += 1
        ctx.steps.append(step)
        for tc in step.tool_calls:
            self.call_ref[tc.id] = StepRef(ctx=ctx.ctx_id, i=step.i)
        return step

    def add_tool_call(self, ctx: Context, step: Step, call: ToolCall) -> None:
        """Attach a call to an existing assistant step (logs that split one response)."""
        step.tool_calls.append(call)
        self.call_ref[call.id] = StepRef(ctx=ctx.ctx_id, i=step.i)

    def add_context_file(self, path: str, content: str, loaded_at: StepRef | None) -> None:
        if any(cf.path == path and cf.sha256 == sha256(content) for cf in self.context_files):
            return
        self.context_files.append(
            ContextFile(path=path, content=content, sha256=sha256(content), loaded_at=loaded_at)
        )

    def add_write(self, after: StepRef, path: str, op: str, content: str) -> None:
        self.writes.append(FileWrite(after=after, path=path, op=op, content=content))

    def last_ref(self, ctx: Context) -> StepRef | None:
        return StepRef(ctx=ctx.ctx_id, i=ctx.steps[-1].i) if ctx.steps else None

    def build(self, **fields: Any) -> TraceTree:
        return TraceTree(
            contexts=list(self.contexts.values()),
            context_files=self.context_files,
            workspace_diffs=self.writes,
            **fields,
        )


def file_write_from_call(name: str, args: dict[str, Any]) -> tuple[str, str, str] | None:
    """Recognize file-writing tool calls from Inspect and Claude Code tool sets.

    Returns (path, op, content) or None. Shell commands that write files are not
    parsed; harnesses that need them should log workspace snapshots instead.
    """
    if name in ("write_file", "create_file", "Write"):
        path = args.get("path") or args.get("file_path")
        content = args.get("content") or args.get("file_text") or ""
        return (path, "write", content) if path else None
    if name in ("Edit", "edit_file"):
        path = args.get("file_path") or args.get("path")
        return (path, "edit", args.get("new_string") or args.get("new_str") or "") if path else None
    if name == "MultiEdit":
        path = args.get("file_path")
        edits = args.get("edits") or []
        content = "\n".join(e.get("new_string", "") for e in edits if isinstance(e, dict))
        return (path, "edit", content) if path else None
    if name in ("text_editor", "str_replace_based_edit_tool"):
        path = args.get("path")
        cmd = args.get("command")
        if not path:
            return None
        if cmd == "create":
            return (path, "write", args.get("file_text") or "")
        if cmd in ("str_replace", "insert"):
            return (path, "edit", args.get("new_str") or args.get("insert_text") or "")
    return None
