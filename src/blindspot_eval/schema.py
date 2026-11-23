"""Trace-tree schema shared by converters, verifiers, monitors and results.

A trace is one session: a root context plus every subagent context it spawned,
the context files the harness loaded, and the files written to the workspace.
Every step carries `seq`, its position in global time across all contexts, so
monitors can replay a session in the order things actually happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class StepRef(BaseModel):
    ctx: str
    i: int


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    i: int
    """Index within its context."""
    seq: int
    """Global order across all contexts in the trace."""
    role: Role
    text: str = ""
    kind: str = "message"
    """"message", "meta" (harness-injected user text), or "attachment:<type>"."""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False


class Context(BaseModel):
    ctx_id: str
    parent: str | None = None
    agent_name: str | None = None
    spawned_at: StepRef | None = None
    """The parent step holding the tool call that started this context."""
    system_prompt: str | None = None
    steps: list[Step] = Field(default_factory=list)


class ContextFile(BaseModel):
    path: str
    content: str = ""
    sha256: str | None = None
    loaded_at: StepRef | None = None
    """None means loaded before the first step."""


class FileWrite(BaseModel):
    after: StepRef
    path: str
    op: Literal["write", "edit", "append"] = "write"
    content: str = ""


class Labels(BaseModel):
    """Generator-side ground truth. The sanitizer strips it before any monitor sees a trace."""

    positive: bool | None = None
    """Intended class: a payload case (True) or a matched benign control (False)."""
    verified: bool | None = None
    """The outcome verifier confirmed the payload was released."""
    payload_id: str | None = None
    payload_family: str | None = None
    carrier: str | None = None
    dose: float | None = None
    realized_dose: int | None = None
    E: StepRef | None = None
    E_seq: int | None = None
    R: StepRef | None = None
    R_seq: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class TraceTree(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trace_id: str
    session_id: str
    account_id: str | None = None
    started_at: str | None = None
    context_files: list[ContextFile] = Field(default_factory=list)
    contexts: list[Context]
    workspace_diffs: list[FileWrite] = Field(default_factory=list)
    labels: Labels | None = Field(default=None, alias="_labels")
    meta: dict[str, Any] = Field(default_factory=dict)

    def context(self, ctx_id: str) -> Context:
        for c in self.contexts:
            if c.ctx_id == ctx_id:
                return c
        raise KeyError(ctx_id)

    def root(self) -> Context:
        return next(c for c in self.contexts if c.parent is None)

    def step(self, ref: StepRef) -> Step:
        return self.context(ref.ctx).steps[ref.i]

    def seq_of(self, ref: StepRef) -> int:
        return self.step(ref).seq

    def steps_in_order(self) -> list[tuple[Context, Step]]:
        pairs = [(c, s) for c in self.contexts for s in c.steps]
        return sorted(pairs, key=lambda p: p[1].seq)

    def last_seq(self) -> int:
        return max((s.seq for c in self.contexts for s in c.steps), default=-1)


M = TypeVar("M", bound=BaseModel)


def write_jsonl(records: Iterable[BaseModel], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json(by_alias=True, exclude_none=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, model: type[M] = TraceTree) -> list[M]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(model.model_validate(json.loads(line)))
    return out
