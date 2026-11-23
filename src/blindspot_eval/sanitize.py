"""Strip generator-side information from traces before any monitor sees them.

- Labels and provenance metadata are removed. Labels go to a separate file
  keyed by the new trace id.
- Trace, session, account and tool-call ids are replaced with random tokens.
  One IdMapper is shared across a dataset, so traces that shared an account
  still share one afterwards (account-level arms depend on it).
- Non-conventional file paths are renamed everywhere they appear, because
  generator-chosen names can leak the label (e.g. "fragment_2.txt").
  Conventional names such as CLAUDE.md are kept, since they are realistic.

Context ids ("main", "sub-3") are produced by the converters, carry no
generator information, and are kept so StepRefs in labels stay valid.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import PurePosixPath
from typing import Any, Iterable

from blindspot_eval.schema import Labels, TraceTree

CONVENTIONAL_NAMES = {
    "CLAUDE.md", "AGENTS.md", "GEMINI.md", "README.md", "SKILL.md",
    "copilot-instructions.md", ".cursorrules", "CONTRIBUTING.md",
}
PATH_ARG_KEYS = ("path", "file_path", "filename", "notebook_path")
MIN_BASENAME_LEN = 5
"""Shorter basenames are not replaced in free text, to avoid mangling ordinary words."""


class IdMapper:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.map: dict[str, dict[str, str]] = {}

    def __call__(self, kind: str, value: str | None) -> str | None:
        if value is None:
            return None
        table = self.map.setdefault(kind, {})
        if value not in table:
            table[value] = f"{kind[:1]}{self._rng.getrandbits(48):012x}"
        return table[value]

    def path(self, value: str) -> str:
        table = self.map.setdefault("path", {})
        if value not in table:
            p = PurePosixPath(value.replace("\\", "/"))
            if p.name in CONVENTIONAL_NAMES:
                table[value] = value
            else:
                table[value] = f"file_{self._rng.getrandbits(32):08x}{p.suffix}"
        return table[value]


class LabelRecord(Labels):
    trace_id: str
    account_id: str | None = None


def sanitize(trace: TraceTree, mapper: IdMapper, remap_paths: bool = True) -> tuple[TraceTree, LabelRecord]:
    t = trace.model_copy(deep=True)
    new_id = mapper("trace", t.trace_id)
    record = LabelRecord(
        trace_id=new_id,
        account_id=mapper("account", t.account_id),
        **(t.labels.model_dump() if t.labels else {}),
    )
    t.labels = None
    t.meta = {}
    t.trace_id = new_id
    t.session_id = mapper("session", t.session_id)
    t.account_id = mapper("account", t.account_id)
    t.started_at = None

    for ctx in t.contexts:
        for step in ctx.steps:
            for call in step.tool_calls:
                call.id = mapper("call", call.id)
            if step.tool_call_id:
                step.tool_call_id = mapper("call", step.tool_call_id)

    if remap_paths:
        _remap_paths(t, mapper)
    return t, record


def sanitize_all(
    traces: Iterable[TraceTree], seed: int | None = None, remap_paths: bool = True
) -> tuple[list[TraceTree], list[LabelRecord], IdMapper]:
    mapper = IdMapper(seed)
    clean, records = [], []
    for tr in traces:
        c, r = sanitize(tr, mapper, remap_paths=remap_paths)
        clean.append(c)
        records.append(r)
    return clean, records, mapper


def _collect_paths(t: TraceTree) -> set[str]:
    paths = {cf.path for cf in t.context_files} | {w.path for w in t.workspace_diffs}
    for ctx in t.contexts:
        for step in ctx.steps:
            for call in step.tool_calls:
                for key in PATH_ARG_KEYS:
                    v = call.args.get(key)
                    if isinstance(v, str) and v:
                        paths.add(v)
    return paths


def _remap_paths(t: TraceTree, mapper: IdMapper) -> None:
    replacements: dict[str, str] = {}
    for p in _collect_paths(t):
        new = mapper.path(p)
        if new == p:
            continue
        replacements[p] = new
        base = PurePosixPath(p.replace("\\", "/")).name
        if len(base) >= MIN_BASENAME_LEN and base not in CONVENTIONAL_NAMES:
            replacements.setdefault(base, new)
    if not replacements:
        return
    pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))

    def sub(text: str) -> str:
        return pattern.sub(lambda m: replacements[m.group(0)], text) if text else text

    def sub_any(v: Any) -> Any:
        if isinstance(v, str):
            return sub(v)
        if isinstance(v, list):
            return [sub_any(x) for x in v]
        if isinstance(v, dict):
            return {k: sub_any(x) for k, x in v.items()}
        return v

    for cf in t.context_files:
        cf.path = mapper.path(cf.path)
        cf.content = sub(cf.content)
    for w in t.workspace_diffs:
        w.path = mapper.path(w.path)
        w.content = sub(w.content)
    for ctx in t.contexts:
        if ctx.system_prompt:
            ctx.system_prompt = sub(ctx.system_prompt)
        for step in ctx.steps:
            step.text = sub(step.text)
            for call in step.tool_calls:
                call.args = sub_any(call.args)


def mapping_json(mapper: IdMapper) -> str:
    """The id/path mapping, for generator-side audit only. Never give it to a monitor."""
    return json.dumps(mapper.map, indent=2, sort_keys=True)
