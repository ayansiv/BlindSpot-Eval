from blindspot_eval.convert.claude_code import convert_dir, convert_session
from blindspot_eval.convert.inspect_log import convert_eval_log
from blindspot_eval.schema import TraceTree, read_jsonl, write_jsonl

from conftest import RAW, needs, split_trace


def test_roundtrip_keeps_labels_alias(tmp_path):
    t = split_trace(["alpha", "beta"])
    write_jsonl([t], tmp_path / "t.jsonl")
    raw = (tmp_path / "t.jsonl").read_text()
    assert '"_labels"' in raw
    back = read_jsonl(tmp_path / "t.jsonl")[0]
    assert back.labels.carrier == "C3"
    assert back.model_dump() == t.model_dump()


def test_inspect_subagent_tree(inspect_fixture_log):
    [t] = convert_eval_log(inspect_fixture_log)
    main, sub = t.context("main"), t.context("sub-1")
    assert sub.parent == "main" and sub.agent_name == "researcher"
    spawn = t.step(sub.spawned_at)
    assert [c.name for c in spawn.tool_calls] == ["researcher"]
    # The orchestrator's request is the subagent's user turn.
    assert sub.steps[0].role == "user" and sub.steps[0].text == "Summarize notes.md"
    # The researcher's result reaches main only after the subagent's steps.
    result = next(s for s in main.steps if s.role == "tool" and s.tool_name == "researcher")
    assert result.seq > max(s.seq for s in sub.steps)
    [w] = t.workspace_diffs
    assert (w.path, w.content, w.after.ctx) == ("summary.txt", "fragment one", "sub-1")
    assert main.system_prompt and sub.system_prompt and main.system_prompt != sub.system_prompt
    seqs = [s.seq for _, s in t.steps_in_order()]
    assert seqs == sorted(set(seqs))


WISP = RAW / "wisp-sessions" / "transcripts"
FABLE = RAW / "fable5-traces"


@needs(WISP)
def test_claude_code_workflow_subagents_link_to_their_spawn():
    session = WISP / "-home-me" / "df04f351-9677-4723-b621-7fb5a597caba.jsonl"
    t = convert_session(session)
    subs = [c for c in t.contexts if c.parent == "main"]
    assert len(subs) == 41  # 43 .jsonl files, minus one journal.jsonl per workflow run
    linked = [c for c in subs if c.spawned_at is not None]
    assert len(linked) == 41
    assert {t.step(c.spawned_at).tool_calls[0].name for c in linked} == {"Workflow"}
    assert t.context_files, "expected nested_memory / file attachments as context files"


@needs(FABLE)
def test_claude_code_real_sessions_convert_cleanly():
    traces = convert_dir(FABLE)
    assert len(traces) == 18
    for t in traces:
        TraceTree.model_validate(t.model_dump(by_alias=True))
        seqs = [s.seq for _, s in t.steps_in_order()]
        assert seqs == sorted(set(seqs))
        for w in t.workspace_diffs:
            assert t.step(w.after).role == "tool"
    assert sum(len(t.workspace_diffs) for t in traces) > 0
    assert sum(len(c.steps) for t in traces for c in t.contexts) > 1000
