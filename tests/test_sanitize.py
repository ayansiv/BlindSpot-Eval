from carrier_eval.convert.builder import TraceBuilder
from carrier_eval.sanitize import sanitize_all

from conftest import split_trace


def test_labels_split_off_and_ids_remapped():
    a = split_trace(["alpha"], trace_id="harmful_0181", account_id="acct-7")
    b = split_trace(["beta"], trace_id="benign_0002", account_id="acct-7", positive=False)
    clean, labels, mapper = sanitize_all([a, b], seed=1)
    for t in clean:
        assert t.labels is None and t.meta == {}
        assert "harmful" not in t.trace_id and "benign" not in t.trace_id
    assert clean[0].account_id == clean[1].account_id != "acct-7"
    assert labels[0].trace_id == clean[0].trace_id and labels[0].positive is True
    assert labels[1].positive is False and labels[1].carrier == "C3"


def test_tool_call_ids_stay_paired():
    [t], _, _ = sanitize_all([split_trace(["alpha", "beta"])], seed=2)
    calls = {c.id for c in t.contexts[0].steps[1].tool_calls}
    results = {s.tool_call_id for s in t.context("main").steps if s.role == "tool"}
    assert calls & results


def test_paths_renamed_everywhere_except_conventional_names():
    t = split_trace(["alpha"])
    t.contexts[0].steps[0].text = "See fragment_1.txt and CLAUDE.md before starting."
    b = TraceBuilder()
    b.add_context_file("CLAUDE.md", "Use fragment_1.txt as the scratch file.", loaded_at=None)
    t.context_files = b.context_files
    [c], _, _ = sanitize_all([t], seed=3)
    blob = c.model_dump_json()
    assert "fragment_1" not in blob
    assert "CLAUDE.md" in blob
    assert c.workspace_diffs[0].path.startswith("file_") and c.workspace_diffs[0].path.endswith(".txt")
    write_args = c.context("sub-1").steps[1].tool_calls[0].args
    assert write_args["path"] == c.workspace_diffs[0].path
