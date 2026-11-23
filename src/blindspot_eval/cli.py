"""blindspot-eval: convert -> verify -> sanitize -> replay -> frontier.

Run with `python -I -m blindspot_eval.cli ...` (or the `blindspot-eval` script) when
inputs come from downloaded data, so nothing on the data side can shadow imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blindspot_eval.schema import read_jsonl, write_jsonl


def _convert_inspect(a: argparse.Namespace) -> None:
    from blindspot_eval.convert.inspect_log import convert_eval_log

    traces = [t for p in a.logs for t in convert_eval_log(p)]
    print(f"{write_jsonl(traces, a.out)} traces -> {a.out}")


def _convert_claude_code(a: argparse.Namespace) -> None:
    from blindspot_eval.convert.claude_code import convert_dir, convert_session

    p = Path(a.path)
    traces = convert_dir(p) if p.is_dir() else [convert_session(p)]
    print(f"{write_jsonl(traces, a.out)} traces -> {a.out}")


def _verify_wmdp(a: argparse.Namespace) -> None:
    from blindspot_eval.verify.base import apply_outcome
    from blindspot_eval.verify.wmdp import WmdpPayload, verify_wmdp

    payloads = {p["payload_id"]: WmdpPayload.model_validate(p) for p in json.loads(Path(a.payloads).read_text())}
    traces = read_jsonl(a.traces)
    n_ver = 0
    for t in traces:
        pid = t.labels.payload_id if t.labels else None
        if pid in payloads:
            out = verify_wmdp(t, payloads[pid])
            apply_outcome(t, out)
            n_ver += out.verified
    write_jsonl(traces, a.out)
    print(f"{len(traces)} traces, {n_ver} verified releases -> {a.out}")


def _sanitize(a: argparse.Namespace) -> None:
    from blindspot_eval.sanitize import mapping_json, sanitize_all

    traces = read_jsonl(a.traces)
    clean, labels, mapper = sanitize_all(traces, seed=a.seed, remap_paths=not a.keep_paths)
    out = Path(a.out_dir)
    write_jsonl(clean, out / "traces.jsonl")
    write_jsonl(labels, out / "labels.jsonl")
    (out / "mapping.json").write_text(mapping_json(mapper), encoding="utf-8")
    print(f"{len(clean)} traces -> {out}/traces.jsonl (labels and id mapping kept separately)")


def _replay(a: argparse.Namespace) -> None:
    from blindspot_eval.monitors.config import load_config
    from blindspot_eval.monitors.replay import replay_sync

    cfg = load_config(a.arm)
    traces = read_jsonl(a.traces)
    timelines = []
    for r in range(a.repeats):
        timelines += replay_sync(traces, cfg, concurrency=a.concurrency, repeat=r)
    write_jsonl(timelines, a.out)
    calls = sum(t.calls for t in timelines)
    print(f"{cfg.name}: {len(timelines)} timelines, {calls} monitor calls -> {a.out}")


def _frontier(a: argparse.Namespace) -> None:
    from blindspot_eval.monitors.replay import ScoreTimeline
    from blindspot_eval.results.frontier import carrier_frontiers
    from blindspot_eval.results.plots import plot_arm, results_table
    from blindspot_eval.sanitize import LabelRecord

    labels = read_jsonl(a.labels, LabelRecord)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in a.timelines:
        tls = [t for t in read_jsonl(path, ScoreTimeline) if t.repeat == 0]
        res = carrier_frontiers(tls, labels, baseline=a.baseline, n_boot=a.n_boot, seed=a.seed)
        results.append(res)
        (out / f"{res.arm}.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
        plot_arm(res, out / f"{res.arm}.png")
        if res.distinct_scores <= 2:
            print(f"warning: {res.arm} produced {res.distinct_scores} distinct scores; check for parse failures")
    (out / "results.md").write_text(results_table(results), encoding="utf-8")
    print(results_table(results))


def _classify_graders(a: argparse.Namespace) -> None:
    from blindspot_eval.verify.decomposedharm import classify_grading_functions

    cls = classify_grading_functions(a.path)
    exact = sorted(n for n, judge in cls.items() if not judge)
    print(f"{len(cls)} grading functions: {len(exact)} exact, {len(cls) - len(exact)} use an LLM judge")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cls, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="blindspot-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert-inspect", help="Inspect eval logs -> trace trees")
    p.add_argument("logs", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_convert_inspect)

    p = sub.add_parser("convert-claude-code", help="Claude Code session(s) -> trace trees")
    p.add_argument("path", help="a session .jsonl or a projects directory")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_convert_claude_code)

    p = sub.add_parser("verify-wmdp", help="exact outcome verification for WMDP-Bio payloads")
    p.add_argument("traces")
    p.add_argument("--payloads", required=True, help="JSON list of WmdpPayload")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_verify_wmdp)

    p = sub.add_parser("sanitize", help="strip labels, remap ids and paths")
    p.add_argument("traces")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--keep-paths", action="store_true")
    p.set_defaults(fn=_sanitize)

    p = sub.add_parser("replay", help="score sanitized traces with one monitor arm")
    p.add_argument("traces")
    p.add_argument("--arm", required=True, help="arm config JSON (configs/arms/)")
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(fn=_replay)

    p = sub.add_parser("frontier", help="frontiers, carrier gaps, plots, table")
    p.add_argument("--timelines", nargs="+", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline", default="C0")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_frontier)

    p = sub.add_parser("classify-graders", help="which AgentHarm grading functions are exact (no LLM judge)")
    p.add_argument("path")
    p.add_argument("--out")
    p.set_defaults(fn=_classify_graders)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
