# carrier-eval

Plumbing for the agentic carrier-gap eval. The design, decisions and rationale are in
[`agentic-carrier-gap-eval.md`](agentic-carrier-gap-eval.md). This README covers how to run the code.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,decomposedharm]"
.venv\Scripts\python.exe -I -m pytest -q
```

On Linux or macOS, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

Run Python with `-I` whenever it reads anything under `data/raw/`, so downloaded files can't shadow imports.
Downloaded data isn't in the repo; `data/README.md` says how to fetch it. Tests that need it skip themselves if it is missing.

## Pipeline

Rollouts are generated once, then every monitor arm replays the same sanitized traces.

```powershell
# 1. Harness logs -> trace trees (contexts with parent pointers, global seq, context files, file writes)
carrier-eval convert-inspect logs\*.json --out work\raw.jsonl
carrier-eval convert-claude-code path\to\session.jsonl --out work\raw.jsonl

# 2. Outcome verification (generator side): fills E, R, realized dose, verified
carrier-eval verify-wmdp work\raw.jsonl --payloads work\payloads.json --out work\verified.jsonl

# 3. Sanitize: strips labels to labels.jsonl, remaps ids and generator-chosen paths
carrier-eval sanitize work\verified.jsonl --out-dir work\clean --seed 0

# 4. Replay one arm (configs\arms\*.json; set MONITOR_MODEL / ORACLE_MODEL to any Inspect model id)
$env:MONITOR_MODEL = "vllm/Qwen/Qwen3-8B"
carrier-eval replay work\clean\traces.jsonl --arm configs\arms\M4_tree_notes.json --out work\tl_M4_notes.jsonl

# 5. Frontiers per carrier, area over FPR 0.1-10%, gap vs C0 with cluster-bootstrap CIs, plot + table
carrier-eval frontier --timelines work\tl_*.jsonl --labels work\clean\labels.jsonl --out-dir results

# DecomposedHarm: which AgentHarm graders are exact (no LLM judge)
carrier-eval classify-graders data\raw\decomposedharm-code\repo\Modified_AgentHarm\benchmark\harmful_grading_functions.py
```

## Layout

| Path | What |
|---|---|
| `src/carrier_eval/schema.py` | Trace-tree schema: contexts, steps with global `seq`, context files, workspace writes, generator-side `_labels` |
| `src/carrier_eval/convert/` | Inspect log converter (agent spans -> contexts) and Claude Code transcript converter (subagent files linked by `agentId` / workflow `runId`) |
| `src/carrier_eval/sanitize.py` | Label stripping, id remapping (one mapper per dataset, so accounts stay linked), path renaming |
| `src/carrier_eval/verify/` | Exact outcome verifiers: WMDP-Bio answer keys; DecomposedHarm via AgentHarm graders (exact ones only) |
| `src/carrier_eval/monitors/` | Arm configs (view x scope x cadence x history), scorers (keyword, any Inspect model, guard models with P(unsafe)), replay |
| `src/carrier_eval/results/` | Frontiers with timeliness folded in, partial area on a log axis, carrier gaps, flip rate, plots, table |
| `configs/arms/` | M0-M5 and oracle presets |
| `scripts/make_fixture_log.py` | Scripted Inspect run (mockllm) with a subagent; the converter's test fixture |
| `data/` | See `data/README.md` for provenance, licenses, gating and canaries |

## Not built yet

- `history: retrieval` and `product_memory` (design doc A5) raise `NotImplementedError`.
- Rollout harness and the carrier attacker agent (build step 5), which need the local policy model (A6).
- Rubric-judge verifier for DecomposedHarm tasks without exact graders (31 of 50).
- Activation probes (A2).
