# Pare

A lightweight coding agent built as research infrastructure for studying
self-correction in LLM agents. Pare produces classifiable, reproducible
trajectories from SWE-bench tasks — with full provenance from tool-call events
to final git diffs.

Pare is **not a product.** It is a trajectory-generation instrument.

## What it's for

The core research question:

> Does SFT on trajectories containing tool-call-level error-correction patterns
> transfer self-correction capability to student models, measured by conditional
> recovery rate under controlled failure injection?

Pare generates the trajectories (one JSONL row per run) that feed a downstream
classification + curation + defender pipeline grounded in Liu et al.'s
9-category failure taxonomy [1].

## Pipeline

Four deterministic stages, each a standalone script:

```
prepare_swe_bench_verified  →  materialize_swe_bench_workdirs
         │                              │
         ▼                              ▼
    tasks.jsonl                   per-instance git worktrees
         │                              │
         └──────────────┬───────────────┘
                        ▼
            generate_trajectories  ──►  trajectories.jsonl
                        │
                        ▼
            classify_trajectories  ──►  labels.jsonl + summary.json
```

1. **Prepare** — sample SWE-bench Verified, normalize `FAIL_TO_PASS` to pytest
   node ids (Django unittest form and sympy bare-name form both handled).
2. **Materialize** — clone target repos once, create per-instance git
   worktrees on named branches (required for `git diff` to produce real
   `final_diff`).
3. **Generate** — run the Pare agent headlessly against each workdir with a
   fixed seed. Emits one `TrajectoryRecord` per run, including every tool call,
   token usage, `final_diff`, and Tier 1/2 verification state.
4. **Classify** — rule-based, deterministic labeling (no LLM). Produces Liu
   et al. category counts, recovery-level labels (L1/L2), and toxicity flags.

## Quickstart

Requires **Python 3.12+**.

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the test suite
pytest

# 1. Prepare a 20-task sympy pilot from SWE-bench Verified
python -m experiments.prepare_swe_bench_verified \
  --output-jsonl data/sympy20/tasks.jsonl \
  --sample-size 20 --seed 0 \
  --repo-filter sympy/sympy

# 2. Materialize workdirs
python -m experiments.materialize_swe_bench_workdirs \
  --tasks-jsonl data/sympy20/tasks.jsonl \
  --repos-root data/swebench_repos \
  --workdirs-root data/swebench_workdirs \
  --output-jsonl data/sympy20/tasks_with_cwd.jsonl

# 3. Generate trajectories
python -m experiments.generate_trajectories \
  --tasks-jsonl data/sympy20/tasks_with_cwd.jsonl \
  --trajectory-jsonl data/sympy20/trajectories.jsonl \
  --provider minimax --seeds 0

# 4. Classify
python -m experiments.classify_trajectories \
  --trajectory-jsonl data/sympy20/trajectories.jsonl \
  --tasks-jsonl data/sympy20/tasks_with_cwd.jsonl
```

## Project layout

```
pare/
  agent/         orchestrator, planner, executor, guardrails
  cli/           headless entry point
  context/       memory index + session history
  llm/           provider adapters (OpenAI-compatible only)
  sandbox/       git checkpoint, tool sandbox
  tools/         bash, file_read, file_edit, search
  telemetry.py   JSONL event logger
  curation/      trajectory scoring + sampling (WIP)
  defender/      git-exploitation defender (WIP)
  export/        SFT format export (OpenAI messages / HF chat template)
  trajectory/    classifier + recovery detector
experiments/     the 4-stage pipeline scripts above
tests/           pytest suite (643+ tests)
```

## Design principles

Research-first, not product-first. Every design decision answers "how does
this support the experimental claim?"

- **Determinism** — fixed seeds, no planner replan loop, no `recall_history`
  tool. Trajectories must be reproducible.
- **Variable isolation** — planner step-level `budget` is a soft prompt
  signal only; the executor's hard cap is a fixed external guardrail.
- **Auditability** — every tool call emits a structured event; every
  trajectory carries its full `final_diff` vs `gold_patch`.
- **No sandboxing theater** — Pare does not own Docker. Tier-2 verification
  runs against a caller-supplied Python interpreter.

## Tests

```bash
pytest                           # full suite
pytest tests/test_agent          # orchestrator / planner / guardrails
pytest tests/test_experiments    # pipeline scripts
```

## References

[1] Liu et al. (2025). *A Taxonomy of Failure Modes in LLM-Based Coding Agents.*
    The 9-category Liu taxonomy (A1/A2, B1.1/B1.2, B2.1/B2.2, C1/C2, D) is the
    classification target.

## License

MIT.
