# Pare

A coding agent built as **research infrastructure** for studying
self-correction in LLM agents on real-world bug-fixing tasks.

Pare runs against [SWE-bench Verified](https://www.swebench.com/) and
emits classifiable, fully-reproducible trajectories with provenance from
every tool call to the final `git diff`. Downstream pipelines turn
those trajectories into a recovery-focused SFT corpus and a
failure-injection harness for measuring agent recovery capability under
controlled conditions.

> Pare is **not a product.** It's a trajectory-generation instrument
> for one specific research question:
>
> *Does SFT on trajectories containing tool-call-level error-correction
> patterns transfer self-correction capability to student models,
> measured by conditional recovery rate under controlled failure
> injection?*

---

## What's shipped today (v0.1.1)

| Artefact | Status | Where |
|---|---|---|
| 4-stage trajectory pipeline | ✅ | `experiments/{prepare,materialize,generate,classify}_*.py` |
| OpenAI-format SFT exporter w/ filter + audit report | ✅ | `experiments/export_sft_dataset.py`, [`pare/trajectory/sft_export.py`](pare/trajectory/sft_export.py) |
| Failure-injection harness (3 fault types, REGISTRY + revert-always contract) | ✅ | [`pare/eval/failure_injection.py`](pare/eval/failure_injection.py), [`experiments/run_failure_injection.py`](experiments/run_failure_injection.py) |
| Liu et al. 9-category classifier + L1/L2/L3 recovery detector | ✅ | [`pare/trajectory/classifier_liu.py`](pare/trajectory/classifier_liu.py) |
| sympy-20 3-arm pilot dataset | ✅ | [`docs/dataset_card.md`](docs/dataset_card.md) |
| κ inter-rater annotation protocol | ✅ | [`docs/annotation_protocol.md`](docs/annotation_protocol.md) |
| Test suite (Win + Linux/WSL) | ✅ | 729 passed, 22 skipped |
| Trained student model + capability eval | ❌ | see *Limitations* below |
| `compute_kappa.py` + second-labeller data | ❌ | needs external annotator |

---

## Pipeline

Six deterministic stages, each a standalone script under `experiments/`.
Stages 1–4 generate the dataset; 5–6 turn it into trainable + evaluable
artefacts.

```
1. prepare_swe_bench_verified  ─►  tasks.jsonl
                                       │
2. materialize_swe_bench_workdirs  ─►  per-instance git worktrees
                                       │
3. generate_trajectories       ─►  trajectories.jsonl
   • flat ReAct loop, fixed seed
   • ablation switches:
       --use-orient   --use-planner   --use-test-nudge
                                       │
4. classify_trajectories       ─►  labels.jsonl
   • Liu 9-category, rule-based (no LLM)
   • L1/L2/L3 recovery levels
                                       │
5. export_sft_dataset          ─►  sft.jsonl  (+ <output>.report.json)
   • OpenAI fine-tuning chat format
   • outcome / recovery / toxic filters w/ schema validation
                                       │
6. run_failure_injection       ─►  fault_injection.jsonl  (+ report)
   • REGISTRY × tasks × seeds
   • apply → run → revert (revert-always contract)
   • per-fault aggregate table
```

A v0 pilot — 20 SWE-bench Verified tasks across 8 repos (django 50%,
astropy 15%, sklearn 10%, ...), 3 ablation arms, 1 seed, MiniMax-M2.5
generator — is documented in [`docs/dataset_card.md`](docs/dataset_card.md)
with full row counts, drop reasons, and known biases.

---

## Quickstart

Requires **Python 3.12+**.

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Full test suite
pytest                        # 729 passed, 22 skipped, ~70s

# 1. Sample 20 tasks from SWE-bench Verified
python -m experiments.prepare_swe_bench_verified \
    --output-jsonl data/sympy20/tasks.jsonl \
    --sample-size 20 --seed 0

# 2. Materialize per-instance git worktrees
python -m experiments.materialize_swe_bench_workdirs \
    --tasks-jsonl    data/sympy20/tasks.jsonl \
    --repos-root     data/swebench_repos \
    --workdirs-root  data/swebench_workdirs \
    --output-jsonl   data/sympy20/tasks_with_cwd.jsonl

# 3. Generate trajectories (one of three ablation arms)
python -m experiments.generate_trajectories \
    --tasks-jsonl       data/sympy20/tasks_with_cwd.jsonl \
    --trajectory-jsonl  data/sympy20/arm3_full.jsonl \
    --provider          minimax \
    --use-orient --use-planner --use-test-nudge \
    --seeds             0

# 4. Classify (writes <trajectory>.labels.jsonl beside the input)
python -m experiments.classify_trajectories \
    --trajectory-jsonl  data/sympy20/arm3_full.jsonl \
    --tasks-jsonl       data/sympy20/tasks_with_cwd.jsonl

# 5. Export SFT JSONL (recovery-only slice)
python -m experiments.export_sft_dataset \
    --trajectory-jsonl  data/sympy20/arm3_full.jsonl \
    --output-jsonl      data/sft/recovery_only.jsonl \
    --include-recovery-only

# 6. (optional) Run the failure-injection harness
python -m experiments.run_failure_injection \
    --tasks-jsonl   data/sympy20/tasks_with_cwd.jsonl \
    --output-jsonl  data/eval/fault_injection.jsonl \
    --workdir-root  data/swebench_workdirs \
    --faults        all \
    --seeds         0
```

---

## Project layout

```
pare/
  agent/         flat ReAct loop + orient_v2 / planner_v2 pre-passes
  cli/           single headless entry point (run_headless_flat_react)
  context/       compactor + memory index
  eval/          failure_injection.py — fault REGISTRY + run_with_fault
  llm/           OpenAI-compatible provider adapters
  sandbox/       InstanceContainer (Docker), DockerEvalSession, ImageBuilder
  tools/         bash, file_read, file_edit, search
  curation/      sampler.py — token-budget-matched sampling for SFT mixes
  defender/      git_exploitation_defender (WIP — rules in place, no real
                                            cases tested yet)
  export/        legacy SFT exporter (superseded by trajectory/sft_export)
  trajectory/    schema, classifier_liu, recovery_detector_v2, sft_export
  telemetry.py   JSONL event logger
experiments/     six-stage pipeline + ablation plotting
scripts/         one-off migration scripts (e.g. v0.1.0 → v0.1.1)
tests/           729 tests across Win + Linux
docs/            dataset_card.md, annotation_protocol.md
```

---

## Design principles

Every decision answers *"how does this support the experimental claim?"*

- **Determinism.** Fixed seeds, no planner-replan loop, no
  `recall_history` tool. Trajectories must be byte-reproducible.
- **Variable isolation.** Planner step-level `budget` is a soft prompt
  signal only; the executor's hard cap is a fixed external guardrail.
  The three ablation switches (`--use-orient`, `--use-planner`,
  `--use-test-nudge`) toggle a single variable each.
- **Auditability.** Every tool call emits a structured `ToolCallEvent`;
  every trajectory carries its full `final_diff` against `gold_patch`.
- **No sandboxing theater.** Pare doesn't own Docker. Tier-2
  verification runs against a caller-supplied Python interpreter
  inside an `InstanceContainer`.
- **Honest naming.** Schema fields mean what they say. v0.1.1 renamed
  `tier1_pass` → `has_diff` because the field was always literally "the
  agent produced a non-empty diff", not a real tier-1 verifier — see
  [`docs/dataset_card.md`](docs/dataset_card.md) §2.1.

---

## Limitations

Documented in plain text because hand-waving over them is what makes
research tools fragile.

- **No trained student model yet.** The recovery-SFT corpus exists; the
  fine-tuning loop on it does not. The headline "did SFT-on-recovery
  improve recovery rate" question is unanswered.
- **Failure-injection runs against a stub agent.** The current CLI
  default is a `dry_run_agent_runner` returning a synthetic trajectory.
  Wiring up the real headless runner needs either a host-mode agent
  variant or in-container fault application — both are P1
  architectural changes, not landed.
- **Pilot is not a capability claim.** N=20 tasks × 1 seed × 1
  provider, 38 recovery-only SFT rows. The pipeline works; statistical
  power is paper-grade-elsewhere.
- **Classifier κ unvalidated.** The annotation protocol is written
  ([`docs/annotation_protocol.md`](docs/annotation_protocol.md)) but
  there's no second-labeller data yet. Until a κ run lands, treat
  classifier outputs as proposed labels, not ground truth.
- **Repo coverage skewed.** "sympy20" is a historical name — the
  actual pilot is 50% django / 15% astropy / 10% sklearn. See
  [`docs/dataset_card.md`](docs/dataset_card.md) §4 for the full
  distribution.

---

## Tests

```bash
pytest                              # full suite
pytest tests/test_eval              # failure-injection scaffold
pytest tests/test_trajectory        # schema + classifier + SFT export
pytest tests/test_experiments       # CLI scripts (round-trip + filters)
pytest tests/test_scripts           # migration script
```

---

## References

- Liu et al. (2025). *A Taxonomy of Failure Modes in LLM-Based Coding
  Agents.* The 9-category taxonomy (A1/A2, B1.1/B1.2, B2.1/B2.2,
  C1/C2, D) is the classification target.
- Kumar et al. (2024). *Training Language Models to Self-Correct via
  Reinforcement Learning (SCoRe).* Orthogonal — SCoRe targets CoT
  self-correction on math; Pare studies *tool-call-level*
  self-correction on code, where errors are externally observable.
- Jimenez et al. (2024). *SWE-bench: Can Language Models Resolve
  Real-World GitHub Issues?* Task source.

## License

MIT.
