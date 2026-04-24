---
dataset_name: pare-recovery-sft-swebench20-v0
version: 0.1.0-pilot
schema: openai-chat
license: MIT
source_tasks: princeton-nlp/SWE-bench_Verified
generation_provider: minimax (MiniMax-M2.5)
classifier: rule-based (Liu et al. 2025 taxonomy + L1/L2/L3 recovery)
status: pilot — not for capability claims
---

# Pare Recovery-SFT Dataset — v0 pilot card

> **Status: pilot.** This card documents a 20-task × 3-arm pilot generated
> on a single seed. It's a pipeline-validation artefact, not a corpus you
> should train a production model on. The row counts are too small for any
> capability claim to hold up to statistical scrutiny.
>
> See `plan.md` P1 for the scale-up (3 seeds × multi-repo, ~225 raw
> trajectories) that this pilot is a dry-run for.

## 1. What this dataset is

A JSONL where each line is an **OpenAI fine-tuning chat conversation**
reconstructed from one Pare agent trajectory on a SWE-bench Verified
task. Rows are optionally filtered to those containing at least one
**recovery event** (L1/L2/L3 per `pare.trajectory.recovery_detector_v2`),
making it a candidate corpus for teaching a student model to
self-correct at tool-call granularity.

**Research question** (from `plan.md`):

> Does SFT on trajectories containing tool-call-level error-correction
> patterns transfer self-correction capability to student models,
> measured by conditional recovery rate under controlled failure
> injection?

## 2. Row schema

```json
{
  "messages": [
    {"role": "user",      "content": "<SWE-bench problem statement>"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0_0", "type": "function",
         "function": {"name": "search", "arguments": "{...}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_0_0",
     "content": "<tool output>"},
    ...
  ],
  "metadata": {
    "trajectory_id": "traj-1776874535-9544e5e0",
    "instance_id":   "astropy__astropy-14508",
    "seed": 0,
    "model": "MiniMax-M2.5",
    "final_passed": true,
    "tier1_pass": true, "tier2_pass": true,
    "input_tokens": 151738, "output_tokens": 4106,
    "tool_call_count": 19,
    "outcome": "verified_with_recovery",
    "contains_recovery": true,
    "highest_recovery_level": "L1",
    "recovery_event_count": 1,
    "liu_categories": ["A2"],
    "is_toxic": false
  }
}
```

Key format decisions:

- **Assistant content is always empty string.** Pare's flat-ReAct loop
  doesn't persist free-text reasoning between tool calls; only the
  structured tool-call sequence is recorded. Empty content is the
  correct SFT target for "train on tool-call decisions, not a paraphrase
  of thought."
- **`tool_call_id` is deterministic** (`call_<turn_id>_<call_index>`),
  synthesized at export time. The wire-format id isn't persisted in
  `TrajectoryRecord` and isn't needed — OpenAI's format only requires
  uniqueness within a conversation.
- **`arguments` is a JSON string**, not an object (OpenAI requirement).
- **One assistant message per `turn_id`** with all parallel tool_calls
  of that turn, followed by one `tool` reply per call in order. This
  preserves the ReAct cadence the agent was actually recorded in.

## 3. Collection protocol

Four deterministic stages (all under `experiments/`):

```
prepare_swe_bench_verified
   ↓  tasks.jsonl
materialize_swe_bench_workdirs
   ↓  per-instance git worktrees on named branches
generate_trajectories --provider minimax --seeds 0
   ↓  <arm>.jsonl (one TrajectoryRecord per run)
classify_trajectories
   ↓  <arm>.labels.jsonl  (Liu categories + L1/L2/L3 recovery)
export_sft_dataset
   ↓  <arm>.<filter>.jsonl  (this file)
```

### Ablation arms

Three arms, same 20 tasks and same seed, differing only in agent
configuration:

| Arm | `--use-orient` | `--use-planner` | `--use-test-nudge` | Purpose |
|---|:--:|:--:|:--:|---|
| arm1 baseline          | ✗ | ✗ | ✗ | Control                                     |
| arm2 prepasses         | ✓ | ✓ | ✗ | Adds orient_v2 + planner_v2 upfront pre-passes |
| arm3 prepasses + nudge | ✓ | ✓ | ✓ | + B2.1 Wrong-Fix "no test after N edits" nudge |

## 4. Statistics (v0 pilot)

### Source task distribution (20 instances from SWE-bench Verified)

| Repo          | Count | % |
|---------------|------:|--:|
| django        | 10    | 50% |
| astropy       |  3    | 15% |
| scikit-learn  |  2    | 10% |
| sympy         |  1    |  5% |
| pytest-dev    |  1    |  5% |
| pylint-dev    |  1    |  5% |
| pydata        |  1    |  5% |
| matplotlib    |  1    |  5% |

> **Naming note.** The on-disk directory is `data/sympy20/` for historical
> reasons — the first pilot was meant to be sympy-only, then was broadened
> to a mixed sample without renaming. A future data release will rename to
> `swebench20`. Row `metadata.instance_id` is the authoritative source.

### Outcome distribution (after classifier, before SFT filters)

All three arms produced **15/20 verified** (one-shot + with-recovery),
i.e. a flat 75% success rate across arms. Test_nudge had no measurable
effect in this slice — B2.1 Wrong-Fix signature (edit/bash ratio ≥ 3.0)
does not occur in this 20-instance sample (all ratios observed < 0.5).
This is a **clean null**, reported as such.

### Exported row counts per arm

| Arm | raw | `recovery_only` | `all_verified` |
|---|---:|---:|---:|
| arm1 baseline          | 20 | **15** | 15 |
| arm2 prepasses         | 20 | **12** | 15 |
| arm3 prepasses + nudge | 20 | **11** | 15 |
| **all-arms pooled**    | 60 | **38** | 45 |

(`all_verified` = `verified_one_shot` ∪ `verified_with_recovery`.
`recovery_only` = `contains_recovery == True`.)

## 5. Known biases and tradeoffs

### 5.1 Arm-quality ↔ training-signal tradeoff (important)

The better-configured arm produces **fewer** recovery demonstrations:

```
arm1 baseline           → 15 recovery rows  ← most training signal
arm2 prepasses          → 12
arm3 prepasses + nudge  → 11                ← least training signal
```

This is **expected**, not a bug. orient_v2 / planner_v2 reduce the
frequency of the stumbles the agent has to recover from. For a
self-correction SFT corpus this is a tension we have to manage:

- **Training bucket** should be weighted toward baseline arms (more
  recovery examples per trajectory).
- **Eval bucket** should be weighted toward full arms (the agent we
  actually ship runs with prepasses on).

A future release will publish the two buckets separately rather than
a single pooled JSONL.

### 5.2 Repo imbalance

Django is 50% of tasks. A student SFT'd on this pilot will likely
overfit Django's test-collection patterns (`manage.py test`,
`./runtests.py`). P1 rebalances via stratified multi-repo sampling.

### 5.3 Single generation seed

All three arms ran on `seed=0`. Per-task variance is not estimated.
P1 adds `seed ∈ {0, 1, 2}`.

### 5.4 Single provider

All generations use `MiniMax-M2.5` via `--provider minimax`. Results
don't generalize to other provider/model combos in this pilot.

### 5.5 Classifier validation still pending

The Liu-taxonomy classifier + recovery detector are rule-based and
locally tested, but **not yet validated against a human labeller**
(Cohen's κ run is P0.3 in `plan.md`, scheduled). Treat every
`outcome` / `liu_categories` / `highest_recovery_level` field as
*a provisional classifier assertion* until κ lands.

### 5.6 Toxic rows are dropped by default

The exporter's `drop_toxic=True` default filters out trajectories the
classifier flags as Liu C1 (Premature Success) or C2 (Fabricated
Patch). This is intentional for training corpora but leaves the
"what does a toxic trajectory look like" question unanswered in this
release. The raw trajectory JSONLs retain them.

## 6. Reproduction

Given `data/sympy20/arm{1_baseline,2_prepasses,3_full}.jsonl` already
produced by stages 1–4 (see `README.md` for the upstream commands),
this dataset is regenerated by:

```bash
mkdir -p data/sft/sympy20

for arm in arm1_baseline arm2_prepasses arm3_full; do
  python -m experiments.export_sft_dataset \
    --trajectory-jsonl data/sympy20/${arm}.jsonl \
    --output-jsonl     data/sft/sympy20/${arm}.recovery_only.jsonl \
    --include-recovery-only

  python -m experiments.export_sft_dataset \
    --trajectory-jsonl data/sympy20/${arm}.jsonl \
    --output-jsonl     data/sft/sympy20/${arm}.all_verified.jsonl \
    --include-outcome verified_one_shot \
    --include-outcome verified_with_recovery
done

cat data/sft/sympy20/arm*.recovery_only.jsonl \
    > data/sft/sympy20/all_arms.recovery_only.jsonl
cat data/sft/sympy20/arm*.all_verified.jsonl \
    > data/sft/sympy20/all_arms.all_verified.jsonl
```

Every export writes a `<output>.report.json` sidecar with filter
provenance and drop counts — attach that to any downstream
paper-table claim.

## 7. Intended use

✅ **Appropriate:**
- End-to-end pipeline validation (format → trainer → student output)
- Smoke SFT runs (does the loss decrease? does the student emit
  format-compliant tool_calls?)
- Drafting dataset-card templates for the P1 scale release

❌ **Not appropriate:**
- Any capability claim (e.g. "SFT on Pare recovery rows improves
  self-correction rate by X%")
- Benchmark leaderboard scores
- Comparative studies against other corpora

Wait for v1 (P1 scale, 200+ recovery rows, multi-seed, multi-repo
balanced) before making any quantitative claim.

## 8. Provenance / version pinning

| Field | Value |
|---|---|
| Pipeline code | commit `c46a55d` on `claude/great-carson-333acd` |
| Generation date | 2026-04-22 |
| Classifier version | `pare.trajectory.classifier_liu` @ same commit |
| Source dataset | `princeton-nlp/SWE-bench_Verified` test split |
| Task sampler seed | 0 (via `experiments.prepare_swe_bench_verified`) |
| Agent seed | 0 |
| Provider | `minimax` (MiniMax-M2.5) |
| Instances | 20 mixed-repo (see §4) |

## 9. Changelog

- **v0.1.0-pilot (2026-04-22)** — initial 20-task × 3-arm export.
  38 recovery-only rows, 45 all-verified rows pooled.
