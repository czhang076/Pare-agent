# Pare — Research Project Plan

**Author:** Chenz
**Date:** April 2026
**Status:** Infrastructure built, pivoting to research instrument
**Language:** Python 3.12+

---

## 1. Research Question & Hypotheses

### 1.1 Core question

**Does training on failure-recovery trajectories improve a coding model's ability to self-correct, compared to training on clean (one-shot success) trajectories alone?**

### 1.2 Hypotheses

| # | Hypothesis | Claim | Priority |
|---|---|---|---|
| H1 | Including verified trajectories (Tier 1+2 pass) in SFT data outperforms including unverified ones | Verification-based filtering improves data quality | Secondary |
| **H2** | **Adding failure→recovery trajectories to SFT data improves model self-correction beyond what equal-token clean-only data provides** | **Recovery experience teaches self-correction** | **Primary** |
| H3 | Toxic trajectories (hallucinated success, never-recovered failures) degrade model performance when included in SFT | Data quality > data quantity | Secondary |

H2 is the main hypothesis. H1 and H3 provide supporting evidence for the data curation methodology.

### 1.3 Why this matters

Current SFT trajectory datasets for coding models filter to successes only. This discards failure-recovery sequences where the model encountered an error, diagnosed it, and corrected course — arguably the most valuable learning signal for real-world robustness. No published work systematically studies this.

---

## 2. Pare as Research Infrastructure

Pare is **not a product**. It is a data collection instrument — a coding agent designed to generate classifiable trajectories with full provenance.

### 2.1 Design principles (research-oriented)

| Principle | Meaning | Implication |
|---|---|---|
| **Determinism** | Same input → same execution trace (modulo LLM sampling) | No replan loops, no adaptive retries, seed-controlled |
| **Reproducibility** | Every trajectory fully reconstructable from logs | Instance ID, model, seed, pare version, full JSONL, git SHAs |
| **Variable isolation** | Each design choice is independently togglable | Plan vs flat mode, verification on/off, recovery detection on/off |
| **Auditability** | Every step's before/after state is observable | Git checkpoint per step, verification result per step |

### 2.2 Architecture (simplified for research)

```
┌──────────────────────────────────────────────────┐
│  Headless Entry Point                            │
│  pare run "task" --headless --output result.json │
├──────────────────────────────────────────────────┤
│  Agent: Orient → Plan → Execute (no replan)      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │ Orient  │─▶│  Plan   │─▶│ Execute (bounded│  │
│  │ 0 LLM   │  │ 1 LLM   │  │ ReAct + verify) │  │
│  └─────────┘  └─────────┘  └─────────────────┘  │
├──────────────────────────────────────────────────┤
│  Tools: bash · file_read · file_edit · search    │
├──────────────────────────────────────────────────┤
│  Git Checkpoint (per-step SHA tracking)          │
├──────────────────────────────────────────────────┤
│  Verification: Tier 1 (syntax + diff) · Tier 2   │
├──────────────────────────────────────────────────┤
│  Trajectory JSONL Output (strict schema)         │
└──────────────────────────────────────────────────┘
```

### 2.3 What was cut (and why)

| Cut | Reason |
|---|---|
| Interactive CLI (`pare/cli/`) | Research is headless-only |
| Rich/prompt_toolkit deps | Not needed for batch execution |
| Replan loop | Introduces non-determinism, pollutes trajectory classification |
| Anthropic adapter | Only need OpenAI-compatible (DeepSeek/Qwen as teacher) |
| `recall_history` tool | Makes agent behavior non-deterministic, blocks ablation |
| CI permission tiers | Research doesn't need CI scenario |
| Compaction-triggered system note | Coupled to recall_history |

### 2.4 What is kept and strengthened

| Module | Role in research |
|---|---|
| Git checkpoint | Atomic rollback tracking — core of recovery pattern detection |
| Two-tier hard verification | Core filtering for trajectory classification |
| Session JSONL | Raw trajectory data — schema must be strict |
| Orient → Plan → Execute | Trajectory generation strategy (ablatable: plan vs flat) |
| Guardrails | Budget enforcement ensures deterministic step boundaries |
| Heuristic compression | Keeps long tasks running without burning tokens |
| OpenAI-compatible adapter | Talks to DeepSeek/Qwen/any teacher model |

---

## 3. Trajectory Taxonomy

### 3.1 Trajectory labels (mutually exclusive)

| Label | Definition | SFT value |
|---|---|---|
| **toxic** | LLM reports success but Tier 1 verification fails (syntax error, empty diff) | Negative — exclude or use as negative examples |
| **failed** | Task not completed, no successful recovery | Low — exclude from main training |
| **weakly_verified** | LLM reports success, Tier 1 passes, no Tier 2 | Moderate — include with caution |
| **fully_verified** | Tier 1 + Tier 2 (test_command) pass | High — gold standard |
| **one_shot_success** | Fully verified, no failure events in trajectory | High — clean positive example |
| **failure_recovery** | Fully verified, contains failure→rollback→success pattern | **Core research signal** — H2 test data |

### 3.2 Recovery levels (within failure_recovery trajectories)

| Level | Pattern | Complexity |
|---|---|---|
| L1: Retry | Same approach, minor fix (e.g., syntax error → fix typo) | Low |
| L2: Strategy switch | Different approach after failure (e.g., regex → AST parsing) | Medium |
| L3: Goal decomposition | Breaks failed step into sub-steps | High |

Recovery level classification is rule-based: compare tool call sequences and file targets before/after rollback.

### 3.3 Classification rules (deterministic, zero LLM)

```
1. Did any verification check fail?
   → Yes + LLM claimed success → TOXIC
   → Yes + LLM acknowledged failure → continue

2. Did the task complete (final verification pass)?
   → No → FAILED

3. Which verification tier passed?
   → Tier 1 only → WEAKLY_VERIFIED
   → Tier 1 + Tier 2 → FULLY_VERIFIED

4. Were there failure→recovery events?
   → No → ONE_SHOT_SUCCESS
   → Yes → FAILURE_RECOVERY (+ assign L1/L2/L3)
```

All classification is deterministic. No LLM calls. Implemented in `pare/trajectory/classifier.py`.

---

## 4. Dataset Construction Protocol

### 4.1 Source tasks

SWE-bench Lite (300 instances). Split:
- **Training pool:** 250 instances (trajectory generation + SFT)
- **Held-out evaluation:** 50 instances (never seen during trajectory generation)

### 4.2 Trajectory generation

For each training instance:
- Model: DeepSeek-V3 (or Qwen-2.5-Coder-32B) as teacher
- Seeds: 3 runs per instance (seed 0, 1, 2) → up to 750 raw trajectories
- Mode: Orient → Plan → Execute (no replan)
- Git exploitation defense applied before each run
- Full JSONL + git SHAs recorded per step

### 4.3 Classification & filtering

1. Run `classifier.py` on all 750 trajectories → labels assigned
2. Run `recovery_detector.py` on failure_recovery trajectories → L1/L2/L3
3. Discard toxic trajectories entirely
4. Pool the rest by label

### 4.4 Experimental groups (for H2)

| Group | Composition | Control variable |
|---|---|---|
| **Clean-only** | Only one_shot_success trajectories | Baseline |
| **Mixed** | one_shot_success + failure_recovery (natural ratio) | H2 test |
| **Recovery-enriched** | one_shot_success + failure_recovery (2x natural ratio via oversampling) | H2 sensitivity |
| **Unfiltered** | All non-toxic trajectories (including weakly_verified, failed) | H1/H3 control |

**Critical control: token budget matching.** Groups are sampled so total training tokens are equal (±5%), not sample count. This isolates the effect of trajectory composition from training compute.

### 4.5 SFT export format

Each trajectory → OpenAI messages format (system + user + assistant turns with tool calls and results). Compatible with standard LoRA training scripts and HuggingFace chat templates.

---

## 5. Experimental Design

### 5.1 Independent variable

Composition of SFT training data (4 groups per §4.4).

### 5.2 Dependent variables

| Metric | Evaluation set | What it measures |
|---|---|---|
| SWE-bench Lite pass rate | 50 held-out instances | In-distribution coding ability |
| HumanEval+ pass rate | Full HumanEval+ | Out-of-distribution generalization |
| Recovery rate | 50 held-out (among initially-failed attempts) | Self-correction ability (H2 core metric) |
| Token efficiency | 50 held-out | Tokens per resolved instance |

### 5.3 Fixed variables (controlled)

| Variable | Setting | Justification |
|---|---|---|
| Base model | Qwen-2.5-Coder-7B (or similar) | Small enough for LoRA, large enough to show effects |
| LoRA config | r=16, alpha=32, dropout=0.05 | Standard settings |
| Training tokens | Matched across groups (±5%) | Isolate composition effect |
| Epochs | 3 | Standard for SFT |
| Eval model | Same checkpoint, greedy decoding | Reproducible |
| Teacher model | Fixed (DeepSeek-V3 or Qwen-32B) | Same teacher for all trajectories |
| Seed | Fixed per run | Reproducible |

### 5.4 Ablation studies

1. **Verification filtering:** Compare fully_verified vs weakly_verified vs unfiltered → H1
2. **Recovery level:** Compare L1-only vs L2+L3-only in recovery-enriched group → recovery complexity effect
3. **Plan vs flat mode:** Generate trajectories with both, compare quality metrics → Orient→Plan value

---

## 6. Evaluation Metrics

### 6.1 Primary

- **SWE-bench Lite pass@1** on held-out 50 instances (3 runs, median)
- **Recovery rate**: among tasks where the model initially fails, what fraction does it successfully recover? (Measured on held-out set)

### 6.2 Secondary

- **HumanEval+ pass@1** (OOD generalization)
- **Tokens per resolved instance** (efficiency)
- **Trajectory quality distribution** (what % toxic / failed / clean / recovery per group)

### 6.3 Statistical tests

- McNemar's test for pass rate differences between groups
- Bootstrap confidence intervals (95%) for all metrics
- Report effect sizes, not just p-values

---

## 7. Methodology Validity Safeguards

### 7.1 Git exploitation defense

**Problem:** Teacher model may "cheat" by accessing git history that contains the fix. If the SWE-bench instance was created from a real PR, commits after the issue date may contain the solution.

**Defense:** Before trajectory generation, rewrite the repo's git history to remove all commits after the issue creation date. Implemented in `pare/defender/git_exploitation_defender.py`.

### 7.2 Data contamination prevention

- Held-out evaluation instances are never used for trajectory generation
- Teacher model and student model are different (teacher generates, student is fine-tuned)
- HumanEval+ provides OOD check independent of SWE-bench

### 7.3 Classifier validity

- Classifier is pure rules → zero subjectivity, perfect reproducibility
- Pilot study: 20 trajectories manually labeled by author, compute Cohen's κ against classifier
- Target: κ ≥ 0.7 (substantial agreement) → proceed to full experiment
- If κ < 0.7 → revise classification rules before scaling

### 7.4 Confounding variables checklist

| Confound | Control method |
|---|---|
| Model choice (teacher) | Fixed across all trajectory generation |
| Prompt template | Fixed system prompt, no per-group tuning |
| Random seed | 3 seeds per instance, report median |
| Token budget | Matched across experimental groups (±5%) |
| Training hyperparameters | Fixed LoRA config, same epochs |
| Data ordering | Shuffled with fixed seed |
| Instance difficulty | Same instances across groups (subset matching if needed) |

---

## 8. Timeline & Milestones

### Phase 1: Infrastructure (Done)

- [x] LLM adapter (OpenAI-compatible, DeepSeek/MiniMax/Qwen)
- [x] Tool system (bash, file_read, file_edit, file_create, search)
- [x] ReAct executor with guardrails
- [x] Git checkpoint (setup/commit/rollback/finalize/abort)
- [x] Orient phase (zero-LLM repo scanning)
- [x] Planner (LLM plan generation + JSON fallback)
- [x] Hybrid Orient → Plan → Execute loop
- [x] Headless batch mode (`--headless --output result.json`)
- [x] Token tracking in ExecutionResult
- [x] Hard verification Tier 1 (syntax check + diff non-empty)
- [x] Heuristic context compression (3-stage, no LLM calls)
- [x] 369 tests passing

### Phase 2: Research Instrument (Current — ~3 weeks)

**Step 1: Code cleanup (2-3 days)**
- [x] Cut interactive CLI, replan loop, Anthropic adapter, recall_history, CI tiers
- [x] Simplify to headless-only entry point
- [x] Verify core pipeline still works: orient → plan → execute → JSONL

**Step 2: Trajectory schema + classifier (3-5 days)**
- [x] Define strict JSONL trajectory schema (`pare/trajectory/schema.py`)
- [x] Implement trajectory classifier — pure rules, zero LLM (`pare/trajectory/classifier.py`)
- [x] Implement recovery pattern detector with L1/L2/L3 levels (`pare/trajectory/recovery_detector.py`)
- [x] Unit tests for all classification edge cases

**Step 3: Git exploitation defender (2-3 days)**
- [x] Implement history rewrite: cut commits after issue date (`pare/defender/git_exploitation_defender.py`)
- [x] Test on sample SWE-bench instances

**Step 4: SFT exporter + token budget sampler (3-5 days)**
- [x] SFT export: trajectory → OpenAI messages format (`pare/export/sft_exporter.py`)
- [x] Token budget matching sampler (`pare/curation/sampler.py`)
- [x] Verify exported format works with LoRA training script

**Step 5: Hard verification Tier 2 (1-2 days)**
- [ ] Opt-in `test_command` from config, run after step completion
- [ ] Wire into trajectory classification (weakly_verified vs fully_verified)

### Phase 3: Pilot Experiment (~1 week)

- [ ] Generate 20 trajectories (diverse SWE-bench instances)
- [ ] Run classifier → get label distribution
- [ ] Manually label same 20 trajectories → compute κ against classifier
- [ ] **Go/No-Go:** κ ≥ 0.7 AND recovery trajectory ratio ≥ 15%
  - If recovery ratio < 15%: consider active injection strategy (intentionally degrade agent to trigger more recoveries)
- [ ] Document pilot findings

### Phase 4: Full Experiment (~3-4 weeks)

- [ ] Generate 750 trajectories (250 instances × 3 seeds)
- [ ] Classify all trajectories
- [ ] Sample 4 experimental groups (token-budget matched)
- [ ] Run LoRA fine-tuning (4 groups × same config)
- [ ] Evaluate on held-out SWE-bench + HumanEval+
- [ ] Statistical analysis (McNemar's test, bootstrap CI)

### Phase 5: Writing (~2-3 weeks)

- [ ] Results analysis and visualization
- [ ] Paper draft (or thesis chapter)
- [ ] Open-source release: code, trajectories, training configs, evaluation scripts

---

## 9. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| H2 null result (recovery doesn't help) | Main hypothesis fails | Pre-planned fallback framings: trajectory length bias, recovery type heterogeneity, small model capacity |
| Too few recovery trajectories (< 15%) | Underpowered experiment | Active injection: intentionally weaken agent (remove Orient, reduce budget) to trigger more failures |
| Teacher model too good (few failures) | Same as above | Use weaker teacher (Qwen-7B) or harder SWE-bench instances |
| Git exploitation contaminates data | Invalid experimental results | Defender runs before every trajectory, verified in pilot |
| Classifier disagrees with human labels | Classification methodology invalid | Pilot study catches this, revise rules before scaling |
| LoRA training doesn't converge | No results | Use proven hyperparams, start with known-good baseline |
| Compute budget insufficient | Can't run all experiments | Prioritize H2 (2 groups: clean-only vs mixed), defer ablations |

---

## 10. Open Science Commitment

All artifacts will be published:
- Full source code (Pare + experiment scripts)
- All trajectory JSONL files (raw + classified)
- Training configs and LoRA checkpoints
- Evaluation scripts and raw results
- Pilot study human labels and inter-annotator data

Negative results will be reported honestly. If H2 doesn't hold, we analyze why and publish the analysis.

---

## Appendix A: Implementation Status (Existing Code)

### Modules retained from product phase

| Module | Location | Lines | Tests | Status |
|---|---|---|---|---|
| LLM adapter | `pare/llm/` | ~600 | 42 | Stable |
| Tool system | `pare/tools/` | ~500 | 30 | Stable |
| Agent core | `pare/agent/` | ~800 | 50 | Needs simplification (cut replan) |
| Git checkpoint | `pare/sandbox/` | ~250 | 19 | Stable |
| Context manager | `pare/context/` | ~400 | 25 | Stable |
| Headless mode | `pare/cli/headless.py` | ~130 | 13 | Stable |
| Telemetry | `pare/telemetry.py` | ~100 | 8 | Stable |

### Modules to add

| Module | Location | Purpose |
|---|---|---|
| Trajectory schema | `pare/trajectory/schema.py` | Strict JSONL definition for trajectory data |
| Classifier | `pare/trajectory/classifier.py` | Rule-based trajectory labeling |
| Recovery detector | `pare/trajectory/recovery_detector.py` | L1/L2/L3 recovery pattern identification |
| Git defender | `pare/defender/git_exploitation_defender.py` | History rewrite for data integrity |
| SFT exporter | `pare/export/sft_exporter.py` | Trajectory → OpenAI messages format |
| Token sampler | `pare/curation/sampler.py` | Token-budget-matched group sampling |
| Experiment scripts | `experiments/` | End-to-end experiment automation |

### Target directory structure

```
pare/
├── pare/
│   ├── llm/                  # OpenAI-compatible adapter only
│   ├── tools/                # bash, file_read, file_edit, file_create, search
│   ├── agent/                # orient, plan, execute (no replan)
│   ├── sandbox/              # git_checkpoint
│   ├── context/              # memory, history, compactor
│   ├── trajectory/           # schema, classifier, recovery_detector
│   ├── curation/             # sampler (token budget matching)
│   ├── defender/             # git_exploitation_defender
│   └── export/               # sft_exporter
├── experiments/              # experiment scripts
│   ├── generate_trajectories.py
│   ├── classify_trajectories.py
│   ├── sample_groups.py
│   ├── run_sft_training.py
│   └── evaluate_models.py
├── configs/                  # per-experiment YAML configs
└── tests/
```

### Dependencies (minimal)

```toml
dependencies = [
    "openai>=1.50.0",    # LLM adapter (OpenAI-compatible APIs)
    "anyio>=4.0",        # Async runtime
    "tiktoken>=0.7.0",   # Token counting
]
```

3 runtime dependencies. Rich, prompt_toolkit, anthropic removed.
