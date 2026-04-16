# Pare — Research Project Plan

**Author:** Chenz
**Date:** April 2026
**Status:** Infrastructure built, taxonomy pivot to tool-call-centric Liu et al. framework
**Language:** Python 3.12+

> **Internal note:** This plan has pivoted from step-level rollback-based recovery taxonomy to tool-call-level intra-step error-correction pattern detection, grounded in Liu et al. (2025) [1]. All references to git rollback as a recovery mechanism have been removed. Pare's git checkpoint module is retained as audit-only infrastructure. Recovery is defined as an observable pattern in the tool-call sequence within a ReAct turn, not as an orchestrator-level retry.

---

## 1. Research Question & Hypotheses

### 1.1 Core question

**Does SFT on trajectories containing intra-step tool-call-level error-correction patterns transfer self-correction capability to student models, measured by conditional recovery rate under controlled failure injection?**

### 1.2 Hypotheses

| # | Hypothesis | Claim | Priority |
|---|---|---|---|
| H1 | Verification-based filtering (Tier 1+2 pass) improves SFT data quality independent of recovery structure | Filtering matters | Secondary |
| **H2** | **Trajectories containing tool-call-level error-correction patterns (error signal followed by targeted correction within the same turn sequence) produce student models with higher conditional recovery rate than one-shot success trajectories, when controlled for token budget** | **Recovery experience teaches self-correction** | **Primary** |
| H3 | Toxic trajectories (containing Liu et al.'s C2 Premature Success or B2.1 Logic Error labels) degrade model performance when included in SFT | Data quality > data quantity | Secondary |

H2 is the main hypothesis. H1 and H3 provide supporting evidence for the data curation methodology.

### 1.3 Why this matters

The failure taxonomy for coding agents has been established by Liu et al. [1], who categorize 9 failure types across 3 stages (orientation, execution, verification) from a descriptive analysis of SWE-bench trajectories. Majgaonkar et al. [2] and Mehtiyev & Assuncao [3] further characterize behavioral differences between success and failure trajectories at scale (9,374 trajectories in [3]), identifying patterns like "context-before-edit" as predictive of success.

However, these analyses remain descriptive — they classify failures but do not study whether exposure to failure-correction patterns during training transfers self-correction capability to student models. On the prescriptive side, PALADIN [11] demonstrates single-step self-correction on ToolBench's API-level failure space via retrieval from an exemplar library. SCoRe [9] trains self-correction via RL on the model's own output distribution, but studies intrinsic self-correction (internal reconsideration without external tool feedback).

**Our contribution occupies the unexplored intersection:** taking Liu et al.'s descriptive failure taxonomy and using it prescriptively — as a curation signal for SFT trajectory selection — in SWE-bench's code-level failure space with multi-step agentic recovery (external tool feedback loops). This is distinct from PALADIN (different failure space, different recovery mechanism) and from SCoRe (agentic correction via tool feedback vs. intrinsic reconsideration).

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
│  Git Checkpoint (audit-only, not for recovery)   │
├──────────────────────────────────────────────────┤
│  Verification: Tier 1 (syntax + diff) · Tier 2   │
├──────────────────────────────────────────────────┤
│  Execution Environment (SWE-bench Docker)        │
│  swebench.harness manages container lifecycle    │
│  Agent tools execute inside per-instance container│
├──────────────────────────────────────────────────┤
│  Trajectory JSONL (tool-call-centric, turn_id)   │
└──────────────────────────────────────────────────┘
```

**Execution environment:** Each SWE-bench instance runs in its own Docker container provided by `swebench.harness`. This handles Python/dependency version differences per instance. Pare does not build or manage Docker images — it delegates container lifecycle to the official harness. Git checkpoint operates *inside* the container's repo for per-step state auditing, while Docker provides runtime isolation (per-instance dependencies). Two layers, two concerns.

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
| `StepAttempt.rolled_back` field | Never set to True anywhere in codebase — dead state |
| Step-based `recovery_detector.py` | Replaced by tool-call-centric detector v2 |
| L1/L2/L3 step-rollback classification | Replaced by tool-call-level L1 local correction / L2 tactical switch / L3 exploratory recovery |

### 2.4 What is kept and strengthened

| Module | Role in research |
|---|---|
| Git checkpoint | Per-step state audit for data integrity verification; **not** used for recovery pattern detection |
| Two-tier hard verification | Core filtering for trajectory classification |
| Session JSONL | Raw trajectory data — schema must be strict |
| Orient → Plan → Execute | Trajectory generation strategy (ablatable: plan vs flat) |
| Guardrails | Budget enforcement ensures deterministic step boundaries |
| Heuristic compression | Keeps long tasks running without burning tokens |
| OpenAI-compatible adapter | Talks to DeepSeek/Qwen/any teacher model |

---

## 3. Trajectory Taxonomy

Based on Liu et al. (2025) [1] 3-stage, 9-category failure taxonomy. Pare implements automated detection for 8 categories (excluding B1.3 Redundant Implementation as orthogonal to this study). Each trajectory receives one or more Liu et al. category labels, plus a trajectory-level boolean property `contains_recovery`.

### 3.1 Liu et al. Taxonomy Adoption

#### 3.1.1 Automated failure category detection

| Stage | Category | Detection rule | Automation level |
|---|---|---|---|
| **A. Orientation** | A1: Missing Context | Agent edits file without prior `file_read` on that file in the same turn sequence | Fully automated |
| | A2: Mislocalization | Agent edits file F, but error/test failure references file G ≠ F | Fully automated |
| **B. Execution** | B1.1: Incomplete Fix | Final diff touches fewer files/hunks than the gold patch (when gold available) | Fully automated |
| | B1.2: Insufficient Testing | Trajectory contains no `bash` tool call matching test runner keywords (`pytest`/`unittest`/`python -m`/`manage.py test`/`tox`/`nose`) despite Tier 2 being configured | Fully automated |
| | B2.1: Logic Error | Tier 2 test fails with assertion error (not syntax/import) | Fully automated |
| | B2.2: Syntax Error | `compile()` fails on any edited `.py` file | Fully automated |
| **C. Verification** | C1: False Negative | Agent reports failure but Tier 2 actually passes (detected at trajectory end only) | Fully automated |
| | C2: Premature Success | Agent's final turn claims completion, but Tier 1 verification fails (empty diff or syntax error) | Fully automated (last turn only) |

**Design notes:**
- C2 detection is restricted to the agent's final turn to avoid false positives on intermediate "looks good" statements
- A1 is simplified from Liu et al.'s original (which includes import closure analysis) — we check only file_read before file_edit
- When both B2.1 and C2 could apply, C2 takes priority (more severe: agent is deluded about success)
- B1.3 (Redundant Implementation) is excluded as it requires semantic understanding beyond rule-based detection

#### 3.1.2 Trajectory-level outcome labels

| Label | Definition |
|---|---|
| **toxic** | Contains C2 (Premature Success) or B2.2 (Syntax Error in final state) |
| **failed** | Tier 2 fails, no recovery to passing state |
| **weakly_verified** | Tier 1 passes, Tier 2 not configured or not run |
| **verified_one_shot** | Tier 1 + Tier 2 pass, `contains_recovery == False` |
| **verified_with_recovery** | Tier 1 + Tier 2 pass, `contains_recovery == True` |

#### 3.1.3 Trajectory-level recovery property

A trajectory has `contains_recovery = True` iff:
1. It contains at least one `ToolCallEvent` with `error_signal != NONE`
2. A subsequent `ToolCallEvent` with strictly greater `(turn_id, call_index_in_turn)` tuple than the error event (i.e., either in a later turn, or later in the same turn) makes a targeted correction (same `target_file`, or same `tool_name` with materially different params)
3. The trajectory terminates with Tier 1 + Tier 2 verification pass

### 3.2 Recovery levels (tool-call granularity)

| Level | Pattern | Example | Theoretical basis |
|---|---|---|---|
| **L1: Local correction** | Same `tool_name`, same `target_file`, params differ only in the correction (typo fix, re-edit same section) | `file_edit` fails with syntax error → next `file_edit` fixes the typo | Simple retry |
| **L2: Tactical switch** | Different `tool_name` or different `target_file`, but addressing the same `error_signal` | `bash find` fails → agent uses `search` tool instead; or `file_edit` on wrong file → edits correct file | Strategy adaptation |
| **L3: Exploratory recovery** | Error followed by investigation sequence (multiple `file_read`/`search` calls) before correction | Test failure → agent reads 3 files to understand the bug → then edits | Metacognitive recovery; supported by Mehtiyev et al. [3] finding that context-gathering-before-edit predicts success |

Recovery level is assigned to each `(error_event, correction_event)` pair, not to the trajectory as a whole. A trajectory may contain multiple recovery events at different levels.

### 3.3 Classification pipeline

Execution order for each trajectory:

1. **Convert** raw trajectory to `ToolCallEvent` sequence (schema v2 with `turn_id` boundaries)
2. **Extract** `error_signal` for each `ToolCallEvent` via `error_signal_extractor` (rule-based regex/parser on tool result content)
3. **Classify** Liu et al. 8 categories via `classifier_liu.py` (one independent function per category, returns bool)
4. **Detect** recovery events via `recovery_detector_v2` — find error-correction pairs, assign L1/L2/L3, set `contains_recovery`
5. **Assign** trajectory-level outcome label (§3.1.2) based on verification results + `contains_recovery`

All classification is deterministic. No LLM calls.

### 3.4 Error signal taxonomy

The `error_signal` enum, extracted from `tool_result` content by rule-based regex:

| Signal | Detection pattern |
|---|---|
| `NONE` | `tool_result.success == True` and no error patterns in output |
| `SYNTAX_ERROR` | Python `SyntaxError` / `IndentationError` in output, or `compile()` failure |
| `TEST_FAILURE` | Must satisfy AND: (a) `tool_name == "bash"`, (b) `params` contains test runner keyword (`pytest` / `unittest` / `python -m` / `manage.py test` / `tox` / `nose`), (c) `result_content` matches `FAILED` / `FAIL:` / `AssertionError` / `errors=\d+` / non-zero exit code |
| `RUNTIME_ERROR` | `Traceback` / `Exception` / `Error:` patterns (excluding syntax) |
| `COMMAND_NOT_FOUND` | `command not found` / `No such file or directory` / exit code 127 |
| `EMPTY_DIFF` | `git diff` shows no changes after write tool calls |
| `TIMEOUT` | Tool execution exceeded time limit |
| `BLOCKED` | Guardrail blocked the tool call |
| `OTHER` | `tool_result.success == False` but no pattern match |

**Pilot validation required:** Error signal extractor precision and recall must both exceed 80% on pilot trajectories (per-tool-call manual inspection). If recall is low, expand regex patterns or add structured parsers.

---

## 4. Dataset Construction Protocol

### 4.1 Source tasks

SWE-bench Lite (300 instances). Split:
- **Training pool:** 250 instances (trajectory generation + SFT)
- **Held-out evaluation:** 50 instances (never seen during trajectory generation)

### 4.2 Trajectory generation

For each training instance:
1. `swebench.harness` creates a Docker container with the correct repo state and Python/dependency versions
2. Git exploitation defender rewrites history inside the container (cut commits after issue date)
3. Pare agent runs inside the container: Orient → Plan → Execute (no replan)
4. Git checkpoint tracks per-step SHAs inside the container's repo
5. On completion: Tier 1 + Tier 2 verification, trajectory JSONL extracted
6. Container is destroyed

Per-instance parameters:
- Model: DeepSeek-V3 (or Qwen-2.5-Coder-32B) as teacher
- Seeds: 3 runs per instance (seed 0, 1, 2) → up to 750 raw trajectories
- Each tool call is recorded as a `ToolCallEvent` with `turn_id`, enabling tool-call-level pattern analysis downstream

### 4.3 Classification & filtering

1. Convert all 750 trajectories to `ToolCallEvent` sequences
2. Run `error_signal_extractor` on each tool call
3. Run `classifier_liu.py` → Liu et al. category labels per trajectory
4. Run `recovery_detector_v2` → `contains_recovery` + L1/L2/L3 per recovery event
5. Assign trajectory-level outcome labels (§3.1.2)
6. Discard toxic trajectories entirely
7. Pool the rest by outcome label

### 4.4 Experimental groups

Groups are split into two modules so each module varies exactly one independent variable.

**Module A — H2 (primary): Does recovery experience teach self-correction?**

All groups use only **fully_verified** trajectories. The sole variable is recovery trajectory proportion, based on `contains_recovery` property.

| Group | Composition | Variable |
|---|---|---|
| **A1: Clean-only** | `contains_recovery == False` fully_verified trajectories | Baseline (0% recovery) |
| **A2: Mixed** | Natural ratio of `verified_one_shot` + `verified_with_recovery` | Natural recovery ratio |
| **A3: Recovery-enriched** | 2× natural ratio via oversampling `verified_with_recovery` | Sensitivity test |

**Module B — H1/H3 (secondary): Does verification quality matter?**

Baseline is A1 (Clean-only, fully_verified). Each group adds one type of lower-quality data. **All groups use only one_shot_success trajectories (no recovery patterns)** — this prevents H2's recovery effect from contaminating Module B's verification-tier comparison.

| Group | Composition | Variable |
|---|---|---|
| **A1 (shared baseline)** | Only fully_verified one_shot_success | Baseline |
| **B1: +Weakly verified** | A1 + weakly_verified one_shot_success (no recovery) | Verification tier effect (H1) |
| **B2: +Toxic 10%** | A1 + 10% toxic (labeled C2 Premature Success per Liu et al.) | Low toxic contamination (H3) |
| **B3: +Toxic 30%** | A1 + 30% toxic (labeled C2 Premature Success per Liu et al.) | High toxic contamination (H3) |

**Critical control: instance-matched sampling & token budget matching.** Within each module, groups are sampled so they are **instance-matched** (each group draws from the exact same cross-section of SWE-bench instances) to eliminate task difficulty as a confounding variable. Additionally, total training tokens are matched (±5%). This isolates the effect of trajectory composition from both training compute and instance complexity. Module A and Module B do not need to match each other (different hypotheses). Module A isolates recovery effect; Module B isolates verification tier effect. No variable crosses modules.

**Module B data feasibility risk:** Toxic trajectories (C2 Premature Success) are typically very short (3-5 tool calls) because the agent declares success early. A1 verified_one_shot trajectories are much longer (15-30 tool calls). This means "30% toxic by token budget" may require more toxic samples than exist. Phase 3.3 must include a toxic sample feasibility check: if total toxic one_shot tokens < 35% of A1's token budget, B3 automatically downgrades to a single "10% toxic" group, and the 10%/30% dose-response comparison is dropped. Alternatively, B3 may use sample-count matching instead of token matching, with an explicit departure noted in the paper limitations.

### 4.5 SFT export format

Each trajectory → OpenAI messages format (system + user + assistant turns with tool calls and results). Compatible with standard LoRA training scripts and HuggingFace chat templates.

---

## 5. Experimental Design

### 5.1 Independent variables

- **Module A (H2):** Recovery trajectory proportion in fully_verified SFT data (0% / natural / 2× natural)
- **Module B (H1/H3):** Verification quality of SFT data (fully_verified baseline vs. +weakly_verified / +toxic)

### 5.2 Dependent variables

| Metric | Evaluation set | What it measures |
|---|---|---|
| Conditional recovery rate under controlled failure injection | 50 held-out instances | **Primary metric for H2.** All student models face identical injected failures (pre-sampled real error messages from training trajectories). Isolates recovery capability from task difficulty variation. See §6.5. |
| SWE-bench Lite pass rate | 50 held-out instances | In-distribution coding ability (sanity check) |
| HumanEval+ pass rate | Full HumanEval+ | Out-of-distribution generalization |
| Natural recovery rate | 50 held-out (among initially-failed attempts) | Ecological self-correction signal (secondary to controlled rate) |
| Token efficiency | 50 held-out | Tokens per resolved instance |

### 5.3 Fixed variables (controlled)

| Variable | Setting | Justification |
|---|---|---|
| Base model | Qwen-2.5-Coder-1.5B | Smaller capacity maximizes the capability gap vs teacher, making self-correction SFT transfer effects more visible |
| LoRA config | r=16, alpha=32, dropout=0.05 | Standard settings |
| Training tokens | Matched across groups (±5%) | Isolate composition effect (justified by Mehtiyev et al. [3] length-confounding finding) |
| Epochs | 3 | Standard for SFT |
| Eval model | Same checkpoint, greedy decoding | Reproducible |
| Teacher model | Fixed (DeepSeek-V3 or Qwen-32B) | Same teacher for all trajectories |
| Seed | Fixed per run | Reproducible |

### 5.4 Ablation studies

1. **Recovery level:** Compare L1-only vs L2/L3-only in A3 (recovery-enriched) group → tests whether exploratory recovery (L3) is more valuable than local retry (L1)
2. **Taxonomy category ablation:** Separately exclude each Liu et al. failure category from training data, observe effect on student recovery rate
3. **Plan vs flat mode:** Generate trajectories with both, compare quality metrics → Orient→Plan value

---

## 6. Evaluation Metrics

### 6.1 Primary

- **Conditional recovery rate under controlled failure injection** (§6.5) — directly tests H2's causal claim with identical failure conditions across all groups

### 6.2 Secondary

- **SWE-bench Lite pass@1** on held-out 50 instances (3 runs, median) — sanity check
- **Natural recovery rate** (§6.4) — ecological validity, but potentially incomparable denominators
- **HumanEval+ pass@1** (OOD generalization)
- **Tokens per resolved instance** (efficiency)

### 6.3 Statistical tests

- McNemar's test for pass rate differences between groups
- Bootstrap confidence intervals (95%) for all metrics
- Report effect sizes, not just p-values

### 6.4 Evaluation protocol: student model as agent

Recovery rate (the core H2 metric) cannot be measured by pass@1 alone — it requires observing whether the student model can detect failure and course-correct within an agentic loop. Therefore:

**Evaluation harness:** Each fine-tuned student model is loaded into Pare and runs the same Orient → Plan → Execute loop used during trajectory generation, with the same tools and guardrails. The only difference is the LLM backend (student model instead of teacher).

**Procedure per held-out instance:**
1. Student model runs in Pare with greedy decoding (temperature=0), fixed seed
2. Full trajectory JSONL is recorded (same tool-call-centric schema as training data)
3. Classifier labels the eval trajectory (Liu et al. categories + recovery detection)
4. Tier 1 + Tier 2 verification determines pass/fail

**Natural recovery rate measurement:** For each group's student model, count instances where:
- The model initially encounters a failure (verification fail, error in tool output, etc.)
- The model subsequently recovers and produces a passing solution
- Recovery rate = (recovered instances) / (instances with initial failure)

**Practical requirement:** The student model (Qwen-2.5-Coder-1.5B + LoRA) must support tool_use via OpenAI-compatible API (e.g., served via vLLM with `--enable-auto-tool-choice`). If native tool_use is unreliable, fall back to text-format tool calls with model profile `supports_native_tool_use = false`.

### 6.5 Controlled failure injection protocol

**Problem:** If we measure recovery rate only from naturally-occurring failures, different groups' student models will have different failure base rates. A stronger model fails on fewer instances → smaller denominator → recovery rate becomes incomparable across groups (small-sample noise, different failure populations).

**Solution: controlled perturbation.** We pre-sample real error messages from training-set trajectories of the same repository family. During student model evaluation, at a predetermined agent step (e.g., after the first `file_edit`), we replace the real `tool_result` with the sampled error message. All student model groups face identical injected failures.

**Procedure:**
1. Select ~20 held-out instances where the **teacher model** (not any student model) succeeds on the first attempt — this defines a "baseline-solvable" instance set that is neutral to all student groups, and can be frozen during Phase 3 before any student training begins
2. For each instance, inject a real error message (sampled from training trajectories of the same repo) at a predetermined step. **Implementation requirement:** This must be achieved by constructing and injecting a structurally complete `ToolCallEvent` object into the context history, not via direct string replacement of prompt text, to avoid corrupting the trajectory schema.
3. Allow the student model N additional steps (e.g., 10) to recover
4. Score: did the model diagnose the injected failure and produce a corrected edit that passes real verification?
5. Conditional recovery rate = (instances with successful correction after injection) / (total injected instances)

**Why teacher model for instance selection:** Using A1 student to select instances would create two problems: (a) serial dependency — A1 must finish training before any injection eval can start; (b) selection bias — instances where A1 succeeds are systematically easier, inflating recovery rates for A2/A3. Using the teacher model breaks both: injection instances are frozen in Phase 3, and difficulty is defined by a model external to all experimental groups.

**Why this works:** Every group's student model faces the exact same failure at the exact same point. The only variable is the model's learned recovery behavior. This is standard controlled perturbation — analogous to adversarial robustness evaluation but for self-correction.

Implementation details deferred to `experiments/failure_injection.py`; design phase scheduled before Phase 4.

---

## 7. Methodology Validity Safeguards

### 7.1 Git exploitation defense

This is a data integrity safeguard, not a research novelty claim.

**Problem:** Teacher model may "cheat" by accessing git history that contains the fix. If the SWE-bench instance was created from a real PR, commits after the issue date may contain the solution.

**Defense:** Before trajectory generation, rewrite the repo's git history to remove all commits after the issue creation date. Implemented in `pare/defender/git_exploitation_defender.py`.

### 7.2 Data contamination prevention

- Held-out evaluation instances are never used for trajectory generation
- Teacher model and student model are different (teacher generates, student is fine-tuned)
- HumanEval+ provides OOD check independent of SWE-bench

### 7.3 Classifier validity

Classification is based on deterministic rules derived from Liu et al. (2025) [1] 9-category taxonomy, of which we automate 8 categories (excluding B1.3 Redundant Implementation as orthogonal to this study). Rules have κ = 1.0 by construction (no subjectivity in execution).

**Pilot study (two-annotator protocol):**
1. Author and one independent annotator each label 20 pilot trajectories using Liu et al.'s original taxonomy definitions (not our automation rules)
2. Compute inter-annotator Cohen's κ between the two humans
3. Then compare both human labelings against rule-based classifier output; disagreements trigger rule refinement
4. This validates that our rules faithfully operationalize Liu et al.'s taxonomy

- Target: inter-annotator κ ≥ 0.7 (substantial agreement) → classification rules are operationalizable
- If κ < 0.7 → ambiguous taxonomy definitions or edge cases; revise classification rules and re-label before scaling

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
| Trajectory length | Token budget matching (justified by Mehtiyev et al. [3] length-confounding finding) |

### 7.5 Related work positioning

**vs Liu et al. [1]:** We adopt their 3-stage 9-category failure taxonomy and extend it in two ways: (a) automated rule-based detection for 8 of 9 categories, and (b) prescriptive use for SFT data curation. Liu et al.'s analysis is descriptive (classifying failures post-hoc); we use their categories as curation signals to test whether failure-type composition affects SFT transfer.

**vs SCoRe [9]:** SCoRe studies intrinsic self-correction — the model reconsidering its own output without external feedback, trained via RL on the model's own distribution. We study agentic correction — the model responding to external tool feedback (error messages, test failures) within a ReAct loop. Different information sources (internal reconsideration vs. external observation), different granularities (single-turn vs. multi-turn with tool calls), orthogonal findings. Our approach is SFT-based (cheaper, more data-efficient) while SCoRe requires on-policy RL.

**vs PALADIN [11]:** Disjoint failure space (API-level on ToolBench vs. code-level on SWE-bench) and disjoint recovery mechanisms (retrieval from an exemplar library vs. within-file reasoning from tool feedback). PALADIN's correction is single-step (retrieve and retry); ours is multi-step (investigate, diagnose, edit). Our work can be seen as testing whether PALADIN's finding ("exposure to correction patterns helps") generalizes from API calls to code editing.

**vs Mehtiyev et al. [3]:** We use their two key findings as design constraints: (a) the "context-before-edit" pattern as theoretical support for L3 exploratory recovery's value — agents that investigate before editing succeed more, so L3 recovery trajectories may be the most valuable SFT signal; (b) their trajectory-length confounding finding as justification for our token-budget matching across experimental groups.

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
- [x] Trajectory schema v1 + classifier v1 + recovery detector v1 (step-based, now deprecated)
- [x] Git exploitation defender
- [x] SFT exporter + token budget sampler
- [x] Hard verification Tier 2
- [x] Experiment scripts (generate, classify, materialize workdirs)

### Phase 2: Tool-Call-Centric Pivot ✅ COMPLETE (2026-04-16)

**Phase 2.1: ToolCallEvent schema + turn_id instrumentation (2-3 days)** ✅
- [x] Define `ToolCallEvent` dataclass: `turn_id`, `call_index_in_turn`, `global_index` (monotonic global index to support temporal comparison across turn boundaries), `tool_name`, `params`, `params_hash`, `target_file`, `result_success`, `result_content`, `error_signal`, `timestamp`
- [x] Instrument executor to emit `ToolCallEvent` records per tool call
- [x] Wire into trajectory JSONL output (alongside existing step-level data)
- [x] Tests for schema validation

**Phase 2.2: Error signal extractor (2-3 days)** ✅
- [x] Implement `error_signal_extractor.py`: rule-based regex/parser for 9 error signal types (§3.4)
- [x] Cover: Python traceback, pytest output, bash errors, compile failures, empty diff, guardrail blocks
- [x] Unit tests with real error samples from pilot session logs
- [x] Validate precision + recall > 80% on pilot data

**Phase 2.3: Recovery detector v2 (2-3 days)** ✅
- [x] Implement `recovery_detector_v2.py`: find error-correction pairs in `ToolCallEvent` sequence
- [x] Assign L1/L2/L3 levels per pair
- [x] Set trajectory-level `contains_recovery` boolean
- [x] Tests for all recovery level patterns

**Phase 2.4a: Liu et al. classifier — core 4 categories (2-3 days)** ✅
- [x] Implement `classifier_liu.py` with priority categories: **B2.1** (Logic Error), **B2.2** (Syntax Error), **C1** (False Negative), **C2** (Premature Success)
- [x] These 4 are sufficient for H2 (`contains_recovery` detection) and H3 (toxic label = C2 + B2.2)
- [x] Each function takes `ToolCallEvent` sequence + verification results → bool
- [x] Integration test: classify pilot trajectories

**Phase 2.4b: Liu et al. classifier — extended 4 categories (3-4 days, can overlap with Phase 3)** ✅
- [x] Implement: **A1** (Missing Context — requires file_read-before-edit tracking), **A2** (Mislocalization — requires parsing error file refs from tool_result), **B1.1** (Incomplete Fix — requires gold patch diff comparison), **B1.2** (Insufficient Testing — semi-automated, needs manual review on n=50)
- [x] A1/A2/B1.1 each need an auxiliary parser (issue text file mentions, unified diff, error file extraction)
- [x] These are paper-richness contributions, not H2 critical path
- [x] Can be completed during or after Phase 3 pilot without blocking experiment
- Note: B1.1 detector accepts `final_diff` / `gold_patch` as optional kwargs; pipeline wiring to an SWE-bench gold-patch source is deferred until Phase 3 instance selection.

**Phase 2.5: Integration (1-2 days)** ✅
- [x] Wire new classifier (core 4 categories from 2.4a) into `classify_trajectories.py` experiment script
- [x] Update `_build_trajectory_record()` in headless.py to include `ToolCallEvent` data
- [x] Verify end-to-end: generate → classify → label distribution
- [x] Phase 3 can start after 2.5 completes; Phase 2.4b runs in parallel with Phase 3

**Phase 2.6: Cleanup (1 day)** ✅
- [x] Deprecate (not delete) `recovery_detector.py` v1 and `classifier.py` v1 — emit `DeprecationWarning` on use, kept as regression harness for `pare.curation.sampler`
- [x] Remove `StepAttempt.rolled_back` field (dead state — never set to True by runtime); `rolled_back` key tolerated on JSONL decode for backward compat
- [x] Update this plan to mark Phase 2 as done

### Phase 3: Pilot Experiment (~1-2 weeks)

**Phase 3.1: Single trajectory deep inspection (1 day)**
- [ ] Run 1 trajectory on a simple SWE-bench instance (sufficient budget: per-step=15, total=60)
- [ ] Print raw session JSONL, inspect every tool call and result manually
- [ ] Identify: what do real error messages look like? What recovery patterns actually occur?
- [ ] **Go/No-Go:** Agent can complete at least read → edit → verify cycle

**Phase 3.2: Error extractor validation (2-3 days)**
- [ ] Run 5 trajectories on diverse instances
- [ ] Run error_signal_extractor on all tool calls
- [ ] Manually check every extraction: precision and recall per error type
- [ ] **Go/No-Go:** Precision > 80% AND recall > 80% across error types

**Phase 3.3: Taxonomy validation (3-5 days)**
- [ ] Run 20 trajectories (diverse SWE-bench instances, sufficient budget)
- [ ] Run full classification pipeline (Liu categories + recovery detection)
- [ ] Two-annotator labeling: author + independent annotator label same 20
- [ ] Compute inter-annotator κ
- [ ] **Go/No-Go criteria (all must pass):**
  - κ ≥ 0.7 (inter-annotator agreement on Liu et al. categories)
  - Recovery trajectory ratio ≥ 15% (if < 15%: adopt controlled agent weakening, see below)
  - Module B toxic feasibility: estimate toxic one_shot token pool from pilot distribution. If projected total < 35% of A1 token budget at full scale, downgrade B3 now (see §4.4)

**Active injection trade-off (if natural recovery ratio < 15%):**

If pilot shows insufficient natural recovery trajectories, we adopt controlled agent weakening: reduced context budget, no Orient phase, and/or lower max_iterations per step. This increases failure frequency and thus recovery opportunities.

However, this shifts the study from observational (natural agent behavior) to interventional (artificially constrained agent). Consequences:
- Trajectories generated under weakened conditions may not represent how frontier teacher models behave in practice
- Conclusions about recovery trajectory value would be qualified: "under resource-constrained agent settings"
- Paper limitation section will explicitly state: "Results are demonstrated under controlled agent weakening. Generalizability to frontier teacher models operating at full capacity requires further study."

This is an acceptable trade-off: a well-powered interventional study is more informative than an underpowered observational one. The controlled failure injection eval (§6.5) provides a complementary signal that is independent of how trajectories were generated.

### Phase 4: Full Experiment (~3-4 weeks)

**Pre-requisite: Controlled failure injection design (3-5 days)**
- [ ] Design and implement `experiments/failure_injection.py`
- [ ] Select injection points and error message sampling strategy
- [ ] Validate on 5 instances with baseline model
- [ ] **Must complete before starting full training runs**

**Main experiment:**
- [ ] Generate 750 trajectories (250 instances × 3 seeds)
- [ ] Classify all trajectories (full pipeline: ToolCallEvent → error signals → Liu categories → recovery detection)
- [ ] Sample experimental groups: Module A (3 groups) + Module B (3 groups, shared baseline)
- [ ] Run LoRA fine-tuning (6 groups × 3 seeds × same config)
- [ ] Evaluate: controlled failure injection + natural recovery rate + SWE-bench pass@1 + HumanEval+
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
| Compute budget insufficient | Can't run all experiments | Prioritize Module A only (3 groups), defer Module B and ablations. See §11 for budget estimate |
| Error signal extractor recall too low | Recovery events systematically missed | Pilot Phase 3.2 validates per-tool-call; target precision + recall > 80%. If recall is low, expand regex or add structured parsers. Must fix before Phase 4 |
| Controlled failure injection too complex | Delayed Phase 4 start | Dedicated 3-5 day design phase before full experiment. Not allowed to modify injection mechanism after experiment starts |

---

## 10. Open Science Commitment

All artifacts will be published:
- Full source code (Pare + experiment scripts)
- All trajectory JSONL files (raw + classified)
- Training configs and LoRA checkpoints
- Evaluation scripts and raw results
- Pilot study human labels and inter-annotator data
- Error signal extractor precision/recall analysis

Negative results will be reported honestly. If H2 doesn't hold, we analyze why and publish the analysis.

---

## 11. Compute & Cost Budget

### 11.1 Trajectory generation (Phase 4)

| Item | Estimate | Notes |
|---|---|---|
| Instances | 250 (SWE-bench Lite training pool) | |
| Seeds per instance | 3 | |
| Total trajectories | 750 | |
| Avg agent runtime per trajectory | ~5 min | Bounded ReAct, max ~30 tool calls |
| Total agent wall-clock | ~150 hours | Parallelizable across instances (estimate excludes container lifecycle overhead, actual wall-clock may be 2×+) |
| Teacher model API cost | ~$150–300 | DeepSeek-V3 at ~$0.20–0.40/trajectory (input+output tokens) |

### 11.2 SFT training (Phase 4)

| Item | Estimate | Notes |
|---|---|---|
| Base model | Qwen-2.5-Coder-1.5B | |
| LoRA config | r=16, alpha=32, dropout=0.05 | |
| GPU hours per run | 6–10 (A100 80GB) | ~3 epochs over trajectory data |
| Groups | 6 (Module A: 3, Module B: 3 with shared baseline = 5 unique) | |
| Seeds per group | 3 | |
| Total training runs | 15 unique (5 groups × 3 seeds) | A1 baseline shared |
| Total GPU hours | 90–150 (A100) | |

### 11.3 Evaluation (Phase 4)

| Item | Estimate | Notes |
|---|---|---|
| Eval instances | 50 held-out | |
| Student model runs per group | 50 × 3 seeds = 150 | Each runs Pare agent loop |
| Total eval runs | 150 × 5 groups = 750 | |
| Eval infra | vLLM serving 7B model, 1× A100 | Sequential or batched |
| HumanEval+ | Lightweight, ~1 hr per group | Standard code generation |

### 11.4 Total budget summary

| Resource | Estimate | Fallback (Module A only) |
|---|---|---|
| Teacher API cost | $150–300 | Same (all trajectories needed) |
| Training GPU hours | 90–150 A100-hrs | 45–75 (3 groups × 3 seeds) |
| Eval GPU hours | ~50 A100-hrs | ~25 |
| Total GPU hours | ~140–200 A100-hrs | ~70–100 |

**Action item:** Confirm GPU allocation with advisor before Phase 4. If budget is constrained, prioritize Module A (H2) and defer Module B ablations.

---

## Appendix A: Implementation Status

### Modules retained from product/infrastructure phase

| Module | Location | Status |
|---|---|---|
| LLM adapter | `pare/llm/` | Stable |
| Tool system | `pare/tools/` | Stable |
| Agent core | `pare/agent/` | Stable (replan removed) |
| Git checkpoint | `pare/sandbox/` | Stable (audit-only role) |
| Context manager | `pare/context/` | Stable |
| Headless mode | `pare/cli/headless.py` | Stable |
| Telemetry | `pare/telemetry.py` | Stable |
| Trajectory schema v1 | `pare/trajectory/schema.py` | Stable (will be extended, not replaced) |
| Git defender | `pare/defender/git_exploitation_defender.py` | Stable |
| SFT exporter | `pare/export/sft_exporter.py` | Stable |
| Token sampler | `pare/curation/sampler.py` | Stable |

### Modules to add (Phase 2)

| Module | Location | Purpose |
|---|---|---|
| ToolCallEvent schema | `pare/trajectory/schema_v2.py` | Tool-call-centric schema with `turn_id`, `error_signal`, `target_file` |
| Error signal extractor | `pare/trajectory/error_signal_extractor.py` | Rule-based error detection from tool result content |
| Recovery detector v2 | `pare/trajectory/recovery_detector_v2.py` | Tool-call-level error-correction pair detection, L1/L2/L3 |
| Liu et al. classifier | `pare/trajectory/classifier_liu.py` | 8 automated categories, one function per category |
| Classifier v2 | `pare/trajectory/classifier_v2.py` | Orchestrates error extraction → Liu categories → recovery → trajectory label |
| Failure injection | `experiments/failure_injection.py` | Controlled failure injection for eval (Phase 4 pre-req) |

### Modules to deprecate (not delete)

Status (Phase 2.6, 2026-04-16): v1 modules emit `DeprecationWarning` on use; `rolled_back` field removed from `StepAttempt`.

| Module | Location | Status |
|---|---|---|
| Recovery detector v1 | `pare/trajectory/recovery_detector.py` | Deprecated — `rolled_back` gate removed; now flags any failed→success pair. Kept only as regression harness. |
| Classifier v1 | `pare/trajectory/classifier.py` | Deprecated — `TrajectoryClassifier.__init__` warns. Still used by `pare.curation.sampler` pending Phase 3 rewrite. |
| `StepAttempt.rolled_back` field | `pare/trajectory/schema.py` | **Removed.** Field dropped from dataclass; decoder tolerates the key on legacy JSONL for backward compat. |

### Target directory structure

```
pare/
├── pare/
│   ├── llm/                  # OpenAI-compatible adapter only
│   ├── tools/                # bash, file_read, file_edit, file_create, search
│   ├── agent/                # orient, plan, execute (no replan)
│   ├── sandbox/              # git_checkpoint (audit-only)
│   ├── context/              # memory, history, compactor
│   ├── trajectory/           # schema_v2, error_signal_extractor, recovery_detector_v2,
│   │                         # classifier_liu, classifier_v2 (+ deprecated v1 modules)
│   ├── curation/             # sampler (token budget matching)
│   ├── defender/             # git_exploitation_defender
│   └── export/               # sft_exporter
├── experiments/              # experiment scripts
│   ├── generate_trajectories.py
│   ├── classify_trajectories.py
│   ├── sample_groups.py
│   ├── run_sft_training.py
│   ├── evaluate_models.py
│   └── failure_injection.py
├── configs/                  # per-experiment YAML configs
└── tests/
```

### Dependencies (minimal)

```toml
dependencies = [
    "openai>=1.50.0",    # LLM adapter (OpenAI-compatible APIs)
    "anyio>=4.0",        # Async runtime
    "tiktoken>=0.7.0",   # Token counting
    "swebench>=2.0",     # Docker harness for SWE-bench instances
]
```

4 runtime dependencies. Docker sandboxing is handled entirely by `swebench.harness` — no custom container management code in Pare.

---

## References

### Coding Agent Failure Analysis

[1] Simiao Liu, Fang Liu, Liehao Li, Xin Tan, Yinghao Zhu, Xiaoli Lian, Li Zhang. "An Empirical Study on Failures in Automated Issue Solving." arXiv:2509.13941 [cs.SE], September 2025.

[2] Oorja Majgaonkar et al. "Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories." ICSE 2026. arXiv:2511.00197 [cs.SE], October 2025.

[3] Tural Mehtiyev, Wesley Assuncao. "Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure." arXiv:2604.02547 [cs.SE], April 2026.

[4] Minh V. T. Thai et al. "SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios." arXiv:2512.18470 [cs.SE], December 2025.

[5] "SWE-Compass: Towards Unified Evaluation of Agentic Software Engineering." arXiv:2511.05459, November 2025.

### General Agent Failure Taxonomy

[6] "Why Do Multi-Agent LLM Systems Fail?" (MAST). arXiv:2503.13657. ICLR 2025 Building Trust Workshop.

[7] "Where LLM Agents Fail and How They can Learn From Failures." arXiv:2509.25370. UIUC.

[8] "Diagnosing AI Agent Failures from Execution Trajectories." (AgentRx). arXiv:2602.02475. Microsoft Research.

### Self-Correction & SFT

[9] Aviral Kumar et al. "Training Language Models to Self-Correct via Reinforcement Learning." (SCoRe). ICLR 2025. arXiv:2409.12917.

[10] "LEDEX: Training LLMs to Better Self-Debug and Explain Code." NeurIPS 2024.

[11] "PALADIN: Self-Correcting Language Model Agents to Cure Cascading Failures." arXiv:2509.25238.

### SWE-bench Infrastructure

[12] Carlos E. Jimenez, John Yang et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024.

[13] John Yang, Carlos E. Jimenez et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." NeurIPS 2024.

[14] Xingyao Wang et al. "OpenHands: An Open Platform for AI Software Developers as Generalist Agents." 2025.

[15] "SWE-smith: Scaling Data for Software Engineering Agents." 2025.

[16] You Wang, Michael Pradel, Zhongxin Liu. "Are 'Solved Issues' in SWE-bench Really Solved Correctly?" arXiv:2503.15223, March 2025.

### SFT Data Strategy

[17] Edward Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
