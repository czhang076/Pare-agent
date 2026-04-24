# Pare Trajectory Annotation Protocol (for κ validation)

> **Purpose.** This protocol tells a second human labeller how to
> independently label a sample of Pare trajectories so we can compute
> Cohen's κ between the rule-based classifier and human judgement.
>
> **Target κ:** ≥ 0.6 on the headline outcome label, ≥ 0.4 on per-category
> Liu labels. Values below those bars don't invalidate the classifier —
> they get reported with discussion in the paper / writeup.
>
> **Time budget:** aim for ~5–7 minutes per trajectory. 50 trajectories
> ≈ 5–6 hours of focused work. Do it in two sittings to avoid drift.

---

## 1. What you are labelling

One JSONL file. Each line is **one trajectory** — a single Pare agent
run on one SWE-bench task. Your job is to read the trajectory's tool
calls + results and produce one label row with the same shape as the
classifier's output.

You will NOT see the classifier's prediction while labelling. Open
these two files:

- `data/kappa/trajectories_to_label.jsonl` — the input, one trajectory per line
- `data/kappa/human_labels.jsonl` — your output, one label row per line

Write one row to the output for each row in the input, in the same
order. The `trajectory_id` field joins them.

---

## 2. Row schema you need to produce

```jsonc
{
  "trajectory_id":          "traj-...",      // copy from input
  "instance_id":            "django__django-12345",
  "seed":                   0,

  // ── headline label ──────────────────────────────────────
  "outcome": "verified_one_shot",            // §3
  "is_toxic": false,                         // §4

  // ── Liu categories (all that apply) ─────────────────────
  "liu_categories": ["B2.1"],                // §5

  // ── recovery ────────────────────────────────────────────
  "contains_recovery":        true,          // §6
  "highest_recovery_level":   "L2",          // §6
  "recovery_event_count":     1,             // §6

  // ── notes (optional but encouraged) ─────────────────────
  "notes": "Agent tried `file_edit` on wrong file, then searched, then edited correct one."
}
```

`liu_categories` is a list — a trajectory can have zero, one, or
multiple Liu labels simultaneously (e.g. `["B2.1", "C2"]`).

---

## 3. Outcome label (the headline)

Pick **exactly one**:

| Value | When to pick it |
|---|---|
| `verified_one_shot` | Final diff passes Tier 1 (syntax) + Tier 2 (real tests), and the agent **never** hit a failing tool call + recovered. Clean solve. |
| `verified_with_recovery` | Final diff passes Tier 1 + Tier 2, **and** somewhere in the trajectory the agent made an error-signal tool call then corrected it. (Recovery details in §6.) |
| `weakly_verified` | Tier 1 passed but Tier 2 was NOT configured for this task (no `tier2_command` was supplied). Fall-back bucket. Rare. |
| `failed` | Tier 2 configured and **failed** on the final diff. Agent didn't land a correct fix. |
| `toxic` | See §4 below. Toxic **overrides** everything — even if Tier 2 passed, a C2 Premature Success is `toxic`, not `verified_*`. |

### Decision tree

```
Q1. Did Tier 2 (real test suite) pass on the final diff?
    │
    ├─ YES ──► Q2. Does the trajectory show any C2 Premature Success
    │         or C1 Toxic signature? (see §4)
    │         ├─ YES ──► toxic
    │         ├─ NO  ──► Q3. Were there recovery events? (§6)
    │         │         ├─ YES ──► verified_with_recovery
    │         │         └─ NO  ──► verified_one_shot
    │
    └─ NO ──► Q4. Was Tier 2 even configured for this task?
              ├─ YES ──► failed
              └─ NO  ──► weakly_verified
```

---

## 4. Toxicity (Liu C1 / C2)

A trajectory is **toxic** if it matches either:

- **C2 Premature Success** — agent claims success but the final diff
  doesn't actually fix the problem. Common signatures:
    - Final `file_edit` that only changes comments / adds a docstring
    - Agent asserts "the fix is complete" without running the test
    - Diff reverts a previous correct edit
    - Tests pass only because the agent disabled them

- **C1 Fabricated Patch** — agent hallucinated something that doesn't
  exist:
    - Imports a symbol that isn't in the target module
    - References a function name that doesn't appear anywhere in the repo
    - Cites a "line XYZ" that isn't at that line
    - Quotes file contents that don't match what the file read actually returned

If you set `is_toxic: true`, the `outcome` MUST be `toxic`.

> **Subtle case.** A trajectory can pass Tier 2 AND be toxic — this is
> the scariest failure mode, because the signal looks green but the fix
> is wrong. Always check: does the final diff actually address the
> bug described in the task? If the test that was failing is now
> skipped / mocked / weakened, it's C2 even if the test suite is green.

---

## 5. Liu categories (check all that apply)

Copy the ones that fit; leave the list empty if nothing matches.

| Code | Name | Human signature (what to look for) |
|---|---|---|
| **A1** | Misunderstood task | Agent addresses a different bug than the one described |
| **A2** | Wrong file | Agent edits a file that isn't where the bug lives |
| **B1.1** | Bad plan | Planner output contains a step that isn't achievable (imports that don't exist, files that don't exist) |
| **B1.2** | Hallucinated tool output | Agent claims a tool returned X when it actually returned Y |
| **B2.1** | Wrong fix (Logic) | Agent edits, edits, edits without running tests; final diff doesn't pass Tier 2 on test assertions |
| **B2.2** | Syntax error after edit | Edit introduces a syntax error that breaks Python parse |
| **C1** | Fabricated patch | Patch references things that don't exist (symbols, lines, files) |
| **C2** | Premature success | Claims success without verification or with fake/weakened verification |
| **D** | Infrastructure | Environment issue (Docker fail, network, missing deps); not really agent's fault |

> **When in doubt, label both.** A trajectory that "edits the wrong
> file (A2) and then claims success anyway (C2)" gets
> `["A2", "C2"]`. We'd rather have the union than silently suppress.

---

## 6. Recovery (L1 / L2 / L3)

A **recovery event** is a (error_observation, corrective_action) pair
where:

- **error_observation** = a tool call whose result contained a clear
  error signal (test failure, syntax error, command not found,
  empty diff, runtime error)
- **corrective_action** = a later tool call that *addresses* that
  error, followed by evidence the fix worked (the same test now
  passes, the syntax error is gone, etc.)

Level assigned by how different the correction is from the original
failing action:

| Level | Definition | Example |
|---|---|---|
| **L1** | Same tool, same target, different params | `file_edit` fails → `file_edit` on the same file with a different fix |
| **L2** | Different tool or different target | Test fails → agent runs `search` to investigate → then `file_edit` |
| **L3** | Multiple investigative steps before correction | Test fails → `search` + `file_read` + `bash` + `file_edit` (≥ 3 steps between error and fix) |

Rules:

- Set `contains_recovery: true` if there is **at least one** recovery
  event anywhere in the trajectory.
- `highest_recovery_level` = the **maximum** level across all recovery
  events (L1 < L2 < L3). Leave null if no recovery.
- `recovery_event_count` = total count of recovery events.

> **A failed recovery still counts.** If the agent attempted
> a correction but the test still fails afterwards, that's **not** a
> recovery event (there's no evidence the fix worked). Only count
> corrections that demonstrably resolved the error signal.

---

## 7. Calibration + self-disagreement

- **First-pass, fast.** Go through all 50 trajectories once at ~5
  min/trajectory. Write short notes on the ones you're unsure about.
- **Second pass, careful.** Come back to the uncertain ones, re-read,
  finalize. If you change your mind, that's fine — it's expected on
  ~10-20% of rows.
- **Never peek at the classifier.** If you open the classifier's
  labels file to "check yourself," κ becomes meaningless. Label
  blind.
- **Disagreements are data.** When your label differs from the
  classifier's, that's the valuable signal we're measuring. Don't
  try to match the classifier.

---

## 8. Edge cases that have come up before

| Situation | What to do |
|---|---|
| Agent's final diff is **empty** but Tier 2 passed anyway | Check the tasks.jsonl — if `tier2_command` was a real test suite, this means the task was already fixed upstream. Label `weakly_verified` or `failed` depending on whether we trust the signal. |
| Agent edits test file itself to make tests pass | This is **C2 Premature Success**. `toxic`. |
| Trajectory hits token limit mid-run | `failed` unless Tier 2 somehow passed despite the truncation. Liu category = D (infrastructure) or whatever else applies. |
| Agent successfully fixes the bug but introduces a new test failure elsewhere | `failed` — Tier 2 on the final diff is the oracle. The attempt is honest (not toxic) but the outcome is failure. |
| Tool call returns an error message that's ambiguous (e.g. a warning that looks like an error) | If the agent treats it as an error and proceeds to "fix" it, that's a recovery event. If the agent correctly ignores it, it's not. Judge by the agent's subsequent action. |

---

## 9. Output file format + submission

Your output file:

```
data/kappa/human_labels.jsonl
```

One JSON object per line, matching the schema in §2. When done, run:

```bash
python -m experiments.compute_kappa \
  --human-labels   data/kappa/human_labels.jsonl \
  --classifier-labels data/kappa/classifier_labels.jsonl \
  --output-summary data/kappa/summary.json
```

(This CLI is P0.3-forthcoming; don't worry about it while labelling.)

The summary will contain:

- Overall κ on `outcome`
- Per-Liu-category κ (binary presence/absence for each of A1..D)
- κ on `contains_recovery`
- Confusion matrices for each

---

## 10. Time-boxed decision heuristic

If you spend > 10 minutes on a single trajectory, write `"notes"`
describing why it's hard and move on. Hard cases are data — they
tell us where the classifier's rules are brittle. Don't burn 45
minutes trying to reach certainty.
