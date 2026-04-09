# Pare — The Coding Agent That Never Breaks Your Repo

## Project Plan & Technical Specification

**Author:** Chenz
**Date:** April 2026
**Status:** Phase 1 complete, Phase 2 in progress
**Language:** Python 3.12+
**License:** MIT

---

## 1. What is Pare?

Pare is an **open-source, headless-first coding agent** that runs anywhere — terminal, CI pipeline, or SWE-bench harness — with any OpenAI-compatible LLM.

### 1.1 One-line pitch

> **The coding agent that never breaks your repo.**

### 1.2 Dual audience, dual narrative

**For individual developers** (emotional selling point — drives open-source adoption):
- Safe rollback: let the agent edit freely, roll back instantly if it fails. Your repo is never in danger.
- Headless CI: run in GitHub Actions, cron jobs, pre-commit hooks — where IDE agents can't go.

**For researchers & cost-sensitive teams** (rational selling point — drives enterprise/research adoption):
- Token efficiency: significantly fewer tokens than SWE-agent on equivalent tasks.
- Trajectory generation: produce high-quality SFT training data at lower cost per trajectory.

### 1.3 Three measurable claims

| # | Claim | How to prove | Audience |
|---|---|---|---|
| 1 | **Safe rollback** — failed tasks leave the repo clean | Git checkpoint demo video | Individual devs |
| 2 | **Token efficiency** — significantly fewer tokens than SWE-agent at equal success rate | 50-task benchmark, public scripts + logs + failure samples (3 runs, median, full JSONL) | Researchers |
| 3 | **Long-task memory** — 30+ min sessions without accuracy decay | Long-session benchmark, mid/late success rate curves | Both |

These claims drive architecture and roadmap decisions. "Significantly fewer" intentionally avoids a hard percentage — the benchmark report gives exact numbers with confidence intervals. Hardcoding "40-60%" before having data is a credibility risk.

### 1.3 Why not use X?

| Tool | Limitation Pare solves |
|---|---|
| **SWE-agent** | Blind ReAct burns tokens exploring. Bash-only tools. litellm hangs. Context truncation loses findings. |
| **Aider** | Single-turn edit, no tool loop. Great repo-map idea but no iterative agent. |
| **OpenHands** | Heavy microservice stack. Over-engineered for a lean framework. |
| **Claude Code** | Proprietary, TypeScript-only, Anthropic-locked. Great architecture ideas worth clean-room reimplementation. |
| **VS Code Agents** | IDE-locked, closed-source, cloud-only. Can't run in CI or headless. |

### 1.4 Design principles

1. **Headless-first.** The primary mode is `pare run "fix the bug" --headless`. Interactive CLI is a wrapper, not the core.
2. **Provider-agnostic.** Any OpenAI-compatible API works. Switch models with `--model`. Target: cheap models (MiniMax M2.7, DeepSeek) performing close to expensive ones.
3. **Token-miserly.** Every design choice minimizes token waste: zero-LLM orient, per-step budgets, heuristic compression (no LLM calls for compaction).
4. **Safe by default.** Git checkpoint before every mutation. Atomic rollback. User's branch untouched until finalize.
5. **Measurable.** Token counts in every result. Built-in SWE-bench harness. Public benchmarks.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Entry Points                                            │
│  CLI (interactive) · Headless (batch/CI) · SWE-bench     │
├──────────────────────────────────────────────────────────┤
│  Agent Orchestrator — Hybrid Loop                        │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────────┐     │
│  │ Orient  │─▶│  Plan   │─▶│ Execute (bounded     │     │
│  │ 0 LLM   │  │ 1 LLM   │  │ ReAct + checkpoint)  │     │
│  └─────────┘  └─────────┘  └─────────┬────────────┘     │
│                     ▲                 │                   │
│                     └── replan ◄──────┘                   │
├──────────────────────────────────────────────────────────┤
│  Tool Registry                                           │
│  bash · file_read · file_edit · file_create · search     │
│  Permission gates: auto / confirm_once / always_confirm  │
├──────────────────────────────────────────────────────────┤
│  Context Manager                                         │
│  Memory Index (always loaded, ~500 tokens)               │
│  Session History (JSONL, searchable, never bulk-loaded)  │
│  3-Stage Heuristic Compression                           │
├──────────────────────────────────────────────────────────┤
│  LLM Adapter — Provider-Agnostic                         │
│  OpenAI-compatible (MiniMax, DeepSeek, GPT, Claude*)     │
│  chat() + chat_stream() + count_tokens()                 │
│  Self-managed retry · token tracking                     │
├──────────────────────────────────────────────────────────┤
│  Git Checkpoint                                          │
│  Working branch isolation · commit-per-step              │
│  Atomic rollback · squash-merge finalize                 │
└──────────────────────────────────────────────────────────┘

* Claude via Anthropic adapter (native SDK)
```

### 2.1 Key decisions

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.12+ | LLM SDK ecosystem, async native |
| LLM abstraction | Thin per-provider adapters | litellm unreliable; each adapter ~150 lines, fully understood |
| Agent loop | Hybrid: Orient → Plan → Execute | Orient=zero-cost context; Plan=global direction; Execute=tactical flexibility |
| Memory | 2-layer (Index + History) | Index always in context; History searchable but not loaded. TopicStore cut — no auto-population mechanism, dead complexity |
| Compression | 3-stage heuristic (no LLM calls) | LLM-based summarization burns tokens, contradicts "token-miserly" principle |
| Safety | Git checkpoint | Atomic rollback, diff visibility, branch isolation — all free from git |
| Entry point | Headless-first | CI/CD and SWE-bench are primary; interactive CLI is a thin wrapper |

---

## 3. Module Specifications

### 3.1 LLM Adapter (`pare/llm/`)

Provider-agnostic interface. The rest of the framework never imports `openai` or `anthropic` directly.

```python
class LLMAdapter(ABC):
    async def chat(messages, tools?, temperature?, max_tokens?) -> LLMResponse
    async def chat_stream(messages, tools?, ...) -> AsyncIterator[StreamChunk]
    def count_tokens(messages) -> int

@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCallRequest]
    stop_reason: StopReason
    usage: TokenUsage  # input/output/cache tokens

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
```

**Provider factory:**
- `"openai"` → OpenAIAdapter (also covers MiniMax, DeepSeek, any OpenAI-compatible API via `base_url`)
- `"anthropic"` → AnthropicAdapter (native SDK, prompt cache support)

**Implemented quirk handling:**
- MiniMax M2.7: `<think>` tag stripping (regex in adapter + state machine in streaming)
- Temperature clamping to (0.0, 1.0] for providers that reject 0.0
- Tool call argument JSON parsing (OpenAI returns strings, Anthropic returns dicts)

**Files:** `base.py`, `openai_adapter.py`, `anthropic_adapter.py` (P1), `retry.py`, `token_counter.py`

### 3.2 Tool System (`pare/tools/`)

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict          # JSON Schema
    mutation_type: MutationType   # READ | WRITE | EXECUTE
    permission_level: PermissionLevel  # AUTO | CONFIRM_ONCE | ALWAYS_CONFIRM

    async def execute(params, context) -> ToolResult
```

**MVP tools (all implemented):**

| Tool | Mutation | Permission | Notes |
|---|---|---|---|
| `bash` | EXECUTE | ALWAYS_CONFIRM | Timeout 30s, output truncated to 200 lines |
| `file_read` | READ | AUTO | Line ranges, encoding detection, max 100KB |
| `file_edit` | WRITE | CONFIRM_ONCE | str_replace with whitespace-insensitive fallback |
| `file_create` | WRITE | CONFIRM_ONCE | Fails if file exists, auto-creates parents |
| `search` | READ | AUTO | ripgrep wrapper, max 50 results |

**Whitespace-insensitive fallback** (unique to Pare): When exact `old_str` match fails, normalizes whitespace and retries. Open-source models frequently hallucinate minor whitespace diffs — this recovers ~15% of otherwise-failed edits.

**`recall_history` tool** (P2): Lets the agent grep its own session history via `SessionHistory.search()`. Zero cost unless the agent actively calls it. Solves the "heuristic compression might drop critical info" problem — if the agent needs something that was compressed away, it can search for it. Unlike TopicStore (which required agent to proactively write), this is passive retrieval — simple and robust.

**CI permission model** (P2): Headless mode needs bash to work without interactive confirmation. Solution: tiered exact-match allow-list (not prefix matching — prefix matching can be bypassed).

| Mode | Flag | Allowed commands | Use case |
|---|---|---|---|
| Interactive | (default) | All, with ALWAYS_CONFIRM on bash | Daily dev use |
| CI | `--ci` | Read-only: `ls`, `cat`, `grep`, `find`, `git diff/status/log/show`, `python -m pytest`, `npm test` | Safe CI pipelines |
| Bench | `--ci --bench` | CI list + `pip install -e .`, `pip install -r requirements.txt`, `npm install`, `cargo build` | SWE-bench evaluation |

**Always denied** (regardless of mode): `git push`, `git reset --hard`, `rm -rf`, `curl`/`wget` to external URLs, `sudo`, any command with `|` piping to `rm`/`dd`/`mkfs`.

Custom allow-list via `.pare/ci-policy.toml` (P3). Rationale: without this, "headless-first" is a lie — bash blocks on every call in CI. The `--bench` tier exists because SWE-bench tasks require `pip install -e .` to run tests.

**Files:** `base.py`, `bash.py`, `file_read.py`, `file_edit.py`, `search.py`, `recall_history.py` (P2)

### 3.3 Context Manager (`pare/context/`)

**2-layer memory (simplified from original 3-layer design):**

| Layer | What | Size | Lifecycle |
|---|---|---|---|
| Memory Index | Structured summary: repo structure, key signatures, current task, plan status | ~500-1000 tokens | Always in every LLM call |
| Session History | Raw JSONL of all messages/tool calls | Unbounded on disk | Searchable via keyword, never bulk-loaded |

TopicStore (original Layer 2) was cut — it had no automatic population mechanism and added complexity without value.

**3-stage heuristic compression (no LLM calls):**

```
Stage 1: Trim old tool results
  Keep last 5 in full, replace older with 1-line summaries
  Cost: 0 tokens, instant

Stage 2: Truncate verbose output
  Cap each remaining result at 50 lines
  Cost: 0 tokens, instant

Stage 3: Drop oldest messages (last resort)
  Drop oldest messages until under threshold
  Always preserve: system prompt + memory index + last 4 messages
  Cost: 0 tokens, instant
```

Original stages 3-4 (LLM-based extract/summarize) were cut. They consumed tokens for compression — directly contradicting the "token-miserly" principle. The heuristic stages handle 90%+ of real cases.

**Files:** `manager.py`, `memory.py`, `compactor.py`, `history.py`

### 3.4 Agent Orchestrator (`pare/agent/`)

The brain. Two execution modes:

**Flat mode** (`use_planning=False`): Pure bounded ReAct. For simple tasks.

**Hybrid mode** (`use_planning=True`): Orient → Plan → Execute.

#### Phase 1: Orient (0 LLM calls)

4 concurrent async scans:
1. Directory tree (depth-limited, ignore noise dirs)
2. Code signatures (regex: `def`, `class`, `function`, `func`)
3. Key file heads (README, pyproject.toml, etc. — first 50 lines)
4. Git state (branch, recent commits, uncommitted changes)

Output: `RepoContext` → injected into Memory Index.

This is **the core of the token savings story**. SWE-agent spends 10-20 LLM turns just understanding the repo. Pare gets that context for free.

#### Phase 2: Plan (1 LLM call)

Input: system prompt + Memory Index + task
Output: structured JSON plan (`PlanStep[]`)

```python
@dataclass
class PlanStep:
    step_number: int
    goal: str                 # Coarse-grained ("modify the auth module")
    target_files: list[str]
    expected_tools: list[str]
    budget: int = 15          # Max iterations for this step
    success_criteria: str
```

Fallback: if LLM returns invalid JSON, create a single-step plan with budget=30.

#### Phase 3: Execute (bounded ReAct per step)

For each step:
1. Git checkpoint
2. Bounded ReAct loop (max `step.budget` iterations)
3. **Hard verification (Tier 1, always on):**
   - Syntax check on edited .py files (`compile()`)
   - `git diff` non-empty check (agent claims edit but nothing changed → fail)
4. **Hard verification (Tier 2, opt-in):**
   - If `test_command` declared in config → run it; failure = step failure
5. On verified success → checkpoint, advance to next step
6. On failure → trigger replan (max 3 replans)

See §5.6 for rationale on why auto-detection of test commands is intentionally not done.

#### Guardrails

| Guard | Default | Action |
|---|---|---|
| Step budget | 15 iterations | Stop step, trigger replan |
| Total tool calls | 100 | Stop task |
| Consecutive errors | 3 | Inject warning |
| Repeated action (hash) | 2 | Inject warning, then block |
| Read-before-write | Always | Block edit without prior read |

**Files:** `orchestrator.py`, `orient.py`, `planner.py`, `executor.py`, `guardrails.py`

### 3.5 Git Checkpoint (`pare/sandbox/`)

```python
class GitCheckpoint:
    async def setup() -> None       # Create pare/working-{sha} branch
    async def checkpoint(msg) -> str  # Stage + commit, return SHA
    async def rollback(sha?) -> None  # Hard reset to checkpoint
    async def get_diff_since(sha) -> str
    async def get_full_diff() -> str
    async def finalize() -> None    # Squash-merge to original branch
    async def abort() -> None       # Discard working branch
```

Auto-commits dirty tree before setup. Finalize squash-merges back to original branch — clean single commit.

**Files:** `git_checkpoint.py`

### 3.6 CLI (`pare/cli/`)

Interactive wrapper around the headless core. Streaming display via Rich, input via prompt_toolkit.

**Files:** `app.py` (main loop), `renderer.py` (P1, stream display)

---

## 4. Implementation Status

### Done (Phase 1)

- [x] LLM base types + OpenAI adapter (MiniMax M2.7 validated)
- [x] `<think>` tag filtering (streaming + non-streaming)
- [x] Tool system: base + bash + file_read + file_edit + file_create + search
- [x] Whitespace-insensitive edit fallback
- [x] ReAct executor with guardrails
- [x] Git checkpoint (setup/commit/rollback/finalize/abort)
- [x] Basic CLI with streaming
- [x] Memory Index + Session History
- [x] Context Manager with heuristic compression
- [x] Orient phase (4 concurrent scans)
- [x] Planner (LLM plan generation + replan + JSON fallback)
- [x] Hybrid loop integration (flat + hybrid modes in orchestrator)
- [x] 318 tests passing

### Next (Phase 2 — Headless + Verification + Benchmark)

**Week 1: Foundation**
- [x] **Rename codebase** — rename `forge/` package dir to `pare/`, update all imports
- [ ] **Token tracking in ExecutionResult** — accumulate `TokenUsage` across all LLM calls per run
- [ ] **`--headless` batch mode** — `pare run "task" --headless --output result.json`

**Week 2: Verification + Recall**
- [ ] **Hard verification Tier 1** — syntax check (compile()) + git diff non-empty check, always on
- [ ] **Hard verification Tier 2** — opt-in `test_command` from `.pare/config.toml`
- [ ] **`recall_history` tool** — agent greps session JSONL on demand
- [ ] **Compaction-triggered system note** — inject "your context was compacted, use recall_history" after compression

**Week 3: CI + Harness**
- [ ] **CI permission model** — `--ci` (read-only whitelist) + `--ci --bench` (adds install commands)
- [ ] **SWE-bench harness** — load instance → apply repo → run pare headless → collect patch
- [ ] **Smoke test: 5 tasks** — validate full pipeline end-to-end before scaling

**Week 4: Benchmark**
- [ ] **50-task benchmark** — 3 runs × 50 tasks, median, full JSONL public
- [ ] **Baseline comparison** — same tasks on SWE-agent (pinned SHA), same model/hyperparams
- [ ] **Benchmark report** — exact numbers with confidence intervals, not marketing claims

### Phase 3 — Polish & Release

- [ ] Anthropic adapter (prompt cache support)
- [ ] `pare.toml` / `.pare/ci-policy.toml` configuration
- [ ] README: dual narrative (devs: safe rollback; researchers: token efficiency + trajectory gen)
- [ ] README section: "For Researchers: Trajectory Generation for SFT"
- [ ] Demo video: safe rollback comparison
- [ ] Long-session benchmark (30 min, success rate curve)
- [ ] SWE-bench Verified run (500 instances)
- [ ] GitHub release + PyPI publish

### Explicitly NOT in scope (cut)

| Feature | Why cut |
|---|---|
| TopicStore (Layer 2 memory) | No auto-population mechanism. Dead complexity. |
| LLM-based compression (stages 3-4) | Burns tokens to save tokens. Contradicts core selling point. |
| Hook/lifecycle system | Over-engineered for a new framework with zero users. Can add later if needed. |
| MCP adapter | Ecosystem feature, not core differentiator. P3+ at best. |
| Docker sandbox | Git checkpoint is sufficient safety for MVP. |
| Go rewrite | Premature optimization. Python is fast enough for I/O-bound agent work. |
| Pydantic dependency | Dataclasses + manual validation are lighter. |

---

## 5. Design Decisions & Rationale

### 5.1 Why Hybrid Loop over Pure ReAct?

Pure ReAct (SWE-agent) has no global plan — the agent explores blindly, often spending 10+ tool calls in the wrong file. The Orient phase gives free context; the Plan phase sets direction; Execute stays tactical. **This is the primary mechanism for token savings** — validated by benchmark, not by assertion.

Tradeoff: Plan adds 1 LLM call overhead. For trivial tasks, use flat mode (`use_planning=False`).

### 5.2 Why Custom Adapters over litellm?

litellm caused connection hangs, retried non-retryable errors, and had tool schema edge cases. Each of our adapters is ~150 lines, fully understood. Adding a new provider is 2 hours of implementing 3 methods.

### 5.3 Why Git Checkpoints?

Agent edits are multi-file and multi-step. An undo stack can't atomically rollback "everything from step 3." Git gives us atomic rollback, diff visibility, and branch isolation for free. **This is the "safe rollback" selling point.**

### 5.4 Why Heuristic Compression over LLM Summarization?

LLM-based summarization (stages 3-4 in the original design) requires 1-2 extra LLM calls per compaction. In a long session with multiple compactions, this adds up. Heuristic stages (trim old results → truncate verbose → drop oldest) handle 90%+ of real cases with zero token cost. **Token spent on compression is token wasted.**

### 5.5 Why Headless-First?

VS Code Agents, Cursor, Windsurf own the IDE space. Claude Code owns the interactive CLI space. Pare's moat is running **where they can't**: CI pipelines, GitHub Actions, SWE-bench harnesses, cron jobs. The interactive CLI is a thin convenience wrapper, not the product.

### 5.6 Why Hard Verification over LLM Self-Report?

Open-source models hallucinate success at alarming rates. The agent says "I fixed the bug" but the file has a syntax error, or the test still fails, or the git diff is empty. **Hard verification is not a replacement for user testing — it catches a specific class of LLM self-report lies.**

**Tier 1 (Phase 2, zero-config, always on):**
- Syntax check: `python -c "compile(...)"` for .py files. Other languages: call the corresponding syntax checker if available, skip if not. Zero false positives.
- Git diff non-empty check: agent claims to have edited a file but `git diff` is empty → step marked failed.
- Cost: ~0. Two subprocess calls. 100% reliable on what they cover.

**Tier 2 (Phase 2, opt-in via config):**
- User declares `test_command` in `.pare/config.toml` (e.g., `test_command = "pytest tests/"` or `verify_command = "make check"`)
- If declared → run after each step; failure = step failure regardless of LLM's claim
- If not declared → skip. **No auto-detection.** Auto-detecting test frameworks fails ~30% of the time, and each failure burns user trust.

Rationale for not auto-detecting: "which test runner?" × "which tests are relevant?" × "how long does it take?" × "does it have side effects?" — four open questions that can't be answered reliably without user input. Tier 1 gives us 80% of the value with 0% configuration.

### 5.7 Why Lazy Recall — and Why It's Not Enough Alone

Heuristic compression (stages 1-3) is lossy by design — it trades information for context space. The risk: the agent compressed away a critical finding from turn 5 and can't recover it. Solution: `recall_history` tool that greps the JSONL on disk. Zero cost unless the agent calls it, unlimited depth when it does.

**Known limitation: small models don't know what they don't know.** MiniMax M2.7, DeepSeek, etc. lack metacognition — after compression, they won't think "I should check my history", they'll just hallucinate. Lazy recall alone relies on the model having this awareness.

**Mitigation: compaction-triggered system note (deterministic, not model-dependent).**

When the compactor runs, it injects a system note into the next LLM call:

```
[CONTEXT COMPACTED] Older conversation details have been moved to session history.
If you need to reference any variable, function, file, or decision from earlier turns
that is not in your immediate visible context, you MUST call recall_history(keyword)
to retrieve it. Do NOT guess or assume — search first.
```

This injection is deterministic — it fires every time compression happens, regardless of model capability. It converts the metacognition problem ("does the model know it forgot?") into a prompt-following problem ("does the model follow an instruction?"), which even small models handle better.

The two mechanisms work together:
- `recall_history` tool: gives the capability
- Compaction-triggered note: gives the trigger
- Neither alone is sufficient; together they cover both capable and weak models

---

## 6. Benchmark Plan

### 6.1 Token Efficiency Benchmark

**Setup:**
- 50 tasks from SWE-bench Lite (curated: mix of easy/medium/hard)
- Same tasks run on: Pare vs SWE-agent
- **Same model** (exact model ID + version documented)
- **Same hyperparameters** (temperature, max_tokens, timeout — all documented)
- **3 runs per task, report median** (not cherry-picked best)
- Metrics: success rate, total tokens, tokens per resolved instance

**Deliverables:**
- One-click reproduction script (ideally Docker image)
- Complete trajectory JSONL for every run (not just successes)
- Failure case analysis with root cause categories
- Summary table with confidence intervals, not point estimates
- All hyperparameters, model versions, and environment details in a single `benchmark_config.json`

**Target:** Match SWE-agent success rate with significantly fewer tokens. Report exact numbers — don't claim a percentage before having data.

**Defensive measures** (HN/Reddit will ask):
- "Which SWE-agent version?" → pin git SHA
- "What temperature?" → documented in config
- "Did you cherry-pick?" → 3 runs, median, all trajectories public
- "Can I reproduce?" → yes, here's the Docker image

### 6.2 Safe Rollback Demo

3-minute video:
1. SWE-agent fails on a task → repo left with partial changes
2. Pare fails on same task → `pare abort` → repo clean
3. Pare succeeds → `pare finalize` → single clean commit

### 6.3 Long Session Benchmark

**Setup:**
- 10 multi-step tasks requiring 20+ tool calls each
- Measure success rate in first half vs second half of each session
- Compare: Pare (with compression) vs baseline (no compression, same token budget)

**Target:** <10% success rate degradation in second half.

---

## 7. Dependencies

```toml
[project]
name = "pare"
requires-python = ">=3.12"

dependencies = [
    "openai>=1.50.0",         # LLM adapter (OpenAI-compatible APIs)
    "rich>=13.0",              # CLI output formatting
    "prompt-toolkit>=3.0",     # CLI input (history, async)
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "pytest-mock>=3.14", "ruff>=0.8"]

[project.scripts]
pare = "pare.main:main"
```

3 runtime dependencies. That's it. Anthropic SDK is optional.
