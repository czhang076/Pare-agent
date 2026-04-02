# Forge — A Lightweight, Extensible Coding Agent Framework

## Project Plan & Technical Specification

**Author:** Chenz
**Date:** April 2026
**Status:** Pre-development — Architecture Finalized
**Language:** Python 3.12+
**License:** MIT

---

## 1. Vision & Motivation

### 1.1 What is Forge?

Forge is a **provider-agnostic, extensible coding agent framework** built in Python. It provides a core agent runtime that can power multiple applications: an interactive CLI coding assistant (similar to Claude Code), a SWE-bench solver, or any custom coding automation pipeline.

### 1.2 Why build this?

Existing tools each have critical limitations that Forge addresses:

- **SWE-agent (Princeton):** Bash-only tool interface, brutal context truncation, litellm dependency instability (connection errors, hangs mid-instance). No planning capability — pure ReAct loop walks blindly.
- **Aider:** Not an agent — single-turn edit mode with no tool-use loop. Excellent repo-map idea (tree-sitter signatures) but no iterative reasoning.
- **OpenHands (OpenDevin):** Heavy microservice architecture (EventStream bus, Docker runtime). Good design but over-engineered for a lightweight framework.
- **Claude Code (Anthropic):** Production-grade but proprietary, TypeScript/Bun-only, tightly coupled to Anthropic's API. Key architectural ideas (3-layer memory, gradient compression, read/write tool separation) are worth learning from but need clean-room reimplementation.

### 1.3 Design Principles

1. **Provider-agnostic from Day 1.** Anthropic, OpenAI, DeepSeek, Qwen — switch models with a config flag, not a code change.
2. **Tool system is pluggable.** Native high-performance tools for core operations + MCP protocol adapter for third-party extensions. LLM sees a unified tool schema regardless of backend.
3. **Hybrid agent loop.** Orient → Plan → Execute with bounded ReAct, git checkpoints, and automatic replanning. Not a blind loop; not a rigid pipeline.
4. **Context is a first-class engineering problem.** 3-layer memory architecture + 5-stage gradient compression. The framework should handle 30+ minute coding sessions without context degradation.
5. **Safe by default.** Git checkpoints before every mutation. Permission gates on destructive tools. Budget limits to prevent runaway loops. Graceful degradation when the agent can't solve the problem.
6. **Extensible via hooks.** Pre/post lifecycle events at every critical point. Third parties (or your own future self) can inject logic without modifying core code.

### 1.4 Target Applications

| Application | Priority | Description |
|---|---|---|
| Interactive CLI Agent | P0 (MVP) | Terminal-based coding assistant. User gives a task in natural language, agent reads/edits/tests code. |
| SWE-bench Solver | P1 | Headless mode: takes a GitHub issue + repo, produces a patch. Evaluated on SWE-bench Lite/Verified. |
| CI/CD Integration | P2 | Run as a GitHub Action or pre-commit hook for automated code review/fix. |

---

## 2. Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│  Applications                                                 │
│  CLI Agent · SWE-bench Solver · future: Web UI / CI runner    │
├───────────────────────────────────────────────────────────────┤
│  Agent Orchestrator — Hybrid Loop                             │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────────┐    │
│  │  Orient   │──▶│  Plan    │──▶│  Execute (bounded      │    │
│  │ repo-map  │   │ LLM→JSON │   │  ReAct + checkpoints)  │    │
│  └──────────┘   └──────────┘   └────────┬───────────────┘    │
│                      ▲                   │                    │
│                      └───── replan ◄─────┘                    │
│                                                               │
│  Hooks: pre_step · post_step · pre_tool · post_tool           │
├───────────────────────────────────────────────────────────────┤
│  Tool Registry & Executor                                     │
│  Native Tools (bash, file_read, file_edit, search)            │
│  + MCP Adapter (external tool plugins)                        │
│  + Permission Gates (read=auto, write=confirm, bash=confirm)  │
├───────────────────────────────────────────────────────────────┤
│  Context Manager                                              │
│  3-Layer Memory: Index (always loaded)                        │
│                  Topic Files (on-demand)                       │
│                  Session History (grep-searchable)             │
│  5-Stage Gradient Compression Pipeline                        │
├───────────────────────────────────────────────────────────────┤
│  LLM Adapter — Provider-Agnostic                              │
│  Anthropic · OpenAI · OpenRouter (DeepSeek/Qwen)              │
│  Unified: chat() + chat_stream() + count_tokens()             │
│  Self-managed retry · prompt cache optimization               │
├───────────────────────────────────────────────────────────────┤
│  Sandbox / Runtime                                            │
│  Docker isolation (optional) · File system mount              │
│  Git checkpoint/rollback · Process management                 │
└───────────────────────────────────────────────────────────────┘
```

### 2.1 Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | Fastest LLM SDK ecosystem, async/await native, team familiarity. Go rewrite of perf-critical modules planned for Phase 3. |
| LLM Abstraction | Custom thin adapters per provider | litellm proved unreliable (connection errors, tool-use schema leaks). Each provider gets a dedicated wrapper; retry/timeout controlled by us. |
| Tool Protocol | Native interface + MCP adapter | Core tools (bash, file ops) are native Python for performance. External tools use MCP for ecosystem compatibility. LLM sees a unified ToolSchema. |
| Agent Loop | Hybrid: Orient → Plan → Execute | Pure ReAct walks blind; pure Plan-then-Execute makes stale plans. Hybrid gives global vision + tactical flexibility + replan on failure. |
| Context Management | 3-layer memory + 5-stage compaction | Inspired by Claude Code's architecture. Memory index always in context (~500 tokens), topic files fetched on-demand, history searchable but not loaded. Compression is gradual, not cliff-edge. |
| Safety | Git checkpoints + budget + permission gates | Every write-step is preceded by a git commit. Budget caps prevent infinite loops. Destructive operations require user confirmation. |
| Extensibility | Hook lifecycle system | 25+ events (MVP: 4 key hooks). No core code modification needed to add pre/post logic. |

---

## 3. Module Specifications

### 3.1 LLM Adapter (`forge/llm/`)

#### 3.1.1 Purpose
Provide a provider-agnostic interface for all LLM interactions. The rest of the framework never imports `anthropic` or `openai` directly — only `forge.llm`.

#### 3.1.2 Public Interface

```python
class LLMAdapter(ABC):
    async def chat(messages, tools?, temperature?, max_tokens?) -> LLMResponse
    async def chat_stream(messages, tools?, ...) -> AsyncIterator[StreamChunk]
    def count_tokens(messages) -> int

@dataclass
class LLMResponse:
    content: str                        # Text output
    tool_calls: list[ToolCallRequest]   # Tool invocations
    stop_reason: StopReason             # end_turn | tool_use | max_tokens
    usage: TokenUsage                   # input/output/cache tokens

@dataclass
class ToolCallRequest:
    id: str          # For result correlation
    name: str        # Tool name
    arguments: dict  # Parsed JSON params

@dataclass
class StreamChunk:
    type: str        # text | tool_call_start | tool_call_delta | tool_call_end | usage
    content: str
    tool_call: ToolCallRequest | None

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0    # Anthropic prompt cache hits
    cache_create_tokens: int = 0  # Anthropic prompt cache writes
```

#### 3.1.3 Message Format (Provider-Agnostic)

```python
@dataclass
class Message:
    role: str  # system | user | assistant | tool_result
    content: str | list[ContentBlock]

@dataclass
class ContentBlock:
    type: str  # text | tool_use | tool_result
    text: str = ""
    tool_call: ToolCallRequest | None = None
    tool_call_id: str = ""
```

Each adapter translates this unified format to provider-specific API format:

| Aspect | Anthropic | OpenAI |
|---|---|---|
| System message | Top-level `system` param | In `messages` array |
| Tool result role | `user` with `tool_result` block | `tool` role |
| Tool schema | `input_schema` key | `function.parameters` wrapper |
| Stop reason | `end_turn` / `tool_use` | `stop` / `tool_calls` |
| Tool call location | In `content` blocks | In `message.tool_calls` |
| Tool call arguments | Parsed dict | JSON string (needs parsing) |

#### 3.1.4 Retry Policy

Self-managed retry replaces SDK-level retry. Only retryable errors are retried:

- **Retryable:** RateLimitError, InternalServerError, APIConnectionError, ConnectTimeout, ReadTimeout
- **Not retryable:** BadRequestError (400), AuthenticationError (401), PermissionError (403), NotFoundError (404)

Exponential backoff with `retry-after` header respect. Default: 3 retries, 1s base backoff.

#### 3.1.5 Prompt Cache Optimization (Anthropic-specific)

Messages are assembled with cache hints: system prompt and early conversation history are marked with `cache_control: {"type": "ephemeral"}`. The context manager ensures stable message prefixes — changing content is appended at the end to maximize cache hit rate.

#### 3.1.6 Provider Factory

```python
def create_llm(provider: str, **kwargs) -> LLMAdapter
```

- `"anthropic"` → AnthropicAdapter (direct SDK)
- `"openai"` → OpenAIAdapter (direct SDK)
- `"openrouter"` → OpenAIAdapter with `base_url="https://openrouter.ai/api/v1"` (DeepSeek, Qwen, etc.)

#### 3.1.7 Files

```
forge/llm/
├── __init__.py          # create_llm() factory
├── base.py              # LLMAdapter ABC, LLMResponse, Message, etc.
├── anthropic_adapter.py # Anthropic implementation
├── openai_adapter.py    # OpenAI/OpenRouter implementation
├── retry.py             # RetryPolicy with exponential backoff
└── token_counter.py     # Estimation utils (tiktoken for OpenAI, API for Anthropic)
```

---

### 3.2 Tool Registry & Executor (`forge/tools/`)

#### 3.2.1 Purpose
Define, register, discover, and execute tools. Provide a unified interface that hides whether a tool is native Python or an external MCP server.

#### 3.2.2 Tool Base Class

```python
class MutationType(Enum):
    READ = "read"       # File reads, searches — safe to run concurrently
    WRITE = "write"     # File edits, git operations — must run serially
    EXECUTE = "execute" # Bash commands — serial, may need user confirmation

class PermissionLevel(Enum):
    AUTO = "auto"           # No confirmation needed (reads)
    CONFIRM_ONCE = "once"   # Confirm first time, then auto (file writes)
    ALWAYS_CONFIRM = "always"  # Always ask (bash, destructive ops)

class Tool(ABC):
    name: str
    description: str                  # Shown to LLM
    parameters: dict                  # JSON Schema for input validation
    mutation_type: MutationType
    permission_level: PermissionLevel

    @abstractmethod
    async def execute(self, params: dict, context: ToolContext) -> ToolResult

    def to_schema(self) -> ToolSchema:
        """Convert to LLM-facing schema (unified format)"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
```

#### 3.2.3 Tool Registry

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None
    def get(self, name: str) -> Tool
    def get_all_schemas(self) -> list[ToolSchema]
    def get_schemas_by_mutation(self, *types: MutationType) -> list[ToolSchema]

    async def execute(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute with read/write separation:
        - READ tools: asyncio.gather (concurrent)
        - WRITE/EXECUTE tools: sequential
        Maintains original call order in results.
        """
```

#### 3.2.4 P0 Native Tools (MVP)

**BashTool**
- Runs shell commands in the sandbox
- Captures stdout, stderr, return code
- Timeout: configurable, default 30s
- Truncates output to max_output_lines (default 200)
- Permission: ALWAYS_CONFIRM

**FileReadTool**
- Reads file content with optional line range `[start, end]`
- Returns content with line numbers prepended
- Handles encoding detection (UTF-8, Latin-1 fallback)
- Max file size: 100KB (returns head + tail for larger files)
- Permission: AUTO
- Mutation: READ

**FileEditTool (str_replace style)**
- Takes: file_path, old_str, new_str
- old_str must match exactly once in the file (prevents ambiguous edits)
- Returns: diff preview of the change
- Validates: file exists, old_str is unique, new_str is different
- Permission: CONFIRM_ONCE
- Mutation: WRITE

**FileCreateTool**
- Takes: file_path, content
- Creates new file (fails if file already exists — use FileEditTool for modifications)
- Creates parent directories automatically
- Permission: CONFIRM_ONCE
- Mutation: WRITE

**SearchTool (ripgrep wrapper)**
- Takes: pattern (regex), path (optional, defaults to repo root), file_glob (optional)
- Returns: matching lines with file paths and line numbers
- Max results: 50 (configurable)
- Uses `rg` subprocess — requires ripgrep installed
- Permission: AUTO
- Mutation: READ

#### 3.2.5 P1 Tools (Post-MVP)

**GitTool**
- Subcommands: diff, status, log, checkout, commit
- Used by agent for understanding changes, not just by checkpoint system
- Permission: CONFIRM_ONCE for mutating ops

**TestRunnerTool**
- Auto-detects test framework (pytest, unittest, jest, go test)
- Runs specific test files or full suite
- Parses output to extract pass/fail/error counts
- Timeout: 120s default
- Permission: AUTO (read-only operation from agent's perspective)

#### 3.2.6 MCP Adapter

```python
class MCPAdapter:
    """Connects to an MCP server and registers its tools as native Tool instances"""

    async def connect(self, server_url: str) -> list[Tool]
    async def execute_remote(self, tool_name: str, params: dict) -> ToolResult
```

MCP tools are wrapped in a `MCPToolProxy` class that implements the `Tool` interface. The LLM cannot distinguish between native and MCP tools — they share the same ToolSchema format.

#### 3.2.7 Files

```
forge/tools/
├── __init__.py
├── base.py              # Tool ABC, ToolRegistry, ToolResult, ToolContext
├── bash.py              # BashTool
├── file_read.py         # FileReadTool
├── file_edit.py         # FileEditTool + FileCreateTool
├── search.py            # SearchTool (ripgrep wrapper)
├── git.py               # GitTool (P1)
├── test_runner.py       # TestRunnerTool (P1)
└── mcp_adapter.py       # MCPAdapter + MCPToolProxy
```

---

### 3.3 Context Manager (`forge/context/`)

#### 3.3.1 Purpose
Manage what information the LLM sees at each turn. Solve the core challenge: codebases are huge, conversation histories grow fast, but the context window is finite.

#### 3.3.2 Three-Layer Memory Architecture

Inspired by Claude Code's leaked architecture, adapted for a lightweight framework:

**Layer 1: Memory Index (always in context)**
- A small structured summary (~500-1000 tokens) that is ALWAYS included in every LLM call
- Contains: repo structure overview, key file signatures, task description, current plan status
- Format: Markdown-like plaintext, stored in `MEMORY.md` in the working directory
- Updated via "strict write discipline" — only updated AFTER successful operations, never on speculation or failed attempts
- This ensures the LLM always has a "map" of where it is and what it's doing

```
# Project: my-api
## Structure: src/auth/ (3 files), src/api/ (5 files), tests/ (4 files)
## Key Signatures:
- src/auth/token.py: validate_token(token: str) -> bool, class TokenManager
- src/api/routes.py: login(request), refresh(request)
## Current Task: Add refresh token support to auth module
## Plan Status: Step 2/4 — Modifying validate_token
```

**Layer 2: Topic Files (on-demand)**
- Detailed knowledge about specific topics/modules, stored as separate files
- Retrieved by the orchestrator when a plan step targets a specific area
- Example: `topics/auth_module.md` contains detailed notes about the auth module's behavior
- Not loaded unless relevant to the current step
- Written by the agent during Orient phase or after deep exploration

**Layer 3: Session History (searchable, not loaded)**
- Raw conversation history (all messages, tool calls, results)
- Never fully loaded back into context
- Searchable via grep — agent can query "what did I try for the login function?" and get relevant snippets
- Stored as JSONL: one line per message/event

#### 3.3.3 Five-Stage Gradient Compression Pipeline

When `context.token_count()` exceeds the threshold (default: 70% of model's max context), compaction triggers. Each stage is tried in order; processing stops as soon as token count drops below threshold:

```
Stage 1: Trim old tool results
  └─ Keep only the last 5 tool call results in full
  └─ Replace older results with 1-line summaries
  └─ Cost: zero LLM calls, instant

Stage 2: Truncate verbose tool output
  └─ Cap each remaining tool result at 50 lines
  └─ Append "[truncated — N more lines]"
  └─ Cost: zero LLM calls, instant

Stage 3: Extract session memory
  └─ Send conversation so far to LLM: "Extract key findings and decisions"
  └─ Write output to a topic file
  └─ Replace detailed history with reference: "See topics/session_findings.md"
  └─ Cost: 1 LLM call

Stage 4: Full history summarization
  └─ Send entire conversation to LLM: "Summarize everything done so far"
  └─ Replace all history with the summary
  └─ Cost: 1 LLM call

Stage 5: Oldest message truncation (last resort)
  └─ Drop oldest messages one by one until under threshold
  └─ Always preserve: system prompt, memory index, current step context
  └─ Cost: zero LLM calls, instant, but lossy
```

#### 3.3.4 Context Assembly for LLM Calls

```python
class ContextManager:
    def assemble(self) -> list[Message]:
        """Build the message list for the next LLM call.
        Order matters for prompt cache optimization:
        1. System prompt (stable — cached)
        2. Memory index (mostly stable — cached)
        3. Relevant topic files (stable within a step — cached)
        4. Conversation history (grows — partially cached)
        5. Current step prompt (changes each step — not cached)
        """

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> None
    def push_step_prompt(self, step: PlanStep) -> None
    def needs_compaction(self) -> bool
    def compact(self) -> None  # Runs the 5-stage pipeline
    def update_memory_index(self, updates: dict) -> None
    def search_history(self, query: str) -> list[str]
```

#### 3.3.5 Files

```
forge/context/
├── __init__.py
├── manager.py           # ContextManager — message assembly + compaction trigger
├── memory.py            # MemoryIndex — MEMORY.md read/write + topic file management
├── compactor.py         # CompactionPipeline — 5-stage gradient compression
└── history.py           # SessionHistory — JSONL storage + grep search
```

---

### 3.4 Agent Orchestrator (`forge/agent/`)

#### 3.4.1 Purpose
Implement the hybrid Orient → Plan → Execute loop. This is the "brain" of the framework.

#### 3.4.2 Hybrid Loop Design

**Phase 1: Orient (no LLM, pure code)**

Automatically scans the repository to build initial context:
1. Directory tree (max depth 3, ignore .git, node_modules, __pycache__, etc.)
2. Code signatures (MVP: regex for `def`, `class`, `function`, `func`; P2: tree-sitter AST)
3. Key files: README.md, CONTRIBUTING.md, pyproject.toml, package.json (first 50 lines)
4. Git status: current branch, recent commits, uncommitted changes

Output: `RepoContext` object → written to Memory Index (Layer 1).

Rationale: This phase costs zero tokens but gives the LLM enough orientation to make a meaningful plan. Borrowed from Aider's repo-map concept.

**Phase 2: Plan (1 LLM call)**

Input: system prompt + Memory Index + user task description
Output: Structured JSON plan

```python
@dataclass
class PlanStep:
    step_number: int
    goal: str                      # What this step achieves
    target_files: list[str]        # Files likely to be read/modified
    expected_tools: list[str]      # Which tools this step will likely need
    budget: int = 15               # Max tool calls for this step
    success_criteria: str          # How to know this step is done

@dataclass
class Plan:
    summary: str                   # 1-sentence task summary
    steps: list[PlanStep]
    estimated_complexity: str      # low / medium / high
```

Plan granularity is deliberately coarse — "modify the auth module's token validation" not "change line 47 from == to !=". Fine-grained decisions happen in Execute.

**Phase 3: Execute (bounded ReAct loop per step)**

For each plan step:
1. **Pre-step hook** fires
2. **Git checkpoint** — auto-commit current state
3. **Bounded ReAct loop** — max `step.budget` tool calls
4. **Post-step evaluation:**
   - `completed` → checkpoint success, move to next step
   - `failed` → rollback to pre-step checkpoint, trigger replan
   - `budget_exceeded` → checkpoint partial progress (no rollback), trigger replan
5. **Post-step hook** fires

**Replan Mechanism:**

```python
def replan(context, failed_step, result, remaining_steps) -> list[PlanStep]:
    """Ask LLM to generate a new plan given:
    - What was attempted and why it failed
    - The diff of changes made so far
    - The remaining original steps
    Max replans per task: 3 (configurable)
    If exceeded: stop gracefully, report to user
    """
```

#### 3.4.3 Guard Rails

| Guard | Default | Action |
|---|---|---|
| Step budget (tool calls per step) | 15 | Force replan |
| Consecutive errors | 3 | Force replan |
| Repeated action (same tool + same params) | 2 | Inject warning, then force replan |
| Max file changes per step | 10 | Force replan |
| Max total tool calls per task | 100 | Stop, report to user |
| Max replans per task | 3 | Stop, report to user |

**Loop Detection:** Each tool call is hashed as `f"{tool_name}:{hash(params)}"`. If the same hash appears more than twice, the guardrail injects a system message: "You are repeating the same action. Try a different approach or declare this step failed."

#### 3.4.4 Hook System

```python
class HookEvent(Enum):
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    # P1 expansion:
    # SESSION_START, SESSION_END, PRE_PLAN, POST_PLAN,
    # CONTEXT_COMPACT, REPLAN, ERROR

@dataclass
class HookContext:
    event: HookEvent
    step: PlanStep | None
    tool_call: ToolCall | None
    result: ToolResult | None

class HookManager:
    def register(self, event: HookEvent, handler: Callable) -> None
    def fire(self, event: HookEvent, **kwargs) -> None
```

MVP: 4 hooks (pre/post step, pre/post tool). The hook interface is designed to be extended to 25+ events without changing the core API.

#### 3.4.5 Files

```
forge/agent/
├── __init__.py
├── orchestrator.py      # HybridOrchestrator — orient/plan/execute loop
├── orient.py            # RepoScanner — dir tree, signatures, key files
├── planner.py           # Planner — LLM-based plan generation + replan
├── executor.py          # ExecuteEngine — bounded ReAct + git checkpoint
├── guardrails.py        # GuardRails — budget, loop detection, error streaks
└── hooks.py             # HookManager — lifecycle event system
```

---

### 3.5 Sandbox / Runtime (`forge/sandbox/`)

#### 3.5.1 Purpose
Provide a safe execution environment for agent operations. Handle git checkpoints, process isolation, and file system boundaries.

#### 3.5.2 Git Checkpoint System

```python
class GitCheckpoint:
    FORGE_BRANCH = "forge/working"

    def setup(self) -> str:
        """Create working branch from current HEAD. Returns original branch name."""

    def checkpoint(self, message: str) -> str:
        """Stage all changes, commit, return SHA. Skip if no changes."""

    def rollback(self, to_sha: str | None = None) -> None:
        """Hard reset to specified checkpoint (default: previous)."""

    def get_diff_since(self, sha: str) -> str:
        """Get diff summary for LLM context."""

    def finalize(self, original_branch: str) -> None:
        """Squash-merge working branch back to original, delete working branch."""
```

**Checkpoint lifecycle in Execute:**
```
Step N start    →  checkpoint("before step N")
Step N success  →  checkpoint("completed step N")
Step N failure  →  rollback()  →  replan
Task complete   →  finalize()  →  squash merge to original branch
```

#### 3.5.3 Process Execution

```python
class ProcessRunner:
    async def run(
        self,
        command: str | list[str],
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> ProcessResult:
        """Run a subprocess with timeout and output capture.
        Returns: stdout, stderr, return_code, timed_out flag
        Truncates output to max_lines (default 200).
        """
```

#### 3.5.4 Docker Isolation (P2)

MVP runs directly on the host file system (with git as the safety net). P2 adds optional Docker isolation:
- Mount repo as a volume
- Run all bash commands inside the container
- Configurable base image (Python, Node, Go, etc.)
- Network access controls

#### 3.5.5 Files

```
forge/sandbox/
├── __init__.py
├── git_checkpoint.py    # GitCheckpoint — branch management, commit, rollback
├── process.py           # ProcessRunner — subprocess execution with timeout
└── docker.py            # DockerSandbox (P2) — container-based isolation
```

---

### 3.6 CLI Interface (`forge/cli/`)

#### 3.6.1 Purpose
Terminal-based interactive interface. User types natural language, sees agent thinking/acting in real-time with streaming output.

#### 3.6.2 Technology Choice

**Rich** for output formatting (syntax highlighting, panels, progress bars) + **Prompt Toolkit** for input (history, auto-complete). Not Ink/React (that's TypeScript territory). Not Textual (too heavy for MVP).

#### 3.6.3 UX Flow

```
$ forge "Add refresh token support to the auth module"

◆ Scanning repository...
  └─ 3 directories, 12 files, 847 lines

◆ Planning...
  Step 1: Understand current token validation logic
  Step 2: Modify validate_token to support refresh tokens
  Step 3: Add refresh token generation endpoint
  Step 4: Write and run tests

◆ Step 1/4: Understand current token validation logic
  ├─ [read] src/auth/token.py (lines 1-45)
  ├─ [search] "refresh" in src/
  └─ ✓ Complete (3 tool calls)

◆ Step 2/4: Modify validate_token to support refresh tokens
  ├─ [edit] src/auth/token.py — str_replace
  │   - def validate_token(token: str) -> bool:
  │   + def validate_token(token: str, token_type: str = "access") -> bool:
  ├─ [bash] python -m pytest tests/test_auth.py
  │   FAILED: 2 tests failed
  ├─ [edit] src/auth/token.py — fix assertion
  ├─ [bash] python -m pytest tests/test_auth.py
  │   PASSED: 5/5
  └─ ✓ Complete (4 tool calls)

◆ Done! Changes:
  └─ src/auth/token.py (+23, -5)
  └─ tests/test_auth.py (+45, -0)

? Apply changes to main branch? [Y/n]
```

#### 3.6.4 Streaming Display

LLM streaming output is rendered in real-time:
- **Thinking text** → gray, italic
- **Tool calls** → formatted as "├─ [tool_name] args"
- **Tool results** → truncated to 10 lines with expandable "[show more]"
- **Errors** → red panel
- **Success** → green checkmark

Streaming chunks from `LLMAdapter.chat_stream()` are consumed by a `StreamRenderer` that handles the display state machine.

#### 3.6.5 Slash Commands (MVP: 5)

| Command | Description |
|---|---|
| `/plan` | Show current plan and step status |
| `/history` | Show recent tool call history |
| `/rollback` | Undo last step (git rollback) |
| `/cost` | Show token usage and estimated cost |
| `/help` | List available commands |

#### 3.6.6 Files

```
forge/cli/
├── __init__.py
├── app.py               # Main CLI application — input loop, command routing
├── renderer.py          # StreamRenderer — real-time display of LLM/tool output
├── commands.py          # Slash command handlers
└── themes.py            # Color/style configuration
```

---

## 4. Project Directory Structure (Complete)

```
forge/
├── forge/
│   ├── __init__.py              # Version, top-level exports
│   ├── config.py                # Configuration loading (TOML/env vars)
│   ├── main.py                  # Entry point — CLI bootstrap
│   │
│   ├── llm/                     # § 3.1 — LLM Adapter
│   │   ├── __init__.py          # create_llm() factory
│   │   ├── base.py              # ABC, data classes (LLMResponse, Message, etc.)
│   │   ├── anthropic_adapter.py
│   │   ├── openai_adapter.py
│   │   ├── retry.py             # RetryPolicy
│   │   └── token_counter.py
│   │
│   ├── tools/                   # § 3.2 — Tool System
│   │   ├── __init__.py
│   │   ├── base.py              # Tool ABC, ToolRegistry, ToolResult
│   │   ├── bash.py
│   │   ├── file_read.py
│   │   ├── file_edit.py         # str_replace + create_file
│   │   ├── search.py            # ripgrep wrapper
│   │   ├── git.py               # (P1)
│   │   ├── test_runner.py       # (P1)
│   │   └── mcp_adapter.py       # MCP protocol client
│   │
│   ├── context/                 # § 3.3 — Context Manager
│   │   ├── __init__.py
│   │   ├── manager.py           # ContextManager
│   │   ├── memory.py            # 3-layer memory (index + topics)
│   │   ├── compactor.py         # 5-stage compression pipeline
│   │   └── history.py           # JSONL session history + search
│   │
│   ├── agent/                   # § 3.4 — Orchestrator
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # HybridOrchestrator
│   │   ├── orient.py            # RepoScanner
│   │   ├── planner.py           # LLM-based planning + replan
│   │   ├── executor.py          # Bounded ReAct + checkpoints
│   │   ├── guardrails.py        # Budget, loop detection
│   │   └── hooks.py             # HookManager
│   │
│   ├── sandbox/                 # § 3.5 — Runtime
│   │   ├── __init__.py
│   │   ├── git_checkpoint.py
│   │   ├── process.py           # Subprocess execution
│   │   └── docker.py            # (P2) Docker isolation
│   │
│   └── cli/                     # § 3.6 — CLI Interface
│       ├── __init__.py
│       ├── app.py               # Main loop
│       ├── renderer.py          # Streaming display
│       ├── commands.py          # Slash commands
│       └── themes.py
│
├── tests/
│   ├── test_llm/
│   │   ├── test_anthropic_adapter.py
│   │   ├── test_openai_adapter.py
│   │   ├── test_retry.py
│   │   └── test_message_translation.py
│   ├── test_tools/
│   │   ├── test_bash.py
│   │   ├── test_file_edit.py
│   │   ├── test_search.py
│   │   └── test_registry.py
│   ├── test_context/
│   │   ├── test_manager.py
│   │   ├── test_compactor.py
│   │   └── test_memory.py
│   ├── test_agent/
│   │   ├── test_orchestrator.py
│   │   ├── test_executor.py
│   │   ├── test_guardrails.py
│   │   └── test_planner.py
│   └── test_sandbox/
│       ├── test_git_checkpoint.py
│       └── test_process.py
│
├── forge.toml                   # Default configuration
├── pyproject.toml               # Package metadata + dependencies
├── README.md
└── LICENSE
```

---

## 5. Configuration (`forge.toml`)

```toml
[forge]
name = "forge"
version = "0.1.0"

[llm]
provider = "anthropic"                # anthropic | openai | openrouter
model = "claude-sonnet-4-20250514"    # Model identifier
temperature = 0.0
max_tokens = 4096
timeout = 120.0
max_retries = 3
backoff_base = 1.0

[llm.openrouter]
# Used when provider = "openrouter"
model = "deepseek/deepseek-chat"
base_url = "https://openrouter.ai/api/v1"

[agent]
max_tool_calls_per_step = 15
max_consecutive_errors = 3
max_repeated_actions = 2
max_file_changes_per_step = 10
max_total_tool_calls = 100
max_replans = 3

[context]
compaction_threshold = 0.7            # Trigger at 70% of max context
max_tool_result_lines = 200           # Truncate tool output
history_keep_full = 5                 # Keep last N tool results in full

[sandbox]
use_docker = false                    # P2 feature
git_auto_checkpoint = true
bash_timeout = 30.0
bash_max_output_lines = 200

[tools.permission_defaults]
file_read = "auto"
file_edit = "confirm_once"
file_create = "confirm_once"
bash = "always_confirm"
search = "auto"
```

---

## 6. Dependencies

```toml
[project]
name = "forge"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    # LLM SDKs
    "anthropic>=0.40.0",
    "openai>=1.50.0",

    # CLI
    "rich>=13.0",
    "prompt-toolkit>=3.0",

    # Data validation
    "pydantic>=2.0",

    # Async
    "anyio>=4.0",

    # Configuration
    "tomli>=2.0;python_version<'3.11'",

    # Token counting (OpenAI)
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
    "ruff>=0.8",
    "mypy>=1.13",
]

mcp = [
    "mcp>=1.0",
]

docker = [
    "docker>=7.0",
]

[project.scripts]
forge = "forge.main:main"
```

---

## 7. Development Roadmap

### Phase 1: Foundation (Week 1-2) — "It runs"

**Goal:** End-to-end flow works: user types a task → agent reads files → edits code → user sees output.

| Day | Task | Deliverable |
|---|---|---|
| 1-2 | LLM adapter: base.py + anthropic_adapter.py + retry.py | Can call Claude API with tool-use and streaming |
| 3 | Tool system: base.py + ToolRegistry | Tool interface defined, registry works |
| 4 | P0 tools: bash.py + file_read.py + file_edit.py + search.py | 4 tools registered and executable |
| 5-6 | Simple ReAct loop (no plan/orient yet) | Agent can receive a task and use tools in a loop |
| 7-8 | CLI shell: basic input → stream output | User can interact with agent in terminal |
| 9-10 | Git checkpoint: git_checkpoint.py | Auto-commit before each tool call, rollback on failure |
| 11-12 | Integration testing + bug fixing | End-to-end demo: "fix the bug in X" works |

**Milestone:** Demo video of Forge fixing a real bug in a small repo.

### Phase 2: Intelligence (Week 3-4) — "It thinks"

**Goal:** Hybrid loop with Orient/Plan/Execute. Context management prevents degradation on longer tasks.

| Day | Task | Deliverable |
|---|---|---|
| 13-14 | Orient phase: repo scanner | Auto-generates repo context from directory + signatures |
| 15-16 | Plan phase: planner.py | LLM generates structured JSON plan before execution |
| 17-18 | Execute with bounded ReAct + guardrails | Budget limits, loop detection, replan trigger |
| 19-20 | Context manager: memory.py + compactor.py | 3-layer memory + at least stages 1-2 of compaction |
| 21-22 | OpenAI adapter | Provider-agnostic: switch between Claude and GPT with a flag |
| 23-24 | Hook system (4 MVP hooks) + P1 tools (git, test_runner) | Extensible lifecycle + richer tool set |
| 25-28 | Evaluation on 10 SWE-bench Lite instances | Quantitative baseline: X/10 resolved |

**Milestone:** Forge handles a 20-minute coding session without context degradation. SWE-bench baseline established.

### Phase 3: Polish & Differentiation (Month 2-3) — "It's good"

| Task | Description |
|---|---|
| Full 5-stage compaction pipeline | Stages 3-5 (LLM-based summarization, history compression) |
| OpenRouter adapter | DeepSeek, Qwen via OpenRouter — test on diverse models |
| MCP adapter | Connect external tool servers via MCP protocol |
| SWE-bench Verified evaluation | Run on full benchmark, optimize prompt engineering |
| Prompt cache optimization | Implement cache-aware message assembly for Anthropic |
| Docker sandbox | Optional container isolation for bash execution |
| Go rewrite of hot path (optional) | Rewrite sandbox/process runner in Go for performance story |

### Phase 4: Portfolio & Job Prep (Month 3+)

| Task | Description |
|---|---|
| README + architecture docs | Clear documentation for GitHub portfolio |
| Blog post | "Building a Coding Agent Framework: What I Learned from Claude Code's Architecture" |
| Demo video | 5-min walkthrough showing Forge solving a real issue |
| SWE-bench leaderboard submission | Public benchmark score |
| Interview prep | System design walkthrough of Forge's architecture, tradeoff discussions |

---

## 8. Key Technical Design Decisions & Rationale

This section documents the "why" behind each major decision — the part interviewers care about most.

### 8.1 Why Hybrid Loop over Pure ReAct?

**Observation:** Pure ReAct (SWE-agent style) has no global planning. The agent makes locally optimal decisions that are globally suboptimal — it might spend 10 tool calls exploring the wrong file before realizing the bug is elsewhere.

**Observation:** Pure Plan-then-Execute (Devin style) makes plans before seeing the code. Plans become stale the moment the agent discovers unexpected complexity.

**Decision:** Hybrid with lightweight Orient (zero-cost context gathering) + coarse-grained Plan (big-picture direction) + tactical Execute (detailed ReAct within each step). Replan on failure ensures the plan adapts to reality.

**Tradeoff acknowledged:** The Plan phase adds 1 LLM call of overhead. For trivial tasks ("fix this typo"), it's wasted. Could add a complexity classifier that skips planning for simple tasks — but MVP keeps it simple and always plans.

### 8.2 Why Custom LLM Adapter over litellm?

**Observation:** litellm caused connection errors and hangs during SWE-bench evaluation. Root causes: (1) litellm retries non-retryable errors (400 Bad Request), (2) its timeout defaults don't match provider behavior, (3) tool-use schema translation has edge cases that silently produce invalid payloads.

**Decision:** Thin adapters per provider. Each adapter is ~150 lines, explicitly handles that provider's quirks, and uses our own retry policy that only retries network/rate-limit errors.

**Tradeoff acknowledged:** More code to maintain (one adapter per provider vs. one litellm dependency). But each adapter is simple enough to be fully understood, and adding a new provider is a 2-hour task of implementing 3 abstract methods.

### 8.3 Why Git Checkpoints Instead of Undo History?

**Observation:** Agent edits are often multi-file and multi-step. An undo stack tracks individual edits, but a coding task needs atomic rollback of "everything the agent did in step 3."

**Decision:** Git is the undo system. Every plan step starts with a commit. Rollback = `git reset --hard`. Finalize = squash merge. This gives us atomic rollback, diff visibility, and branch isolation for free.

**Tradeoff acknowledged:** Requires git initialized in the working directory. For non-git repos, the checkpoint system falls back to file-level snapshots (copy-on-write). Also, auto-commits pollute the reflog — finalize squash-merges to keep history clean.

### 8.4 Why 3-Layer Memory over Simple Truncation?

**Observation:** SWE-agent truncates context when it gets too long — losing critical early observations. Claude Code built a 3-layer memory system that keeps essential information always available while offloading details.

**Decision:** Memory Index (~500 tokens) is always present. Topic files are on-demand. Session history is searchable but never bulk-loaded. This means the LLM always knows "where it is" even after compaction.

**Tradeoff acknowledged:** The memory index needs to be kept accurate — "strict write discipline" (only update after successful operations) prevents stale/speculative entries. If the index gets wrong, the agent is worse off than having no index. Mitigation: the index is overwritten from ground truth (actual file system state) at the start of each plan step.

### 8.5 Why Read/Write Tool Separation?

**Observation:** Claude Code's leaked source shows read-only tools execute concurrently, write tools execute serially. This is a deliberate architectural choice.

**Decision:** Each tool declares its `MutationType` (READ, WRITE, EXECUTE). The executor groups concurrent reads and serializes writes. This is correct by construction — no race conditions possible.

**Tradeoff acknowledged:** Some "reads" have side effects (e.g., a bash command that reads but also writes to stdout). The MutationType is declared by the tool author, not verified. Incorrect declaration → potential bugs. Mitigation: bash is always classified as EXECUTE (serial) regardless of the command content.

### 8.6 Why Permission Gates?

**Decision:** Three-tier permission system: AUTO (reads), CONFIRM_ONCE (file writes — ask once, then auto-approve same tool), ALWAYS_CONFIRM (bash — always ask). This balances safety with usability.

**Rationale:** A coding agent that asks "are you sure?" on every file read is unusable. One that silently runs `rm -rf /` is dangerous. The three tiers match user expectations: "of course you can read files, just don't break things without asking."

---

## 9. Innovation & Differentiation

What makes Forge different from existing tools — the story for interviews and the README:

1. **Hybrid Orient→Plan→Execute loop with automatic replanning.** No other open-source framework combines zero-cost orientation, LLM-based planning, bounded execution, and failure-driven replanning in a single coherent loop.

2. **Provider-agnostic from the ground up.** Not a litellm wrapper — purpose-built adapters that handle each provider's quirks honestly. Switch between Claude, GPT-4o, and DeepSeek with one config line.

3. **3-layer memory architecture inspired by (but not copied from) Claude Code.** Lightweight memory index always in context + on-demand topic files + searchable history. Paired with a 5-stage gradient compression pipeline that degrades gracefully instead of cliff-edge truncation.

4. **Git-native safety model.** Every mutation is preceded by a checkpoint. Rollback is atomic. The working branch is isolated. The user's code is never in danger.

5. **MCP-compatible tool system.** Core tools are native Python for performance; external tools plug in via MCP. The LLM sees a unified interface.

6. **Extensible via hooks, not forks.** Pre/post lifecycle events at every critical point. Build a custom SWE-bench harness, a cost tracker, or a logging pipeline without touching core code.

---

## 10. Prompts (System Prompts for Each Phase)

### 10.1 Plan Phase System Prompt

```
You are Forge, an expert coding agent. You are given a repository context and a user task.

Your job is to create a structured plan to accomplish the task.

## Repository Context
{memory_index}

## Rules
- Output ONLY a JSON object matching the schema below. No other text.
- Each step should be a coarse-grained goal ("modify the auth module"), not a fine-grained action ("change line 47").
- Steps should be ordered logically: understand first, then modify, then test.
- Estimate which files each step will need to read or modify.
- Set a realistic budget (max tool calls) for each step. Simple reads: 3-5. Complex edits: 10-15.

## Output Schema
{plan_json_schema}
```

### 10.2 Execute Phase System Prompt

```
You are Forge, an expert coding agent executing step {step_number} of a plan.

## Current Step
Goal: {step.goal}
Target files: {step.target_files}
Success criteria: {step.success_criteria}

## Repository Context
{memory_index}

## Available Tools
{tool_schemas}

## Rules
- Focus ONLY on the current step's goal. Do not work ahead.
- Read files before editing them. Understand before you modify.
- After editing, verify your changes (run tests if available).
- If you realize the step's goal is wrong or impossible, respond with:
  {"status": "failed", "reason": "explanation"}
- When the step is complete, respond with:
  {"status": "completed", "summary": "what was done"}
- You have a budget of {step.budget} tool calls. Use them wisely.
```

### 10.3 Replan System Prompt

```
You are Forge, an expert coding agent. A step in your plan has failed or exceeded its budget.

## What happened
Step {step_number}: {step.goal}
Status: {result.status}
Reason: {result.failure_reason or "budget exceeded"}
Tool calls used: {result.tool_calls_used}
Key observations: {result.observations}

## Changes made so far
{git_diff_summary}

## Remaining original plan
{remaining_steps}

## Task
Create a revised plan for the remaining work. Learn from what failed — do not repeat the same approach. Output ONLY a JSON plan object.
```

### 10.4 Context Compaction Prompt (Stage 3: Extract Session Memory)

```
You are a memory extraction assistant. Given a conversation between an agent and its tools, extract the key findings and decisions into a concise summary.

Focus on:
- What files were examined and what was learned about them
- What changes were made and why
- What worked and what didn't
- Any important constraints or edge cases discovered

Do NOT include raw tool output. Summarize in your own words.
Output a markdown summary, max 500 words.
```

---

## 11. Testing Strategy

### 11.1 Unit Tests (per module)

| Module | Key Test Cases |
|---|---|
| LLM Adapter | Message translation (Anthropic/OpenAI format differences), tool schema conversion, retry on retryable errors only, streaming chunk parsing, stop_reason mapping |
| Tool Registry | Registration + lookup, read/write separation (concurrent reads, serial writes), permission gate enforcement |
| Tools (each) | Happy path, error handling, output truncation, timeout behavior |
| Context Manager | Token counting accuracy, compaction trigger threshold, memory index update discipline, message assembly order |
| Compactor | Each stage independently (stages 1-2 deterministic, stages 3-4 mocked LLM), never drops system prompt or memory index |
| Orchestrator | Orient produces valid RepoContext, Plan parsing handles malformed JSON, Execute respects budget, replan triggers correctly |
| GuardRails | Loop detection (repeated action hash), error streak counting, budget enforcement |
| Git Checkpoint | Checkpoint creates commit, rollback restores state, finalize squash-merges cleanly |

### 11.2 Integration Tests

| Test | Description |
|---|---|
| End-to-end simple fix | Give agent a repo with a known bug + failing test. Agent should fix bug and pass test. |
| End-to-end with replan | Give agent a misleading task that requires replanning. Verify replan triggers and succeeds. |
| Context pressure | Run agent on a task that generates >100 tool calls. Verify compaction fires and agent doesn't crash. |
| Provider switch | Run same task on Anthropic and OpenAI. Verify both produce valid results. |
| Git safety | Verify agent changes are isolated to forge/working branch. Original branch is untouched until finalize. |

### 11.3 Benchmarks (P1-P2)

- SWE-bench Lite (300 instances): target 20%+ resolve rate in Phase 2
- SWE-bench Verified (500 instances): target in Phase 3
- Custom micro-benchmarks: "fix this TypeError", "add a unit test", "refactor this function"

---

## 12. Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| Scope creep (trying to build too much) | Delays MVP, never ships | Strict P0/P1/P2 prioritization. Phase 1 deliverable is a working demo, not a perfect system. |
| LLM API instability | Agent hangs or crashes | Self-managed retry with backoff. Timeout kills. Graceful degradation (report to user, don't crash). |
| Agent breaks user's code | Loss of user trust | Git checkpoint before EVERY mutation. Working branch isolation. Permission gates on destructive ops. |
| Context window overflow | Agent loses coherence | 5-stage compaction. Memory index ensures minimum viable context is always preserved. |
| Agent infinite loops | Wasted tokens and time | Budget per step (15), consecutive error limit (3), loop detection (repeated action hash), global task limit (100). |
| Poor SWE-bench performance | Weak portfolio signal | Invest in prompt engineering. Compare against SWE-agent baseline on same instances. Even matching SWE-agent with a cleaner architecture is a valid story. |

---

## 13. For Claude Code: Implementation Notes

When implementing this plan, follow these guidelines:

1. **Start with `forge/llm/base.py`** — define all data classes and the ABC first. This establishes the type contracts that every other module depends on.

2. **Use Pydantic v2** for all data validation (tool inputs, plan schemas, config). Use `model_validator` for complex constraints.

3. **Use `asyncio` throughout** — every LLM call and tool execution is async. The CLI runs an async event loop.

4. **Type everything** — strict mypy compliance. No `Any` types except in JSON passthrough.

5. **Tests are not optional** — write tests alongside implementation, not after. Target >80% coverage on core modules (llm, tools, context, agent).

6. **Each module should be independently testable** — use dependency injection (pass LLMAdapter, ToolRegistry, etc. as constructor arguments, not global imports).

7. **Use `logging` module** with structured logging (module name, event type). No print() statements.

8. **Error handling philosophy**: catch specific exceptions, not bare `except`. Let unexpected errors propagate. Log context before re-raising.

9. **Git commits during development** should follow conventional commits: `feat(llm): add anthropic adapter`, `test(tools): add bash tool tests`, etc.

10. **README.md** should be written in Phase 1 and updated continuously. Include: what Forge is, how to install, how to use, architecture diagram, how to contribute.
