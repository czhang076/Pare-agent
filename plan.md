# Pare 2.0 — The Opinionated Coding Agent Layer

## Project Plan & Technical Specification

**Author:** Chenz  
**Date:** April 2026  
**Status:** Pivot to Pare 2.0 (Trajectory Inspector & Langfuse Integration) MVP  
**Language:** Python 3.12+  
**License:** MIT  

---

## 1. What is Pare 2.0?

Pare 2.0 is **an opinionated, coding-agent-specific layer built on top of modern observability infrastructure (like Langfuse)**. 

While generic observability platforms like Langfuse or Smith provide excellent event sourcing, LLM traces, and webhook capabilities, they do not understand the specific domain taxonomy of autonomous software engineers (e.g., SWE-bench fault modes, Git worktree states, or bidirectional test validations). 

Pare 2.0 bridges this gap. It does not attempt to reinvent generic AgentOps infrastructure or build a universal "multi-agent operating system." Instead, it acts as a highly specialized lens and execution runner that provides:
1. **Eval-Gated Prompt CI:** Using Langfuse webhooks to gate prompt promotions based strictly on real test pass rates from coding sandboxes.
2. **Coding-Agent-Aware Trajectory Inspector (MVP):** A specialized diff-viewer for agent traces that explains *why* an agent failed a SWE-bench task and *where* it diverged from a successful run.
3. **Causal DAG Context:** Upgrading agent runtime memory to structurally prune dead-end branches, while pushing for upstream OSS contributions to native Agent Graphs in observability platforms.

### 1.1 One-line pitch

> **The specialized CI and trajectory analysis layer for coding agents, built on top of Langfuse.**

---

## 2. Core Architecture & Modules

### 2.1 The Trajectory Inspector (Product Focus)
Generic trace visualizers (like Langfuse's waterfall or native Agent Graph) treat all LLM steps equally. They cannot tell you if a step was a "Syntax Error Loop," a "Context Hallucination," or a "Valid Regression Check."
* **The Concept:** Pare ingests SWE-bench style JSONL trajectory logs (or fetches traces from Langfuse) and applies a coding-specific failure classifier.
* **The Value:** It answers exactly: *"What category of coding failure is this?"* and *"Where did the failed trajectory branch off from the successful one?"*
* **Implementation:** A headless CLI tool (`pare inspect`) that outputs a clean, diff-based HTML report showing the timeline, expanded tool-calls, and semantic failure tags.

### 2.2 Eval-Gated Prompt CI (via Langfuse & DiffVerify)
*This is Pare's CI/CD workflow.*
* **The Concept:** Moving beyond "LLM-as-a-judge" Evals. A prompt is only good if it can make an agent fix a real codebase without breaking existing tests.
* **The Implementation:** We stitch together Langfuse's native `staging`/`production` labels with GitHub Actions and Pare's **DiffVerify harness** (bidirectional test validation). 
* **The Flow:** 
  1. Prompt updated to `staging` in Langfuse.
  2. Webhook triggers Pare to run the agent against a subset of SWE-bench tests.
  3. Pare executes the **Backward Pass** (tests fail on buggy code) and **Forward Pass** (tests pass on fixed code).
  4. If successful, Pare invokes the Langfuse API to automatically tag the prompt as `production`.

### 2.3 Causal DAG Runtime Context
*This is Pare’s agent runtime modification.*
* Linear chat histories poison the agent's context window with failed experiments (Hallucination Contagion).
* Pare 2.0 upgrades the active agent runtime to use a Causal DAG, structurally pruning dead-end branches (e.g., when a test fails).
* **Open Source Strategy:** This exposes a semantic gap in current observability tools. Pare will explicitly tag pruned paths (`metadata.agent_status = 'dead_end'`) in its telemetry and drive upstream contributions to Langfuse’s Agent Graph to visually de-emphasize these dead-end exploration branches.

---

## 3. Four-Week Execution Roadmap

This roadmap focuses entirely on executing the "Opinionated Framework atop Langfuse" pivot.

### Week 1-2: Trajectory Inspector MVP (The CLI)
**Goal:** Build the core lens that differentiates Pare from generic Waterfall trace UI.
- [ ] Build `pare inspect traj_success.jsonl traj_failed.jsonl --diff` CLI command.
- [ ] Ingest standard SWE-bench style JSONL trajectories.
- [ ] Implement the timeline view with expandable tool-calls.
- [ ] Integrate the research branch's failure classifier (assigning SWE-bench taxonomic tags to trajectory segments).
- [ ] Output a lightweight HTML report showing side-by-side trajectory diffs and the exact divergence point.

### Week 3: Langfuse Integration & Thought Leadership (The Signal)
**Goal:** Prove the architectural fit with Langfuse and establish domain expertise.
- [ ] Integrate Pare's execution telemetry to emit standard traces to Langfuse.
- [ ] Render a Pare agent run inside Langfuse's native Agent Graph.
- [ ] Document the gaps in current Agent Graph semantics (specifically the lack of `dead-end` branch devaluation).
- [ ] Write a technical blog post: *"Using Langfuse as the Backbone for Coding Agent Observability"* (focusing on the AgentOps toolchain division of labor).

### Week 4: Langfuse Upstream Contribution (The OSS Impact)
**Goal:** Convert the observed observability gaps into a high-signal OSS contribution.
- [ ] Create a minimalist demo repo demonstrating the difference between a raw Langfuse waterfall and a DAG with pruned dead-ends.
- [ ] Open a high-quality Issue / RFC on the Langfuse repository.
- [ ] Proposed API: Suggest `metadata.agent_status ∈ {success, dead_end, retrying}` semantics for Langfuse's Agent Graph, with native opacity/rendering adjustments for `dead_end` nodes.

### Post-MVP (Backlog)
- [ ] **Fully Automated Eval-Gated CI:** Connect the Langfuse webhook `repository_dispatch` to Pare's DiffVerify sandbox pipeline.
- [ ] **Time-Travel Debugging SDK:** Build utilities to fetch a Langfuse trace and resurrect the exact Git worktree + Docker state at a specific span ID for manual prompt debugging.
