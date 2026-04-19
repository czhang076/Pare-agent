# OPEN QUESTIONS (Phase 3: Multiverse Repair Engine)

Before proceeding with the full implementation of the Multiverse Engine (Phase 3), the following design problems must be resolved based on research pilot data (do not over-engineer yet):

* **1. "First to Green Wins" vs. Review-Ready Output:** Blind auto-merging relies heavily on test adequacy. Multiverse should likely output a "Candidate Patch + human review" PR rather than auto-merging to `main`.
* **2. Ghost Sandbox State Leakage:** `git worktree` only solves file-level isolation. Testing concurrently with shared ports, `/tmp` temporary files, and DB locks will crash.
* **3. Token Economics:** Running 3 top-tier commercial models concurrently causes massive token burn. Requires strict fail-fast rules, budget caps, or dynamic routing to smaller/local models initially.
* **4. Eureka Broadcast Context Poisoning:** Injecting failure traces straight into another Agent's context window can disrupt ongoing tool usage or spread "hallucination contagion". We need an intelligent blackboard rather than naive message passing.
  * *Resolution:* Do NOT use LLM-based gating (adds latency, contradicts deterministic goals). Use strict rule-based frequency and type limits (only broadcast system-level faults like `IMPORT_ERROR`, never `TEST_FAILURE`).
* **5. Attacker Agent Isolation (Deferred):** Full multi-agent architectures (where an independent "Attacker Agent" tries to break the "Fixer Agent's" code) are too high-overhead for MVP. 
  * *Resolution:* DiffVerify MVP will implement "Prompt Isolation" (different system prompts for test generation vs. patch generation) to solve 80% of reward hacking natively. Full Attacker Agent architectures are deferred to late Phase 3 or beyond.

## Pare 2.0 & Next-Gen AgentOps Infrastructure (Future Vision)

## Pare 2.0: The Opinionated Coding Agent Layer (Future Vision)

Rather than building a "universal AgentOps platform" from scratch, Pare 2.0 is envisioned as an **opinionated, coding-agent-specific layer built on top of modern observability infrastructure (like Langfuse)**. It bridges the gap between generic LLM tracing and the specialized needs of autonomous software repair.

* **1. Eval-Gated Prompt CI (via Langfuse & DiffVerify):** 
  * *Repositioning:* Instead of building a new "Evaluation-Driven Version Control" system from scratch, Pare 2.0 will utilize native Langfuse CI webhooks and prompt management. We will stitch together Langfuse's staging/production labeling with Pare's `DiffVerify` harness. Prompt upgrades will only be promoted if they pass coding-agent specific sandboxed test suites (e.g., SWE-bench FAIL_TO_PASS rules).
* **2. Coding-Agent-Aware Trajectory Inspector (Product Focus):**
  * *Repositioning:* Generic trace visualizers (like Langfuse's waterfall or vanilla Agent Graph) do not understand SWE-bench fault taxonomies. Pare will act as a specialized lens that ingests standard trace events and answers: *"What category of coding failure is this?"* and *"Where did the failed trajectory diverge from the successful one?"* This provides a domain-specific Diff View for agent runs.
* **3. Causal DAG Runtime Context & Upstream Contributions:**
  * *Problem:* Linear chat histories poison the agent's context window with failed experiments (Hallucination Contagion).
  * *Idea:* Upgrade the active agent runtime to use a Causal DAG, structurally pruning dead-end branches.
  * *Open Source Strategy:* This runtime feature exposes a gap in current observability tools. We will use this to drive upstream contributions (e.g., proposing `metadata.agent_status = dead_end` semantics to Langfuse's Agent Graph) to visually de-emphasize pruned exploration branches, creating high-signal OSS impact without reinventing the backend.
