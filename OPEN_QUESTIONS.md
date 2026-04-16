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

To address the inherent non-determinism of LLM agents, we need to move beyond traditional software engineering paradigms (like standard Git) and build infrastructure explicitly designed for probabilistic systems.

* **1. Evaluation-Driven Version Control:** 
  * *Problem:* Traditional Git tracks text diffs, but in agent development, a 3-word prompt change can alter system behavior drastically and unpredictably across the entire codebase.
  * *Idea:* Version control must track performance tuples: `(Prompt/System_Diff, Execution_Trace, Evaluation_Pass_Rate)`. We need a mechanism to rollback not just code, but to the semantic baseline that yielded the highest pass rate on the DiffVerify harness. Prompt changes should be gated by statistical confidence.
* **2. Event Sourcing & Trajectory Replay (Time-Travel Debugging):**
  * *Problem:* When an agent hallucinates or creates a destructive patch, the bug is often impossible to reproduce due to model temperature and non-determinism.
  * *Idea:* Treat every agent step as an immutable event. Record the full `[Observation -> Thought -> Action -> Environment_State]` trajectory. This enables "time-travel debugging" where developers can replay the exact context window that led to a failure, fork the trajectory at the exact point of hallucination, and test if a new guardrail prevents it.
* **3. Causal DAG (Directed Acyclic Graph) Context Tracking:**
  * *Problem:* The linear nature of LLM chat limits reasoning. When an agent goes down a rabbit hole (e.g., in Eureka Broadcast), the failed thoughts remain in the context window, causing "Hallucination Contagion" and wasting tokens.
  * *Idea:* Upgrade the agent's working memory from a linear thread to a Causal DAG. Each thought/action is a node. If an exploration branch hits a dead end (e.g., a test fails), we can structurally prune that entire branch (and all its downstream context) from the active window, cleanly backtracking without poisoning future reasoning. Context becomes a "search tree" rather than a "chat log".
