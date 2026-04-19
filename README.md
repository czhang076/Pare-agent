# Pare 2.0

**The opinionated coding-agent observability + CI layer, on top of Langfuse.**

Pare 2.0 doesn't generate code and doesn't run agents. It explains *why* an
agent failed a SWE-bench task, *where* it diverged from a successful run,
and gates prompt promotions on whether they actually fix bugs in real
sandboxes — using Langfuse as the underlying event store.

See [`plan.md`](./plan.md) for the full spec.

## Status

R0 scaffold. The runtime primitives (loop, container, classifiers) live on
the research branch [`claude/great-carson-333acd`](../../tree/claude/great-carson-333acd)
and are consumed here as a library via the `[research]` extra.

The legacy QA & Multiverse Repair codebase is preserved on
[`legacy/product-version`](../../tree/legacy/product-version) for archaeology.

## Roadmap

| Week | Deliverable |
|------|-------------|
| W1-W2 | `pare inspect` CLI MVP — single-trajectory + diff mode + HTML report |
| W3    | Langfuse emitter + `metadata.agent_status` semantics |
| W4    | Langfuse upstream RFC + demo repo for Causal DAG rendering |
| Post-MVP | Eval-Gated Prompt CI via DiffVerify |

## Install (once published)

```bash
pip install pare
pip install 'pare[research]'   # pulls runtime primitives from the research branch
pip install 'pare[langfuse]'   # W3+ Langfuse integration
```

## Usage (target)

```bash
pare inspect traj.jsonl                       # classify a single trajectory
pare inspect ok.jsonl fail.jsonl --diff       # side-by-side with divergence point
pare inspect --langfuse-trace <id>            # pull from Langfuse
```
