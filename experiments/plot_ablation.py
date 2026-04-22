"""Ablation figure generator for Pare trajectory pilots.

Reads N arm JSONL triples (trajectories + labels + summary) produced by
``experiments.generate_trajectories`` + ``experiments.classify_trajectories``,
and emits publication-oriented PNG figures comparing the arms.

Typical invocation (3-arm ablation)::

    python -m experiments.plot_ablation \\
        --arm baseline:data/sympy20/arm1_baseline.jsonl \\
        --arm prepasses:data/sympy20/arm2_prepasses.jsonl \\
        --arm full:data/sympy20/arm3_full.jsonl \\
        --out-dir data/sympy20/figures

The script requires each ``<arm>.jsonl`` to have a sibling
``<arm>.labels.jsonl`` (classifier output). If missing, the arm is skipped
with a loud error — we never silently fill in defaults because Liu-category
plots would be nonsense without them.

Design notes
------------

- **matplotlib only, no seaborn** — one fewer dep, avoids style-sheet drift
  between envs.
- **Aggregation lives in pure functions** so tests can cover the maths
  without touching matplotlib. The ``_render_*`` functions are the only
  side-effecting code; swapping in another backend (plotly for HTML
  reports) would mean re-implementing those five functions.
- **Fail loud on missing labels** — ablation figures that hide missing
  data mislead reviewers more than they help. If labels are missing,
  produce a smaller figure set with a banner rather than fabricate
  outcome=failed defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Input dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmData:
    """One arm's raw trajectory + classifier output, pre-aggregation.

    ``labels`` is keyed by ``trajectory_id`` so ``AggregateArm`` can join
    labels to trajectories without assuming list order matches between the
    two JSONLs (which the classifier preserves today but shouldn't be
    relied on — we'd rather a missing trajectory_id yell than silently
    drop stats).
    """

    name: str
    trajectories: list[dict]
    labels: dict[str, dict]  # trajectory_id -> label row


@dataclass(frozen=True, slots=True)
class AggregateArm:
    """Per-arm metrics — every scalar that ends up in a figure.

    Adding a new figure should extend this struct first so the unit tests
    pin the computation before the plot code sees it. The plotting
    functions read ONLY from ``AggregateArm`` — they never touch raw
    trajectory dicts — which is what lets them be trivially testable
    with fixture dicts.
    """

    name: str
    n_runs: int
    outcome_counts: dict[str, int]
    avg_tool_calls: float
    avg_edits: float
    avg_bash: float
    edit_bash_ratio: float  # mean ratio across runs, bash=0 → edit count
    avg_input_tokens: float
    avg_output_tokens: float
    success_rate: float
    recovery_rate: float  # fraction of trajectories with contains_recovery=True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_arm(name: str, trajectory_jsonl: Path) -> ArmData:
    """Load one arm from its trajectory JSONL + sibling labels JSONL.

    Raises FileNotFoundError with a concrete remediation string if labels
    are missing — "run classify_trajectories first" beats guessing at
    outcomes.
    """
    if not trajectory_jsonl.exists():
        raise FileNotFoundError(f"trajectory jsonl missing: {trajectory_jsonl}")

    labels_path = trajectory_jsonl.with_suffix(".labels.jsonl")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels jsonl missing: {labels_path}\n"
            f"Run: python -m experiments.classify_trajectories "
            f"--trajectory-jsonl {trajectory_jsonl}"
        )

    trajectories: list[dict] = []
    with open(trajectory_jsonl, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                trajectories.append(json.loads(s))

    labels: dict[str, dict] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                row = json.loads(s)
                labels[row["trajectory_id"]] = row

    return ArmData(name=name, trajectories=trajectories, labels=labels)


# ---------------------------------------------------------------------------
# Aggregation (pure, testable)
# ---------------------------------------------------------------------------


def _count_tool(events: list[dict], name: str) -> int:
    return sum(1 for e in events if e.get("tool_name") == name)


def _count_edits(events: list[dict]) -> int:
    """file_edit + file_create — semantically both are edits for B2.1 purposes."""
    return sum(
        1 for e in events
        if e.get("tool_name") in ("file_edit", "file_create")
    )


def aggregate_arm(arm: ArmData) -> AggregateArm:
    """Fold one arm's raw data into publishable scalars.

    Edit/bash ratio uses ``edits / max(bash, 1)`` then averaged across
    runs — keeps divide-by-zero defined as "all edits no bash", which is
    the B2.1 Wrong-Fix signature we actually want to flag.
    """
    n = len(arm.trajectories)
    if n == 0:
        return AggregateArm(
            name=arm.name, n_runs=0, outcome_counts={},
            avg_tool_calls=0.0, avg_edits=0.0, avg_bash=0.0,
            edit_bash_ratio=0.0, avg_input_tokens=0.0, avg_output_tokens=0.0,
            success_rate=0.0, recovery_rate=0.0,
        )

    outcome_counts: Counter[str] = Counter()
    tool_totals = 0
    edit_totals = 0
    bash_totals = 0
    ratio_accum = 0.0
    in_tokens = 0
    out_tokens = 0
    success = 0
    recovery = 0

    for rec in arm.trajectories:
        tid = rec.get("trajectory_id", "")
        label = arm.labels.get(tid, {})
        outcome = label.get("outcome", "unknown")
        outcome_counts[outcome] += 1
        if label.get("contains_recovery"):
            recovery += 1

        events = rec.get("tool_call_events", []) or []
        tool_totals += len(events)
        edits = _count_edits(events)
        bash = _count_tool(events, "bash")
        edit_totals += edits
        bash_totals += bash
        ratio_accum += edits / max(bash, 1)

        tu = rec.get("token_usage", {}) or {}
        in_tokens += int(tu.get("input_tokens", 0))
        out_tokens += int(tu.get("output_tokens", 0))

        if (rec.get("verification", {}) or {}).get("final_passed"):
            success += 1

    return AggregateArm(
        name=arm.name,
        n_runs=n,
        outcome_counts=dict(outcome_counts),
        avg_tool_calls=tool_totals / n,
        avg_edits=edit_totals / n,
        avg_bash=bash_totals / n,
        edit_bash_ratio=ratio_accum / n,
        avg_input_tokens=in_tokens / n,
        avg_output_tokens=out_tokens / n,
        success_rate=success / n,
        recovery_rate=recovery / n,
    )


# ---------------------------------------------------------------------------
# Plotting (side-effecting)
# ---------------------------------------------------------------------------


# Canonical outcome order — keeps colors stable across figures.
# Maps to Liu-et-al roughly: one_shot=A1 clean-win, recovery=A2 recovered,
# wrong_fix=B2.1, premature=C2, failed=B/C rest, unknown=data gap.
_OUTCOME_ORDER = [
    "verified_one_shot",
    "verified_with_recovery",
    "wrong_fix",
    "premature_success",
    "failed",
    "unknown",
]

_OUTCOME_COLORS = {
    "verified_one_shot":       "#4c9f70",  # muted green
    "verified_with_recovery":  "#2b7a4b",  # deeper green — high SFT value
    "wrong_fix":               "#c0392b",  # red
    "premature_success":       "#e67e22",  # orange
    "failed":                  "#95a5a6",  # grey
    "unknown":                 "#d5dbdb",  # light grey
}


def _render_outcome_stacked_bar(arms: list[AggregateArm], out_path: Path) -> None:
    """Stacked bar per arm, segments = outcome labels in canonical order."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(arms)), 4.5))
    names = [a.name for a in arms]
    # normalise to percent so arms of different N stay comparable
    bottoms = [0.0] * len(arms)
    for outcome in _OUTCOME_ORDER:
        heights = []
        for a in arms:
            total = sum(a.outcome_counts.values()) or 1
            heights.append(100.0 * a.outcome_counts.get(outcome, 0) / total)
        ax.bar(
            names, heights, bottom=bottoms,
            color=_OUTCOME_COLORS.get(outcome, "#bdc3c7"),
            label=outcome, edgecolor="white", linewidth=0.5,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("% of trajectories")
    ax.set_ylim(0, 100)
    ax.set_title("Outcome distribution by arm")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)
    # annotate N per arm for reviewer sanity
    for i, a in enumerate(arms):
        ax.text(i, 101, f"n={a.n_runs}", ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_tool_counts(arms: list[AggregateArm], out_path: Path) -> None:
    """Grouped bars: avg tool calls / edits / bash per arm."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = [a.name for a in arms]
    x = np.arange(len(arms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(arms)), 4.5))
    ax.bar(x - width, [a.avg_tool_calls for a in arms], width, label="avg tool calls", color="#34495e")
    ax.bar(x,         [a.avg_edits       for a in arms], width, label="avg file edits", color="#c0392b")
    ax.bar(x + width, [a.avg_bash        for a in arms], width, label="avg bash runs",  color="#2980b9")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("count per run")
    ax.set_title("Tool-call composition by arm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_edit_bash_ratio(arms: list[AggregateArm], out_path: Path) -> None:
    """Single bar per arm: mean edit/bash ratio.

    Lower = more disciplined (agent tests after editing). Ratio >= ~3 is the
    B2.1 Wrong-Fix zone empirically from sympy-20 pilots.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(arms)), 4.0))
    names = [a.name for a in arms]
    values = [a.edit_bash_ratio for a in arms]
    bars = ax.bar(names, values, color="#8e44ad", edgecolor="white", linewidth=0.5)
    ax.axhline(3.0, color="#c0392b", linestyle="--", linewidth=1.0, label="B2.1 danger zone (≥3)")
    ax.set_ylabel("edits / max(bash, 1)")
    ax.set_title("Edit-to-bash ratio by arm — lower = more testing")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_token_cost(arms: list[AggregateArm], out_path: Path) -> None:
    """Grouped bars: avg input + output tokens per arm."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = [a.name for a in arms]
    x = np.arange(len(arms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(arms)), 4.5))
    ax.bar(x - width / 2, [a.avg_input_tokens  for a in arms], width,
           label="avg input tokens",  color="#16a085")
    ax.bar(x + width / 2, [a.avg_output_tokens for a in arms], width,
           label="avg output tokens", color="#d35400")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("tokens per run")
    ax.set_title("Token cost by arm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_recovery_rate(arms: list[AggregateArm], out_path: Path) -> None:
    """Paired bars: success_rate vs recovery_rate per arm.

    The gap between success_rate and recovery_rate is roughly the
    one_shot fraction — a visual check on the outcome stacked bar.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = [a.name for a in arms]
    x = np.arange(len(arms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(arms)), 4.5))
    ax.bar(x - width / 2, [100 * a.success_rate  for a in arms], width,
           label="success rate (%)",  color="#27ae60")
    ax.bar(x + width / 2, [100 * a.recovery_rate for a in arms], width,
           label="recovery-bearing rate (%)", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% of runs")
    ax.set_ylim(0, 100)
    ax.set_title("Success vs recovery-bearing rate by arm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _parse_arm_spec(raw: str) -> tuple[str, Path]:
    """Parse ``name:path`` arm specifier.

    We intentionally don't accept bare paths — naming the arm is required
    because every figure legend/axis label reads from it. Forcing the
    name keeps ``python -m ...`` invocations self-documenting.
    """
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--arm expects 'name:path', got: {raw!r}"
        )
    name, _, path = raw.partition(":")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("arm name must be non-empty")
    return name, Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_ablation",
        description="Render 3-arm (or N-arm) ablation figures from trajectory JSONLs.",
    )
    p.add_argument(
        "--arm", action="append", required=True, type=_parse_arm_spec,
        metavar="NAME:PATH",
        help="arm specifier, e.g. baseline:data/sympy20/arm1.jsonl. "
             "Pass --arm once per arm; order determines plot order.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write PNG figures into (created if missing).",
    )
    p.add_argument(
        "--summary-json", type=Path, default=None,
        help="Optional: also write numeric aggregates as JSON for thesis tables.",
    )
    return p


def run(
    arms_spec: Iterable[tuple[str, Path]],
    out_dir: Path,
    summary_json: Path | None = None,
) -> list[AggregateArm]:
    """Top-level entry — pure-ish, returns the aggregates it just plotted.

    Returning the aggregates lets integration tests assert on scalars
    without having to decode the PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_data = [load_arm(name, path) for name, path in arms_spec]
    aggregates = [aggregate_arm(a) for a in arm_data]

    # Import matplotlib lazily so `run()` can be unit-tested without the
    # dep on minimal CI. Any backend-specific failure becomes a clean
    # import error here, not a surprise deep in the render.
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")  # headless-safe; does nothing if already set
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plot_ablation — pip install matplotlib"
        ) from e

    _render_outcome_stacked_bar(aggregates, out_dir / "outcome_distribution.png")
    _render_tool_counts(aggregates,          out_dir / "tool_counts.png")
    _render_edit_bash_ratio(aggregates,      out_dir / "edit_bash_ratio.png")
    _render_token_cost(aggregates,           out_dir / "token_cost.png")
    _render_recovery_rate(aggregates,        out_dir / "recovery_rate.png")

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                [
                    {
                        "name": a.name,
                        "n_runs": a.n_runs,
                        "outcome_counts": a.outcome_counts,
                        "avg_tool_calls": a.avg_tool_calls,
                        "avg_edits": a.avg_edits,
                        "avg_bash": a.avg_bash,
                        "edit_bash_ratio": a.edit_bash_ratio,
                        "avg_input_tokens": a.avg_input_tokens,
                        "avg_output_tokens": a.avg_output_tokens,
                        "success_rate": a.success_rate,
                        "recovery_rate": a.recovery_rate,
                    }
                    for a in aggregates
                ],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return aggregates


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        aggregates = run(args.arm, args.out_dir, summary_json=args.summary_json)
    except FileNotFoundError as e:
        print(f"[plot-failed] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[plot-failed] {e}", file=sys.stderr)
        return 1

    print(f"[plot-ok] wrote 5 figures to {args.out_dir}")
    for a in aggregates:
        print(
            f"  {a.name}: n={a.n_runs} success={a.success_rate:.1%} "
            f"recovery={a.recovery_rate:.1%} "
            f"tool={a.avg_tool_calls:.1f} edit/bash={a.edit_bash_ratio:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
