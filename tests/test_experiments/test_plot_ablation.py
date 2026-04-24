"""Tests for ``experiments.plot_ablation``.

The aggregation functions are pure; we cover them with fixture dicts so
the maths is locked before any matplotlib call. The ``run()`` entry
point is covered with a tmp_path integration test that writes actual
PNGs (guarded by an import skip when matplotlib isn't installed on the
test machine — Windows dev box doesn't have it, Linux CI does).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.plot_ablation import (
    AggregateArm,
    ArmData,
    _parse_arm_spec,
    aggregate_arm,
    build_parser,
    load_arm,
    run,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _traj(
    *,
    trajectory_id: str,
    instance_id: str = "swe-1",
    tool_names: list[str],
    success: bool = True,
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "instance_id": instance_id,
        "seed": 0,
        "tool_call_events": [
            {
                "tool_name": n,
                "turn_id": i,
                "call_index_in_turn": 0,
                "global_index": i,
                "params": {},
                "result_success": True,
                "result_content": "",
                "error_signal": "NONE",
                "timestamp": 0.0,
            }
            for i, n in enumerate(tool_names)
        ],
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
        },
        "verification": {
            "final_passed": success,
            "has_diff": success,
            "tier2_pass": success,
            "tier2_command": "",
        },
        "metadata": {},
    }


def _label(
    *,
    trajectory_id: str,
    outcome: str,
    contains_recovery: bool = False,
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "instance_id": "swe-1",
        "seed": 0,
        "outcome": outcome,
        "liu_categories": [],
        "is_toxic": False,
        "contains_recovery": contains_recovery,
        "highest_recovery_level": "L1" if contains_recovery else None,
        "recovery_event_count": 1 if contains_recovery else 0,
        "recovery_events": [],
        "error_signal_counts": {},
        "liu_detail": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# _parse_arm_spec
# ---------------------------------------------------------------------------


class TestParseArmSpec:
    def test_round_trip(self):
        name, path = _parse_arm_spec("baseline:data/a.jsonl")
        assert name == "baseline"
        assert path == Path("data/a.jsonl")

    def test_rejects_missing_colon(self):
        with pytest.raises(Exception):
            _parse_arm_spec("data/a.jsonl")

    def test_rejects_empty_name(self):
        with pytest.raises(Exception):
            _parse_arm_spec(":data/a.jsonl")


# ---------------------------------------------------------------------------
# aggregate_arm
# ---------------------------------------------------------------------------


class TestAggregateArm:
    def test_empty_arm_is_safe(self):
        """Zero trajectories → all-zeros aggregate, no crashes.

        This matters because the loader sometimes hands us a freshly
        created arm JSONL that's still empty on a rerun — we want a
        plot-able zero row, not a ZeroDivisionError.
        """
        arm = ArmData(name="empty", trajectories=[], labels={})
        agg = aggregate_arm(arm)
        assert agg.n_runs == 0
        assert agg.success_rate == 0.0
        assert agg.edit_bash_ratio == 0.0

    def test_all_edits_zero_bash_gives_full_edit_count_ratio(self):
        """B2.1 Wrong-Fix signature: 5 edits, 0 bash → ratio = 5.0.

        edit/bash with bash=0 is defined as edits / max(bash,1) = edits,
        so a run with 5 edits and no test yields a ratio of 5 — above
        the 3.0 danger line drawn on the figure.
        """
        trajs = [
            _traj(trajectory_id="t1", tool_names=["file_edit"] * 5, success=False)
        ]
        labels = {"t1": _label(trajectory_id="t1", outcome="wrong_fix")}
        agg = aggregate_arm(ArmData("solo", trajs, labels))
        assert agg.edit_bash_ratio == 5.0
        assert agg.avg_edits == 5.0
        assert agg.avg_bash == 0.0
        assert agg.success_rate == 0.0

    def test_outcome_counts_sum_to_n_runs(self):
        trajs = [
            _traj(trajectory_id="t1", tool_names=["file_read"]),
            _traj(trajectory_id="t2", tool_names=["file_edit"]),
            _traj(trajectory_id="t3", tool_names=["bash"]),
        ]
        labels = {
            "t1": _label(trajectory_id="t1", outcome="verified_one_shot"),
            "t2": _label(trajectory_id="t2", outcome="verified_with_recovery",
                         contains_recovery=True),
            "t3": _label(trajectory_id="t3", outcome="failed"),
        }
        agg = aggregate_arm(ArmData("three", trajs, labels))
        assert sum(agg.outcome_counts.values()) == agg.n_runs == 3
        assert agg.recovery_rate == pytest.approx(1 / 3)

    def test_missing_label_yields_unknown_outcome(self):
        """Trajectory with no matching label row → counted as 'unknown'.

        Must never silently fold into 'failed' — the unknown bucket is
        the honest signal that classifier + trajectory files drifted,
        which the plot caller can see and act on.
        """
        trajs = [_traj(trajectory_id="t1", tool_names=["bash"])]
        agg = aggregate_arm(ArmData("drift", trajs, {}))
        assert agg.outcome_counts == {"unknown": 1}

    def test_token_averaging(self):
        trajs = [
            _traj(trajectory_id="t1", tool_names=["bash"], input_tokens=100, output_tokens=50),
            _traj(trajectory_id="t2", tool_names=["bash"], input_tokens=300, output_tokens=150),
        ]
        labels = {
            "t1": _label(trajectory_id="t1", outcome="failed"),
            "t2": _label(trajectory_id="t2", outcome="failed"),
        }
        agg = aggregate_arm(ArmData("tok", trajs, labels))
        assert agg.avg_input_tokens == 200
        assert agg.avg_output_tokens == 100

    def test_file_create_counts_as_edit(self):
        """file_create is semantically an edit for B2.1 purposes (matches
        ``_maybe_test_nudge`` in pare.agent.loop)."""
        trajs = [_traj(trajectory_id="t1",
                       tool_names=["file_create", "file_create", "bash"])]
        labels = {"t1": _label(trajectory_id="t1", outcome="failed")}
        agg = aggregate_arm(ArmData("create", trajs, labels))
        assert agg.avg_edits == 2.0
        assert agg.avg_bash == 1.0


# ---------------------------------------------------------------------------
# load_arm
# ---------------------------------------------------------------------------


class TestLoadArm:
    def test_missing_trajectory_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_arm("x", tmp_path / "missing.jsonl")

    def test_missing_labels_gives_remediation_hint(self, tmp_path: Path):
        """Missing labels must suggest the classify_trajectories fix —
        ablation figures without labels are nonsense, so we want a loud
        error with the exact command to run, not a mystery KeyError
        300 lines deep in aggregate_arm."""
        tpath = tmp_path / "arm.jsonl"
        _write_jsonl(tpath, [_traj(trajectory_id="t1", tool_names=["bash"])])

        with pytest.raises(FileNotFoundError) as exc:
            load_arm("x", tpath)
        assert "labels jsonl missing" in str(exc.value)
        assert "classify_trajectories" in str(exc.value)

    def test_pairs_labels_by_trajectory_id(self, tmp_path: Path):
        tpath = tmp_path / "arm.jsonl"
        lpath = tpath.with_suffix(".labels.jsonl")
        _write_jsonl(tpath, [
            _traj(trajectory_id="t1", tool_names=["bash"]),
            _traj(trajectory_id="t2", tool_names=["file_edit"]),
        ])
        # Deliberate: labels written in reverse order → assert dict keys still join right.
        _write_jsonl(lpath, [
            _label(trajectory_id="t2", outcome="failed"),
            _label(trajectory_id="t1", outcome="verified_one_shot"),
        ])

        arm = load_arm("x", tpath)
        assert arm.labels["t1"]["outcome"] == "verified_one_shot"
        assert arm.labels["t2"]["outcome"] == "failed"


# ---------------------------------------------------------------------------
# CLI + parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_requires_at_least_one_arm(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--out-dir", "out"])

    def test_accepts_multiple_arms(self):
        parser = build_parser()
        args = parser.parse_args([
            "--arm", "a:x.jsonl",
            "--arm", "b:y.jsonl",
            "--out-dir", "out",
        ])
        assert len(args.arm) == 2
        assert args.arm[0] == ("a", Path("x.jsonl"))
        assert args.arm[1] == ("b", Path("y.jsonl"))


# ---------------------------------------------------------------------------
# End-to-end render (needs matplotlib — skip when absent)
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def test_renders_five_pngs_and_summary(self, tmp_path: Path):
        pytest.importorskip(
            "matplotlib", reason="matplotlib not installed — skipping render e2e"
        )
        """Full run on fixture 2-arm data writes all 5 PNGs + summary JSON."""
        arm1 = tmp_path / "arm1.jsonl"
        _write_jsonl(arm1, [
            _traj(trajectory_id="t1", tool_names=["file_edit"] * 4 + ["bash"],
                  success=True),
            _traj(trajectory_id="t2", tool_names=["file_edit"] * 5,
                  success=False),
        ])
        _write_jsonl(arm1.with_suffix(".labels.jsonl"), [
            _label(trajectory_id="t1", outcome="verified_with_recovery",
                   contains_recovery=True),
            _label(trajectory_id="t2", outcome="wrong_fix"),
        ])

        arm2 = tmp_path / "arm2.jsonl"
        _write_jsonl(arm2, [
            _traj(trajectory_id="t3", tool_names=["bash", "file_edit", "bash"],
                  success=True),
            _traj(trajectory_id="t4", tool_names=["bash"] * 3 + ["file_edit"],
                  success=True),
        ])
        _write_jsonl(arm2.with_suffix(".labels.jsonl"), [
            _label(trajectory_id="t3", outcome="verified_with_recovery",
                   contains_recovery=True),
            _label(trajectory_id="t4", outcome="verified_with_recovery",
                   contains_recovery=True),
        ])

        out_dir = tmp_path / "figs"
        summary = tmp_path / "summary.json"
        aggregates = run(
            [("baseline", arm1), ("nudge", arm2)],
            out_dir=out_dir,
            summary_json=summary,
        )

        # All 5 figures emitted.
        for fname in (
            "outcome_distribution.png",
            "tool_counts.png",
            "edit_bash_ratio.png",
            "token_cost.png",
            "recovery_rate.png",
        ):
            assert (out_dir / fname).exists(), f"missing: {fname}"
            assert (out_dir / fname).stat().st_size > 0, f"empty: {fname}"

        # Summary JSON round-trips.
        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert len(payload) == 2
        assert payload[0]["name"] == "baseline"
        assert payload[1]["name"] == "nudge"
        # Baseline's arm has more edits, fewer bash → higher ratio.
        assert payload[0]["edit_bash_ratio"] > payload[1]["edit_bash_ratio"]

        # Return value mirrors summary scalars.
        assert [a.name for a in aggregates] == ["baseline", "nudge"]
        assert aggregates[1].success_rate == 1.0
