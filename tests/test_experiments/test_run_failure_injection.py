"""Tests for ``experiments.run_failure_injection``.

Locks the orchestration contract before we call it from a real
agent runner. The invariants we cover:

1. Output JSONL has exactly ``len(faults) * len(tasks) * len(seeds)``
   well-formed rows, each with ``seed`` carried through.
2. Per-fault aggregates in the report match what the JSONL says.
3. ``--list-faults`` prints REGISTRY contents and exits 0.
4. Required-flag validation kicks in before touching disk.
5. A revert-failing fault surfaces in ``rows_with_revert_failure``.
6. An exception-raising agent_runner counts as ``n_agent_raised``,
   not ``n_agent_failed``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.run_failure_injection import (
    FaultInjectionRunReport,
    build_parser,
    dry_run_agent_runner,
    main,
    parse_fault_list,
    run_fault_injection_batch,
)
from experiments.generate_trajectories import GenerationTask
from pare.eval.failure_injection import (
    InjectedFault,
    REGISTRY,
    _register,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_workdir_with_python_file(root: Path, instance_id: str) -> Path:
    """Create a minimal repo so ``wrong_import`` has something to target."""
    wd = root / instance_id
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "pkg").mkdir(exist_ok=True)
    (wd / "pkg" / "core.py").write_text(
        '"""module."""\n', encoding="utf-8"
    )
    return wd


def _write_tasks_jsonl(path: Path, instance_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for iid in instance_ids:
            f.write(
                json.dumps({"instance_id": iid, "task": f"fix bug in {iid}"})
                + "\n"
            )


# ---------------------------------------------------------------------------
# parse_fault_list
# ---------------------------------------------------------------------------


class TestParseFaultList:
    def test_all_returns_sorted_registry(self):
        names = parse_fault_list("all")
        assert names == sorted(REGISTRY)

    def test_explicit_csv_preserves_order(self):
        names = parse_fault_list("wrong_import,empty_baseline")
        assert names == ["wrong_import", "empty_baseline"]

    def test_dedups(self):
        names = parse_fault_list("wrong_import,wrong_import")
        assert names == ["wrong_import"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            parse_fault_list(",,")


# ---------------------------------------------------------------------------
# run_fault_injection_batch — happy path
# ---------------------------------------------------------------------------


class TestBatchHappyPath:
    def test_iterates_faults_x_tasks_x_seeds(self, tmp_path: Path):
        """Output row count = |faults| * |tasks| * |seeds| with the
        fault_name + seed carried through. Each row is one
        FaultInjectionResult.to_dict() + a 'seed' field."""
        workdir_root = tmp_path / "workdirs"
        for iid in ("swe-1", "swe-2"):
            _make_workdir_with_python_file(workdir_root, iid)

        tasks = [
            GenerationTask(instance_id="swe-1", task="t1"),
            GenerationTask(instance_id="swe-2", task="t2"),
        ]
        out_jsonl = tmp_path / "out.jsonl"

        report = run_fault_injection_batch(
            tasks,
            fault_names=["wrong_import", "empty_baseline"],
            output_jsonl=out_jsonl,
            seeds=[0, 1],
            agent_runner=dry_run_agent_runner,
            workdir_for=lambda iid: workdir_root / iid,
        )

        # 2 faults * 2 tasks * 2 seeds = 8
        assert report.rows_written == 8
        rows = [
            json.loads(line)
            for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 8

        # Every row carries the 4 essential fields.
        for row in rows:
            assert row["fault_name"] in {"wrong_import", "empty_baseline"}
            assert row["instance_id"] in {"swe-1", "swe-2"}
            assert row["seed"] in {0, 1}
            assert "agent_exit_code" in row
            assert "applied_at" in row

    def test_per_fault_counts_match_rows(self, tmp_path: Path):
        workdir_root = tmp_path / "workdirs"
        _make_workdir_with_python_file(workdir_root, "swe-1")

        tasks = [GenerationTask(instance_id="swe-1", task="t1")]
        report = run_fault_injection_batch(
            tasks,
            fault_names=["wrong_import", "empty_baseline"],
            output_jsonl=tmp_path / "out.jsonl",
            seeds=[0, 1, 2],
            agent_runner=dry_run_agent_runner,
            workdir_for=lambda iid: workdir_root / iid,
        )
        # 1 task * 3 seeds for each fault.
        for name in ("wrong_import", "empty_baseline"):
            assert report.per_fault[name]["n_runs"] == 3
            assert report.per_fault[name]["n_agent_ok"] == 3
            assert report.per_fault[name]["n_agent_failed"] == 0
            assert report.per_fault[name]["n_agent_raised"] == 0
            assert report.per_fault[name]["n_revert_failed"] == 0

    def test_max_instances_caps_tasks(self, tmp_path: Path):
        workdir_root = tmp_path / "workdirs"
        for iid in ("swe-1", "swe-2", "swe-3"):
            _make_workdir_with_python_file(workdir_root, iid)
        tasks = [
            GenerationTask(instance_id=f"swe-{i}", task=f"t{i}")
            for i in (1, 2, 3)
        ]

        report = run_fault_injection_batch(
            tasks,
            fault_names=["empty_baseline"],
            output_jsonl=tmp_path / "out.jsonl",
            seeds=[0],
            agent_runner=dry_run_agent_runner,
            workdir_for=lambda iid: workdir_root / iid,
            max_instances=2,
        )
        assert report.tasks_run == 2
        assert report.rows_written == 2

    def test_unknown_fault_raises_keyerror(self, tmp_path: Path):
        with pytest.raises(KeyError, match="unknown fault"):
            run_fault_injection_batch(
                tasks=[GenerationTask(instance_id="x", task="t")],
                fault_names=["does_not_exist"],
                output_jsonl=tmp_path / "out.jsonl",
                seeds=[0],
                agent_runner=dry_run_agent_runner,
                workdir_for=lambda iid: tmp_path,
            )


# ---------------------------------------------------------------------------
# Error-path counters
# ---------------------------------------------------------------------------


class TestErrorPathCounters:
    """The per-fault counters distinguish 'agent returned non-zero' from
    'agent raised' and surface 'revert failed' separately. These
    distinctions are what makes the JSONL useful for triage later."""

    def test_agent_raised_counts_into_n_agent_raised(self, tmp_path: Path):
        def _exploding_runner(*_args, **_kwargs):
            raise RuntimeError("provider dropped")

        workdir_root = tmp_path / "workdirs"
        _make_workdir_with_python_file(workdir_root, "swe-1")

        report = run_fault_injection_batch(
            tasks=[GenerationTask(instance_id="swe-1", task="t")],
            fault_names=["empty_baseline"],
            output_jsonl=tmp_path / "out.jsonl",
            seeds=[0],
            agent_runner=_exploding_runner,
            workdir_for=lambda iid: workdir_root / iid,
        )
        bucket = report.per_fault["empty_baseline"]
        assert bucket["n_agent_raised"] == 1
        assert bucket["n_agent_failed"] == 0
        assert bucket["n_agent_ok"] == 0
        assert report.rows_with_agent_failure == 1

    def test_agent_nonzero_exit_counts_into_n_agent_failed(self, tmp_path: Path):
        def _failed_runner(*_args, **_kwargs):
            return 1, {"trajectory_id": "t_failed"}

        workdir_root = tmp_path / "workdirs"
        _make_workdir_with_python_file(workdir_root, "swe-1")

        report = run_fault_injection_batch(
            tasks=[GenerationTask(instance_id="swe-1", task="t")],
            fault_names=["empty_baseline"],
            output_jsonl=tmp_path / "out.jsonl",
            seeds=[0],
            agent_runner=_failed_runner,
            workdir_for=lambda iid: workdir_root / iid,
        )
        bucket = report.per_fault["empty_baseline"]
        assert bucket["n_agent_failed"] == 1
        assert bucket["n_agent_raised"] == 0
        assert bucket["n_agent_ok"] == 0

    def test_revert_failure_counts_into_n_revert_failed(self, tmp_path: Path):
        """A fault whose revert raises must surface as
        ``n_revert_failed`` and bump ``rows_with_revert_failure``."""

        def _bad_revert(_workdir: Path, _token: Any) -> None:
            raise OSError("disk full")

        fault = InjectedFault(
            name="_test_revert_boom",
            description="test-only fault whose revert raises",
            applies_to_liu="",
            apply_fn=lambda _p: {},
            revert_fn=_bad_revert,
        )
        _register(fault)
        try:
            workdir_root = tmp_path / "workdirs"
            _make_workdir_with_python_file(workdir_root, "swe-1")

            report = run_fault_injection_batch(
                tasks=[GenerationTask(instance_id="swe-1", task="t")],
                fault_names=["_test_revert_boom"],
                output_jsonl=tmp_path / "out.jsonl",
                seeds=[0],
                agent_runner=dry_run_agent_runner,
                workdir_for=lambda iid: workdir_root / iid,
            )
            bucket = report.per_fault["_test_revert_boom"]
            assert bucket["n_revert_failed"] == 1
            # Agent itself succeeded — the failure is in the cleanup.
            assert bucket["n_agent_ok"] == 1
            assert report.rows_with_revert_failure == 1
        finally:
            REGISTRY.pop("_test_revert_boom", None)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCli:
    def test_list_faults_prints_registry_and_exits_zero(self, capsys):
        rc = main(["--list-faults"])
        assert rc == 0
        out = capsys.readouterr().out
        # Each registered fault appears in the output.
        for name in REGISTRY:
            assert name in out

    def test_missing_required_flags_fails_loud(self, capsys):
        """No --tasks-jsonl, no --output-jsonl, no --workdir-root →
        exit 1 with a [fault-injection-failed] prefix."""
        rc = main([])  # argparse defaults make all of these None
        assert rc == 1
        err = capsys.readouterr().err
        assert "[fault-injection-failed]" in err
        assert "--tasks-jsonl" in err
        assert "--output-jsonl" in err
        assert "--workdir-root" in err

    def test_main_round_trip_writes_jsonl_and_report(
        self, tmp_path: Path, capsys
    ):
        """``main()`` writes the JSONL + a sibling report.json + prints
        the summary table."""
        workdir_root = tmp_path / "workdirs"
        _make_workdir_with_python_file(workdir_root, "swe-1")

        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks_jsonl(tasks_path, ["swe-1"])
        out_path = tmp_path / "out.jsonl"

        rc = main([
            "--tasks-jsonl", str(tasks_path),
            "--output-jsonl", str(out_path),
            "--faults", "empty_baseline",
            "--seeds", "0",
            "--workdir-root", str(workdir_root),
        ])
        assert rc == 0

        # Output JSONL exists and has 1 row.
        rows = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert row["fault_name"] == "empty_baseline"
        assert row["instance_id"] == "swe-1"

        # Report sidecar.
        report_path = Path(str(out_path) + ".report.json")
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_written"] == 1
        assert "empty_baseline" in report["per_fault"]

        # Summary printed to stdout.
        out = capsys.readouterr().out
        assert "[fault-injection-ok]" in out
        assert "empty_baseline" in out

    def test_missing_workdir_for_an_instance_surfaces_as_cli_error(
        self, tmp_path: Path, capsys
    ):
        """If the workdir for any instance doesn't exist, fail loud
        rather than silently apply the fault to a freshly-created empty
        directory (which would mask the real materialize step)."""
        # workdir_root exists, but swe-1 sub-directory does not.
        workdir_root = tmp_path / "workdirs"
        workdir_root.mkdir()

        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks_jsonl(tasks_path, ["swe-1"])

        rc = main([
            "--tasks-jsonl", str(tasks_path),
            "--output-jsonl", str(tmp_path / "out.jsonl"),
            "--faults", "empty_baseline",
            "--workdir-root", str(workdir_root),
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "[fault-injection-failed]" in err
        assert "workdir does not exist" in err

    def test_parser_default_faults_is_all(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-jsonl", "x",
            "--output-jsonl", "y",
            "--workdir-root", "z",
        ])
        assert args.faults == "all"
        assert args.seeds == "0"
