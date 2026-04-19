"""Tests for pare/agent/guardrails.py."""

import pytest

from pare.agent.guardrails import GuardrailConfig, Guardrails


class TestBudgetGuard:
    def test_allows_within_budget(self):
        guard = Guardrails(GuardrailConfig(max_tool_calls=5))
        assert guard.check("bash", {"command": "ls"}) is None

    def test_blocks_when_total_exhausted(self):
        guard = Guardrails(GuardrailConfig(max_tool_calls=2))
        guard.state.total_tool_calls = 2
        msg = guard.check("bash", {"command": "ls"})
        assert msg is not None
        assert "budget exhausted" in msg.lower()

    def test_blocks_when_step_exhausted(self):
        guard = Guardrails(GuardrailConfig(max_tool_calls_per_step=3))
        guard.state.step_tool_calls = 3
        msg = guard.check("bash", {"command": "ls"})
        assert msg is not None
        assert "step" in msg.lower()

    def test_budget_remaining(self):
        guard = Guardrails(GuardrailConfig(max_tool_calls=10))
        guard.state.total_tool_calls = 7
        assert guard.budget_remaining == 3

    def test_step_budget_remaining(self):
        guard = Guardrails(GuardrailConfig(max_tool_calls_per_step=15))
        guard.state.step_tool_calls = 10
        assert guard.step_budget_remaining == 5


class TestConsecutiveErrorGuard:
    def test_allows_under_limit(self):
        guard = Guardrails(GuardrailConfig(max_consecutive_errors=3))
        guard.state.consecutive_errors = 2
        assert guard.check("bash", {"command": "ls"}) is None

    def test_blocks_at_limit(self):
        guard = Guardrails(GuardrailConfig(max_consecutive_errors=3))
        guard.state.consecutive_errors = 3
        msg = guard.check("bash", {"command": "ls"})
        assert msg is not None
        assert "failed" in msg.lower()

    def test_resets_on_success(self):
        guard = Guardrails(GuardrailConfig(max_consecutive_errors=3))
        guard.record_result("bash", {}, success=False)
        guard.record_result("bash", {}, success=False)
        assert guard.state.consecutive_errors == 2
        guard.record_result("bash", {}, success=True)
        assert guard.state.consecutive_errors == 0


class TestLoopDetection:
    def test_allows_first_occurrence(self):
        guard = Guardrails(GuardrailConfig(max_repeated_actions=2))
        assert guard.check("bash", {"command": "ls"}) is None

    def test_allows_second_occurrence(self):
        guard = Guardrails(GuardrailConfig(max_repeated_actions=2))
        guard.record_call("bash", {"command": "ls"})
        assert guard.check("bash", {"command": "ls"}) is None

    def test_blocks_third_occurrence(self):
        guard = Guardrails(GuardrailConfig(max_repeated_actions=2))
        guard.record_call("bash", {"command": "ls"})
        guard.record_call("bash", {"command": "ls"})
        msg = guard.check("bash", {"command": "ls"})
        assert msg is not None
        assert "repeating" in msg.lower()

    def test_different_params_not_detected(self):
        guard = Guardrails(GuardrailConfig(max_repeated_actions=2))
        guard.record_call("bash", {"command": "ls"})
        guard.record_call("bash", {"command": "ls"})
        # Different params = different action
        assert guard.check("bash", {"command": "pwd"}) is None


class TestReadBeforeWrite:
    def test_blocks_edit_without_read(self):
        guard = Guardrails()
        msg = guard.check("file_edit", {"file_path": "main.py", "old_str": "a", "new_str": "b"})
        assert msg is not None
        assert "read it first" in msg.lower()

    def test_allows_edit_after_read(self):
        guard = Guardrails()
        guard.record_result("file_read", {"file_path": "main.py"}, success=True)
        msg = guard.check("file_edit", {"file_path": "main.py", "old_str": "a", "new_str": "b"})
        assert msg is None

    def test_failed_read_not_counted(self):
        guard = Guardrails()
        guard.record_result("file_read", {"file_path": "main.py"}, success=False)
        msg = guard.check("file_edit", {"file_path": "main.py", "old_str": "a", "new_str": "b"})
        assert msg is not None

    def test_different_file_blocked(self):
        guard = Guardrails()
        guard.record_result("file_read", {"file_path": "main.py"}, success=True)
        msg = guard.check("file_edit", {"file_path": "other.py", "old_str": "a", "new_str": "b"})
        assert msg is not None

    def test_file_create_not_blocked(self):
        """file_create doesn't require a prior read (it's a new file)."""
        guard = Guardrails()
        msg = guard.check("file_create", {"file_path": "new.py", "content": "x"})
        assert msg is None

    def test_read_not_blocked(self):
        guard = Guardrails()
        msg = guard.check("file_read", {"file_path": "any.py"})
        assert msg is None

    def test_bash_not_blocked(self):
        guard = Guardrails()
        msg = guard.check("bash", {"command": "ls"})
        assert msg is None


class TestFileChangeLimit:
    def test_allows_within_limit(self):
        guard = Guardrails(GuardrailConfig(max_file_changes_per_step=3))
        guard.state.read_files.add("a.py")  # read first
        guard.state.read_files.add("b.py")
        guard.record_result("file_edit", {"file_path": "a.py"}, success=True)
        guard.record_result("file_edit", {"file_path": "b.py"}, success=True)
        guard.state.read_files.add("c.py")
        msg = guard.check("file_edit", {"file_path": "c.py", "old_str": "x", "new_str": "y"})
        assert msg is None

    def test_blocks_over_limit(self):
        guard = Guardrails(GuardrailConfig(max_file_changes_per_step=2))
        guard.state.read_files.update(["a.py", "b.py", "c.py"])
        guard.record_result("file_edit", {"file_path": "a.py"}, success=True)
        guard.record_result("file_edit", {"file_path": "b.py"}, success=True)
        msg = guard.check("file_edit", {"file_path": "c.py", "old_str": "x", "new_str": "y"})
        assert msg is not None
        assert "too many files" in msg.lower()


class TestResetStep:
    def test_resets_per_step_state(self):
        guard = Guardrails()
        guard.state.step_tool_calls = 10
        guard.state.read_files.add("a.py")
        guard.state.edited_files.add("b.py")
        guard.state.action_hashes["abc"] = 5
        guard.state.total_tool_calls = 50  # Should NOT be reset

        guard.reset_step()

        assert guard.state.step_tool_calls == 0
        assert len(guard.state.read_files) == 1  # Preserved across steps
        assert len(guard.state.edited_files) == 0
        assert len(guard.state.action_hashes) == 0
        assert guard.state.total_tool_calls == 50  # Preserved

    def test_reset_all_clears_everything(self):
        guard = Guardrails()
        guard.state.step_tool_calls = 10
        guard.state.total_tool_calls = 50
        guard.state.read_files.add("a.py")
        guard.state.edited_files.add("b.py")
        guard.state.action_hashes["abc"] = 5
        guard.state.consecutive_errors = 3

        guard.state.reset_all()

        assert guard.state.step_tool_calls == 0
        assert guard.state.total_tool_calls == 0
        assert guard.state.consecutive_errors == 0
        assert len(guard.state.read_files) == 0
        assert len(guard.state.edited_files) == 0
        assert len(guard.state.action_hashes) == 0


class TestAdvisory:
    """Soft-nudge advisories — do NOT block, but return a message to inject."""

    def _simulate_reads(self, guard: Guardrails, n: int) -> None:
        """Simulate ``n`` successful read-only calls (no edits)."""
        for i in range(n):
            guard.record_call("bash", {"command": f"ls -{i}"})
            guard.record_result("bash", {"command": f"ls -{i}"}, success=True)

    def test_no_advisory_under_threshold(self):
        guard = Guardrails(GuardrailConfig(nudge_no_edit_after_n_calls=8))
        self._simulate_reads(guard, 5)
        assert guard.advisory() is None

    def test_no_edit_advisory_fires_after_threshold(self):
        guard = Guardrails(GuardrailConfig(nudge_no_edit_after_n_calls=8))
        self._simulate_reads(guard, 8)
        msg = guard.advisory()
        assert msg is not None
        assert "file_edit" in msg

    def test_no_edit_advisory_suppressed_after_first_edit(self):
        guard = Guardrails(GuardrailConfig(nudge_no_edit_after_n_calls=8))
        self._simulate_reads(guard, 7)
        # One real edit succeeds → the advisory must NOT fire even though
        # total_tool_calls keeps growing past the threshold.
        guard.record_call("file_edit", {"file_path": "x.py"})
        guard.state.read_files.add("x.py")  # satisfy read-before-write bookkeeping
        guard.record_result("file_edit", {"file_path": "x.py"}, success=True)
        self._simulate_reads(guard, 5)
        assert guard.advisory() is None

    def test_no_edit_advisory_cooldown(self):
        guard = Guardrails(
            GuardrailConfig(nudge_no_edit_after_n_calls=5, advisory_cooldown_calls=4)
        )
        self._simulate_reads(guard, 5)
        first = guard.advisory()
        assert first is not None
        # Immediately polling again must NOT re-fire — cooldown active.
        assert guard.advisory() is None
        # After 4 more calls, cooldown elapses → fires again.
        self._simulate_reads(guard, 4)
        second = guard.advisory()
        assert second is not None

    def test_advisory_is_non_blocking(self):
        """Advisory must NEVER return via check() — only via advisory()."""
        guard = Guardrails(GuardrailConfig(nudge_no_edit_after_n_calls=1))
        self._simulate_reads(guard, 3)
        # check() should not surface the advisory.
        assert guard.check("bash", {"command": "echo hi"}) is None
        # But advisory() should.
        assert guard.advisory() is not None

    def test_reset_all_clears_advisory_state(self):
        guard = Guardrails(GuardrailConfig(nudge_no_edit_after_n_calls=2))
        self._simulate_reads(guard, 2)
        assert guard.advisory() is not None  # fires and records
        guard.state.reset_all()
        # After reset, counters are zero → no advisory yet.
        assert guard.advisory() is None
        assert guard.state.total_edits == 0
        assert guard.state.last_advisory_at == {}
