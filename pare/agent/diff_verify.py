"""DiffVerify - Differential Verification for AI Patches.

DiffVerify evaluates AI-generated patches to ensure they aren't cheating
(e.g., deleting assertions, writing vacuous tests).

Mechanism (Bidirectional Test Validation):
1. Forward Pass: Old tests pass on the new code (regression check).
2. Backward Pass: New tests added in the patch must fail on the old base branch.
   If they pass on the base branch, the test is likely vacuous or the bug wasn't actually reproduced.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pare.sandbox.git_checkpoint import _GIT

logger = logging.getLogger(__name__)

@dataclass
class DiffVerifyResult:
    passed: bool
    reason: str
    forward_pass_output: str = ""
    backward_pass_output: str = ""


class DiffVerify:
    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir

    async def _run_cmd(self, cmd: list[str], cwd: Path) -> tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode()

    async def _run_git(self, *args: str, cwd: Path) -> str:
        code, out, err = await self._run_cmd([_GIT, *args], cwd)
        if code != 0:
            raise RuntimeError(f"Git command failed: {err}")
        return out.strip()

    async def evaluate_patch(
        self, 
        base_branch: str, 
        patch_branch: str, 
        test_command: str
    ) -> DiffVerifyResult:
        """
        Evaluate if a patch truly fixes a bug without reward hacking.
        
        Args:
            base_branch: The original branch with the bug.
            patch_branch: The branch containing the AI's proposed fix and tests.
            test_command: The command to run the tests (e.g., 'pytest path/to/tests').
        """
        logger.info(f"DiffVerify initiated: Validating {patch_branch} against {base_branch}")
        
        # We need a clean worktree to run these tests safely.
        # For simplicity in this implementation, we will checkout branches sequentially
        # in the main repo, but ideally we'd use GhostWorktreeManager.
        
        try:
            # 1. Forward Pass: Target branch with fix + tests (Must be GREEN)
            logger.info(f"Forward Pass: Testing patch branch '{patch_branch}' (Expect: GREEN)")
            await self._run_git("checkout", patch_branch, cwd=self.repo_dir)
            code_patch, out_patch, err_patch = await self._run_cmd(test_command.split(), self.repo_dir)
            
            patch_output = out_patch + "\n" + err_patch
            
            if code_patch != 0:
                return DiffVerifyResult(
                    passed=False, 
                    reason=f"Forward Pass Failed: Patch branch does not pass its own tests (Return code {code_patch}).",
                    forward_pass_output=patch_output
                )

            # 2. Backward Pass: Base branch with the NEW tests (Must be RED)
            # To do this correctly, we checkout the base branch, then checkout ONLY the test files from the patch branch.
            logger.info(f"Backward Pass: Testing base branch '{base_branch}' with new tests (Expect: RED)")
            await self._run_git("checkout", base_branch, cwd=self.repo_dir)
            
            # Find test files modified in the patch
            diff_files = await self._run_git("diff", "--name-only", f"{base_branch}...{patch_branch}", cwd=self.repo_dir)
            test_files = [f for f in diff_files.splitlines() if f.startswith("test_") or f.endswith("_test.py")]
            
            if not test_files:
                return DiffVerifyResult(
                    passed=False,
                    reason="Reward Hacking Detected: The patch does not include any tests to verify the fix.",
                    forward_pass_output=patch_output
                )

            # Bring the new tests into the base branch
            for t_file in test_files:
                await self._run_git("checkout", patch_branch, "--", t_file, cwd=self.repo_dir)

            # Run the test command
            code_base, out_base, err_base = await self._run_cmd(test_command.split(), self.repo_dir)
            base_output = out_base + "\n" + err_base

            # Clean up the imported test files
            await self._run_git("reset", "--hard", cwd=self.repo_dir)

            if code_base == 0:
                return DiffVerifyResult(
                    passed=False,
                    reason="Reward Hacking Detected: Backward Pass Failed. The new tests passed on the base branch, meaning they are vacuous and don't actually catch the bug.",
                    forward_pass_output=patch_output,
                    backward_pass_output=base_output
                )

            logger.info("DiffVerify Validation Passed: The patch is mathematically sound.")
            return DiffVerifyResult(
                passed=True,
                reason="Patch passed Bidirectional Validation (Green on patch, Red on base).",
                forward_pass_output=patch_output,
                backward_pass_output=base_output
            )

        except Exception as e:
            logger.error(f"DiffVerify execution error: {e}")
            return DiffVerifyResult(passed=False, reason=f"DiffVerify execution error: {str(e)}")
        finally:
            # Ensure we return to base branch
            try:
                await self._run_git("checkout", base_branch, cwd=self.repo_dir)
            except Exception:
                pass
