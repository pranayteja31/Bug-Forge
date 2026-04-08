import subprocess
import sys
import os
import shutil
import json
import uuid
import re
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from bugforge.models import BugforgeAction, BugforgeObservation


class BugforgeEnvironment(Environment[BugforgeAction, BugforgeObservation, State]):
    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.working_dir = None
        self.config = None
        self.task_id = None
        self.steps_taken = 0
        self.max_steps = 10
        self.files_read = []
        self.patches_applied = 0
        self.total_reward = 0.0
        self._done = False
        self._tests_total = 0
        self._test_order = []

    @staticmethod
    def _strict_score(value: float) -> float:
        """Hackathon requires strict (0, 1), excluding exact 0.0 and 1.0."""
        return round(min(max(value, 0.001), 0.999), 3)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BugforgeObservation:
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self.steps_taken = 0
        self.files_read = []
        self.patches_applied = 0
        self.total_reward = 0.0
        self._done = False
        self.task_id = kwargs.get("task_id", 1)

        self.working_dir, self.config = self._inject_bug(self.task_id)
        self._tests_total, self._test_order = self._discover_tests_metadata()
        test_output = self._run_tests()
        passing, total = self._parse_tests(test_output)

        return BugforgeObservation(
            output=f"Environment ready. Task: {self.config['description']}. Run tests to begin.",
            tests_passing=passing,
            tests_total=total,
            files_read=[],
            steps_remaining=self.max_steps,
            patches_applied=0,
        )

    def step(
        self,
        action: BugforgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BugforgeObservation:
        # Guard: if reset() was never called, return an error observation
        if self.working_dir is None:
            return BugforgeObservation(
                output="Error: call reset() before step().",
                done=True,
            )

        if self._done:
            passing, total = self._parse_tests(self._run_tests())
            return BugforgeObservation(
                output="Episode already done.",
                tests_passing=passing,
                tests_total=total,
                files_read=self.files_read,
                steps_remaining=0,
                patches_applied=self.patches_applied,
                done=True,
            )

        self.steps_taken += 1
        self._state.step_count += 1
        reward = 0.0
        output = ""

        if action.type == "run_tests":
            output = self._run_tests()
            reward = 0.1 if self.steps_taken == 1 else 0.05

        elif action.type == "read_file":
            output = self._read_file(action.file)
            if action.file not in self.files_read:
                self.files_read.append(action.file)
                reward = 0.15 if action.file == self.config["bug_file"] else 0.05

        elif action.type == "apply_patch":
            output, reward = self._apply_patch(action.file, action.old_code, action.new_code)
            self.patches_applied += 1

        elif action.type == "done":
            output, reward = self._finish()
            self._done = True

        if self.steps_taken >= self.max_steps:
            self._done = True

        self.total_reward += reward
        passing, total = self._parse_tests(self._run_tests())

        return BugforgeObservation(
            output=output,
            tests_passing=passing,
            tests_total=total,
            files_read=self.files_read,
            steps_remaining=self.max_steps - self.steps_taken,
            patches_applied=self.patches_applied,
            done=self._done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    def grade(self) -> float:
        test_output = self._run_tests()
        passing, total = self._parse_tests(test_output)
        if passing == total and total > 0:
            efficiency = (self.max_steps - self.steps_taken) / self.max_steps
            return self._strict_score(0.8 + (0.2 * efficiency))
        elif passing > 0:
            return self._strict_score((passing / total) * 0.6)
        elif self.config and self.config["bug_file"] in self.files_read:
            return self._strict_score(0.2)
        return self._strict_score(0.0)

    def _inject_bug(self, task_id):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task_dir = os.path.join(base, "bugs", f"task_{task_id}")
        config_path = os.path.join(task_dir, "bug_config.json")

        with open(config_path) as f:
            config = json.load(f)

        working_dir = os.path.join(task_dir, "working")
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        shutil.copytree(os.path.join(task_dir, "clean"), working_dir)

        bug_file = os.path.join(working_dir, config["bug_file"])
        with open(bug_file, "r") as f:
            content = f.read()
        content = content.replace(config["original_code"], config["buggy_code"])
        with open(bug_file, "w") as f:
            f.write(content)

        return working_dir, config

    def _run_tests(self) -> str:
        if not self.working_dir:
            return "No working directory set. Call reset() first."
        try:
            result = subprocess.run(
                [sys.executable, "tests.py"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error running tests: {str(e)}"

    def _read_file(self, filename: str) -> str:
        try:
            with open(os.path.join(self.working_dir, filename)) as f:
                return f.read()
        except Exception:
            return f"File {filename} not found"

    def _apply_patch(self, filename, old_code, new_code):
        try:
            filepath = os.path.join(self.working_dir, filename)
            with open(filepath) as f:
                content = f.read()
            if old_code not in content:
                return "Patch failed: code not found", 0.0
            passing_before, _ = self._parse_tests(self._run_tests())
            new_content = content.replace(old_code, new_code)
            with open(filepath, "w") as f:
                f.write(new_content)
            after = self._run_tests()
            passing_after, total = self._parse_tests(after)
            if passing_after == total:
                return "Patch applied. ALL TESTS PASS!", 1.0
            elif passing_after > passing_before:
                return f"Patch applied. {passing_after}/{total} tests passing.", 0.3
            elif passing_after < passing_before:
                return "Patch applied but broke passing tests!", 0.0
            return "Patch applied. No change in tests.", 0.0
        except Exception as e:
            return f"Patch error: {str(e)}", 0.0

    def _finish(self):
        test_output = self._run_tests()
        passing, total = self._parse_tests(test_output)
        if passing == total:
            return "Done! All tests pass!", 1.0
        return f"Done. Only {passing}/{total} tests passing.", 0.0

    def _discover_tests_metadata(self):
        if not self.working_dir:
            return 0, []

        tests_path = os.path.join(self.working_dir, "tests.py")
        try:
            with open(tests_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return 0, []

        # Primary signal: explicit invocation order in __main__ block.
        ordered_calls = re.findall(r"^\s*(test_[A-Za-z0-9_]+)\(\)\s*$", content, flags=re.MULTILINE)
        if ordered_calls:
            return len(ordered_calls), ordered_calls

        # Fallback: discovered test functions.
        discovered_defs = re.findall(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", content, flags=re.MULTILINE)
        return len(discovered_defs), discovered_defs

    def _parse_tests(self, output: str):
        if self._tests_total <= 0:
            self._tests_total, self._test_order = self._discover_tests_metadata()

        total = self._tests_total
        if total <= 0:
            # Conservative fallback if test discovery fails.
            return (0, 0) if "ALL_TESTS_PASSED" not in output else (0, 0)

        if "ALL_TESTS_PASSED" in output:
            return total, total

        # Robustly infer which test failed from traceback.
        failed_test_names = re.findall(r"\bin\s+(test_[A-Za-z0-9_]+)\b", output)
        failed_test = None
        for name in failed_test_names:
            if name in self._test_order:
                failed_test = name
                break

        if failed_test:
            return self._test_order.index(failed_test), total

        # Any traceback/exception without a known test anchor => treat as 0 passing.
        if "Traceback" in output or "Error" in output or "Exception" in output:
            return 0, total

        # No explicit pass marker and no clear traceback; stay conservative.
        return 0, total
