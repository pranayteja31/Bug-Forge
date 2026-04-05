# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bugforge Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BugforgeAction, BugforgeObservation


class BugforgeEnv(
    EnvClient[BugforgeAction, BugforgeObservation, State]
):
    """
    Client for the Bugforge Environment.

    Example:
        >>> with BugforgeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.output)
        ...
        ...     result = client.step(BugforgeAction(type="run_tests"))
        ...     print(result.observation.output)
    """

    def _step_payload(self, action: BugforgeAction) -> Dict:
        """
        Convert BugforgeAction to JSON payload for step message.

        Args:
            action: BugforgeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "type": action.type,
            "file": action.file,
            "old_code": action.old_code,
            "new_code": action.new_code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BugforgeObservation]:
        """
        Parse server response into StepResult[BugforgeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with BugforgeObservation
        """
        obs_data = payload.get("observation", {})
        observation = BugforgeObservation(
            output=obs_data.get("output", ""),
            tests_passing=obs_data.get("tests_passing", 0),
            tests_total=obs_data.get("tests_total", 0),
            files_read=obs_data.get("files_read", []),
            steps_remaining=obs_data.get("steps_remaining", 10),
            patches_applied=obs_data.get("patches_applied", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
