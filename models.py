from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List


class BugforgeAction(Action):
    type: str = Field(..., description="Action type: run_tests, read_file, apply_patch, done")
    file: str = Field(default="", description="File to read or patch")
    old_code: str = Field(default="", description="Code to replace")
    new_code: str = Field(default="", description="Replacement code")


class BugforgeObservation(Observation):
    output: str = Field(default="", description="Result of the action")
    tests_passing: int = Field(default=0, description="Number of passing tests")
    tests_total: int = Field(default=0, description="Total number of tests")
    files_read: List[str] = Field(default_factory=list, description="Files read so far")
    steps_remaining: int = Field(default=10, description="Steps remaining")
    patches_applied: int = Field(default=0, description="Patches applied so far")