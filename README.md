---
title: BugForge
emoji: ":bug:"
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# BugForge

BugForge is an OpenEnv environment for training and evaluating agents on a real software engineering task: debugging Python code. Each episode injects exactly one hidden bug into a small task-specific codebase. The agent must run tests, inspect source files, apply an exact patch, and terminate when the issue is fixed.

This environment models a real human workflow rather than a toy game. It focuses on interactive, evidence-driven bug fixing under a limited step budget.

## Why This Environment Matters

Many coding benchmarks only score a final answer. BugForge scores the full debugging trajectory:

- gathering evidence with tests
- selecting the right files to inspect
- making precise patches
- stopping efficiently once the fix is complete

That makes it useful for RL training, agent evaluation, and reward-design research for code-repair systems.

## OpenEnv Summary

- Benchmark: `bugforge`
- Runtime: FastAPI / OpenEnv server
- Reward range: `0.0` to `1.0`
- Environment step budget: `10`
- Baseline inference budget: `6`
- Hugging Face Space: `https://huggingface.co/spaces/pranayteja31/BugForge`
- Live API base: `https://pranayteja31-BugForge.hf.space`

## Action Space

`BugforgeAction` supports the following actions:

- `{"type": "run_tests"}`
- `{"type": "read_file", "file": "filename.py"}`
- `{"type": "apply_patch", "file": "filename.py", "old_code": "...", "new_code": "..."}`
- `{"type": "done"}`

Action semantics:

- `run_tests`: execute the task's `tests.py` and return the traceback or success marker
- `read_file`: return the full contents of a file in the working directory
- `apply_patch`: replace exact `old_code` with `new_code` in the target file
- `done`: end the episode when the agent believes the bug is fixed

## Observation Space

`BugforgeObservation` contains:

- `output`: raw output from the last action
- `tests_passing`: number of passing tests
- `tests_total`: total number of tests
- `files_read`: unique files inspected so far
- `steps_remaining`: remaining environment step budget
- `patches_applied`: number of patch attempts

The OpenEnv step result also carries:

- `reward`: scalar reward for the last action
- `done`: termination flag

## Tasks

BugForge includes three deterministic tasks with increasing difficulty.

### Task 1 - Easy

- Bug: wrong arithmetic divisor in `utils.py`
- Goal: fix the discount calculation
- Skill tested: reading a traceback and repairing a simple single-file bug

### Task 2 - Medium

- Bug: cross-file type mismatch in `models.py`
- Goal: return an integer quantity instead of a string
- Skill tested: following a failing symptom across modules and patching the correct file

### Task 3 - Hard

- Bug: missing fallback branch in `cart.py`
- Goal: restore correct behavior for unsupported coupon types
- Skill tested: identifying and patching a missing control-flow branch

## Reward Design

The reward function provides useful signal across the trajectory instead of only at the end.

- `run_tests`: `0.10` on the first call, `0.05` on later calls
- `read_file`: `0.15` when the true bug file is read for the first time, `0.05` for other first-time reads
- `apply_patch`:
  - `1.00` if all tests pass after the patch
  - `0.30` if the patch increases the number of passing tests
  - `0.00` if the patch has no effect or makes things worse
- `done`:
  - `1.00` if all tests pass
  - `0.00` otherwise

This encourages efficient debugging behavior:

- gather evidence early
- inspect relevant files quickly
- patch precisely
- stop as soon as the task is solved

## Grading

The final score is produced by `BugforgeEnvironment.grade()`:

- solved episodes score in `[0.8, 1.0]` based on efficiency
- partially solved episodes score up to `0.6` based on passed tests
- agents that identify the bug file but fail to fix it can still earn `0.2`

All tasks are deterministic because each episode starts from a clean template and injects a fixed bug from a task config file.

## Baseline Inference

The baseline script is `inference.py` in the project root. It:

- uses the OpenAI Python client
- reads `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` from environment variables
- connects to the deployed Space
- emits the required `[START]`, `[STEP]`, and `[END]` stdout format
- completes all three tasks well under the 20-minute limit

Latest verified baseline on the live Space:

- Task 1: `success=true`, `steps=4`, `score=0.867`
- Task 2: `success=true`, `steps=4`, `score=0.867`
- Task 3: `success=true`, `steps=4`, `score=0.867`

## Setup

Install dependencies:

```bash
pip install -e .
```

Set the inference environment variables:

```powershell
$env:HF_TOKEN = "<your_api_key>"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_SPACE_URL = "https://pranayteja31-BugForge.hf.space"
```

Any OpenAI-compatible provider can be used by changing `API_BASE_URL` and `MODEL_NAME`.

## Running Locally

Start the server:

```bash
python -m bugforge.server.app
```

Test the reset endpoint:

```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{\"task_id\": 1}"
```

Run the baseline:

```bash
python inference.py
```

## Docker

Build the image:

```bash
docker build -t bugforge-env .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 bugforge-env
```

## Validation

Run the pre-submission validator:

```bash
./validate.sh https://pranayteja31-BugForge.hf.space .
```

Expected checks:

- the Space responds to `POST /reset`
- Docker build succeeds
- `openenv validate` passes

## Project Structure

```text
bugforge/
  __init__.py
  client.py
  models.py
  inference.py
  openenv.yaml
  validate.sh
  bugs/
    task_1/
    task_2/
    task_3/
  server/
    app.py
    bugforge_environment.py
```
