---
title: BugForge
emoji: ⚾
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

BugForge is an OpenEnv environment for interactive Python debugging. Each episode injects one hidden bug into a small codebase, and the agent must inspect failures, read files, patch the code, and stop when the task is fixed.

The project originally started from a generic starter template; the authoritative BugForge-specific description is in the sections above, which supersede any stale starter notes that may appear later in this file.

## Overview

- Benchmark: `bugforge`
- Runtime: FastAPI / OpenEnv server
- Reward range: `0.0` to `1.0`
- Environment step budget: `10`
- Baseline inference budget: `6`
- Hugging Face Space: `https://huggingface.co/spaces/pranayteja31/BugForge`
- Live API base: `https://pranayteja31-BugForge.hf.space`

## Action Space

`BugforgeAction` supports:

- `{"type": "run_tests"}`
- `{"type": "read_file", "file": "filename.py"}`
- `{"type": "apply_patch", "file": "filename.py", "old_code": "...", "new_code": "..."}`
- `{"type": "done"}`

## Observation Space

`BugforgeObservation` contains:

- `output`
- `tests_passing`
- `tests_total`
- `files_read`
- `steps_remaining`
- `patches_applied`

## Tasks

- Task 1: wrong divisor in `utils.py`
- Task 2: cross-file type mismatch in `models.py`
- Task 3: missing fallback branch in `cart.py`

## Reward Design

- `run_tests`: `0.10` on the first call, `0.05` later
- `read_file`: `0.15` when reading the true bug file first, `0.05` for other first-time reads
- `apply_patch`: `1.00` for a full fix, `0.30` for partial progress, `0.00` otherwise
- `done`: `1.00` if all tests pass, `0.00` otherwise

## Baseline Inference

`inference.py` uses the OpenAI Python client, emits the required `[START]`, `[STEP]`, and `[END]` stdout format, and completes all three tasks on the live Space.

Latest verified baseline:

- Task 1: `success=true`, `steps=4`, `score=0.867`
- Task 2: `success=true`, `steps=4`, `score=0.867`
- Task 3: `success=true`, `steps=4`, `score=0.867`

## Setup And Validation

Install the package locally:

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

Build the Docker image:

```bash
docker build -t bugforge-env .
```

Run the baseline:

```bash
python inference.py
```

Run the validator:

```bash
./validate.sh https://pranayteja31-BugForge.hf.space .
```

The simplest way to use the Bugforge environment is through the `BugforgeEnv` class:

```python
from bugforge import BugforgeAction, BugforgeEnv

try:
    # Create environment from Docker image
    bugforgeenv = BugforgeEnv.from_docker_image("bugforge-env:latest")

    # Reset
    result = bugforgeenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = bugforgeenv.step(BugforgeAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    bugforgeenv.close()
```

That's it! The `BugforgeEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t bugforge-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**BugforgeAction**
- `type` (str) - action kind: `run_tests`, `read_file`, `apply_patch`, or `done`
- `file` (str) - file to inspect or patch
- `old_code` (str) - exact code to replace
- `new_code` (str) - replacement code

### Observation
**BugforgeObservation**
- `output` (str) - raw output from the last action
- `tests_passing` (int) - number of passing tests
- `tests_total` (int) - total test count
- `reward` (float) - Reward based on message length (length × 0.1)
- `files_read` (list[str]) - unique files inspected so far
- `steps_remaining` (int) - remaining step budget
- `patches_applied` (int) - number of patch attempts

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Bugforge environment server running, you can connect directly:

```python
from bugforge import BugforgeEnv

# Connect to existing server
bugforgeenv = BugforgeEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = bugforgeenv.reset()
result = bugforgeenv.step(BugforgeAction(message="Hello!"))
```

Note: When connecting to an existing server, `bugforgeenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from bugforge import BugforgeAction, BugforgeEnv

# Connect with context manager (auto-connects and closes)
with BugforgeEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(BugforgeAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    BugforgeEnvironment,  # Pass class, not instance
    BugforgeAction,
    BugforgeObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from bugforge import BugforgeAction, BugforgeEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with BugforgeEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(BugforgeAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/bugforge_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
bugforge/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # BugforgeEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── bugforge_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
