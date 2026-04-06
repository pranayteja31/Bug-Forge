"""
BugForge inference script — runs the LLM debugging agent against all tasks.

Reads environment variables:
  - API_BASE_URL: LLM API base URL (default: HuggingFace router)
  - MODEL_NAME: model to use (default: Qwen/Qwen2.5-72B-Instruct)
  - HF_TOKEN / API_KEY: API authentication token
  - IMAGE_NAME: Docker image name for the BugForge environment

Log format (mandatory — do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import textwrap
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")
BENCHMARK = "bugforge"
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a debugging agent. A Python project has a hidden bug.
    Your goal is to find and fix it using these actions:

    {"type": "run_tests"}
    {"type": "read_file", "file": "filename.py"}
    {"type": "apply_patch", "file": "filename.py", "old_code": "...", "new_code": "..."}
    {"type": "done"}

    Strategy:
    1. Always run_tests first to see what's failing
    2. Read relevant files to find the bug
    3. Apply a patch to fix it
    4. Run tests again to verify
    5. Call done when all tests pass

    Respond ONLY with a valid JSON action object. Nothing else.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (strict format — do NOT modify)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = str(error).replace('\n', '\\n').replace('\r', '') if error else "null"
    action_val = str(action).replace('\n', '\\n').replace('\r', '')
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------
def get_action(client: OpenAI, step: int, last_obs: str, history: list) -> str:
    """Ask the LLM for the next action given current observation and history."""
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""\
        Step: {step}
        Current observation: {last_obs}
        Previous steps:
        {history_block}
        Choose your next action.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=30,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if the model wraps the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return '{"type": "run_tests"}'


def parse_action(action_str: str) -> dict:
    """Parse an action string to a dict, falling back to run_tests on error."""
    try:
        return json.loads(action_str)
    except (json.JSONDecodeError, TypeError):
        return {"type": "run_tests"}


# ---------------------------------------------------------------------------
# Environment interaction via WebSocket
# ---------------------------------------------------------------------------
def run_task_ws(client: OpenAI, task_id: int, base_url: str) -> None:
    """Run a single task using the WebSocket endpoint of a running server."""
    import websocket  # websocket-client (sync)

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + "/ws"

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[str] = []

    log_start(task=f"debug-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    ws = None
    try:
        ws = websocket.create_connection(ws_url, timeout=30)

        # Reset
        ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        reset_resp = json.loads(ws.recv())
        obs_data = reset_resp.get("data", {})
        last_obs = json.dumps(obs_data)
        is_done = obs_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if is_done:
                break

            action_str = get_action(client, step, last_obs, history)
            action_dict = parse_action(action_str)

            # Send step
            ws.send(json.dumps({"type": "step", "data": action_dict}))
            step_resp = json.loads(ws.recv())
            obs_data = step_resp.get("data", {})

            reward = obs_data.get("reward", 0.0) or 0.0
            is_done = obs_data.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = json.dumps(obs_data)

            log_step(step=step, action=action_str, reward=reward, done=is_done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if is_done:
                break

        if rewards:
            score = sum(rewards) / MAX_STEPS
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Environment interaction via HTTP (fallback)
# ---------------------------------------------------------------------------
def run_task_http(client: OpenAI, task_id: int, base_url: str) -> None:
    """Run a single task using HTTP endpoints (stateless — each call gets fresh env)."""
    import requests

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[str] = []

    log_start(task=f"debug-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        r = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        reset_data = r.json()
        obs_data = reset_data.get("observation", {})
        last_obs = json.dumps(obs_data)
        is_done = reset_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if is_done:
                break

            action_str = get_action(client, step, last_obs, history)
            action_dict = parse_action(action_str)

            r = requests.post(
                f"{base_url}/step",
                json={"action": action_dict},
                timeout=30,
            )
            r.raise_for_status()
            step_data = r.json()
            obs_data = step_data.get("observation", {})

            reward = step_data.get("reward", 0.0) or 0.0
            is_done = step_data.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = json.dumps(obs_data)

            log_step(step=step, action=action_str, reward=reward, done=is_done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if is_done:
                break

        if rewards:
            score = sum(rewards) / MAX_STEPS
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} HTTP error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Determine server URL — use IMAGE_NAME if provided (Docker), else local
    server_url = os.getenv("HF_SPACE_URL", "http://localhost:8000")

    # Quick healthcheck / warmup ping
    import requests
    for attempt in range(5):
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"[DEBUG] Waiting for server... attempt {attempt + 1}/5", flush=True)
        time.sleep(2)

    # Try WebSocket first (stateful), fall back to HTTP (stateless)
    try:
        import websocket  # noqa: F401
        use_ws = True
    except ImportError:
        use_ws = False

    for task_id in [1, 2, 3]:
        if use_ws:
            run_task_ws(llm_client, task_id, server_url)
        else:
            run_task_http(llm_client, task_id, server_url)


if __name__ == "__main__":
    main()