"""
BugForge inference script.

Reads environment variables:
  - API_BASE_URL: LLM API base URL (default: Hugging Face router)
  - MODEL_NAME: model to use (default: Qwen/Qwen2.5-72B-Instruct)
  - HF_TOKEN / API_KEY: API authentication token
  - HF_SPACE_URL: deployed Space or local server URL

Mandatory log format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
import time
from typing import Any

from openai import OpenAI

# Hackathon evaluator injects API_KEY + API_BASE_URL for proxy-tracked calls.
# Prefer API_KEY first so we never bypass their meter when both are present.
# Phase 2 deep validation expects calls through the injected LiteLLM proxy.
# Use explicit required env vars so we don't accidentally bypass proxy settings.
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "bugforge"
MAX_STEPS = 8
TEMPERATURE = 0.1
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a debugging agent. A Python project has exactly one hidden bug.
    Your goal is to fix the bug in as few steps as possible.

    Available actions:
    {"type": "run_tests"}
    {"type": "read_file", "file": "filename.py"}
    {"type": "apply_patch", "file": "filename.py", "old_code": "exact code to replace", "new_code": "fixed code"}
    {"type": "done"}

    Available files: utils.py, cart.py, models.py, tests.py

    Strategy:
    1. run_tests first
    2. read the most relevant file
    3. apply one exact patch
    4. run_tests again
    5. call done only after all tests pass

    IMPORTANT:
    - Output only one JSON object
    - Do not include markdown fences
    - old_code must be exact
    - Prefer minimal patches
    - Do not propose speculative refactors
    - If a patch fails, inspect the file again and copy exact text
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = str(error).replace("\n", "\\n").replace("\r", "") if error else "null"
    action_val = str(action).replace("\n", "\\n").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def debug_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_action(action_str: str) -> dict[str, Any]:
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", action_str, re.DOTALL)
        if match:
            action_str = match.group(1)
        else:
            start = action_str.find("{")
            end = action_str.rfind("}")
            if start != -1 and end != -1:
                action_str = action_str[start : end + 1]
        action = json.loads(action_str)
        if isinstance(action, dict):
            return action
    except (json.JSONDecodeError, TypeError):
        pass
    return {"type": "run_tests"}


def choose_file_from_output(output: str, files_read: list[str]) -> str:
    """Pick a likely next file using error-shape heuristics, not hardcoded fixes."""
    text = output.lower()

    if "typeerror" in text or "can't multiply sequence" in text or "get_item_quantity" in text:
        return "models.py"
    if "coupon" in text or "apply_coupon" in text or "none" in text:
        return "cart.py"
    if "discount" in text or "calculate_discount" in text or "username" in text or "handle" in text:
        return "utils.py"
    if "shipping" in text or "zone" in text or "zip" in text:
        return "models.py" if "models.py" not in files_read else "cart.py"

    for filename in ("utils.py", "models.py", "cart.py", "tests.py"):
        if filename not in files_read:
            return filename
    return "tests.py"


def should_read_models(observation: dict[str, Any], file_cache: dict[str, str]) -> bool:
    """Escalate from a caller file to models.py when the visible code points there."""
    if "models.py" in observation.get("files_read", []):
        return False

    output = observation.get("output", "").lower()
    cart_content = file_cache.get("cart.py", "")
    if not cart_content:
        return False

    return (
        "from models import" in cart_content
        and (
            "typeerror" in output
            or "can't multiply sequence" in output
            or "shipping" in output
            or "zone" in output
            or "zip" in output
        )
    )


def should_read_tests_before_patch(observation: dict[str, Any], file_cache: dict[str, str]) -> bool:
    """Use tests as cheap grounding for behavioral bugs before the first patch."""
    files_read = observation.get("files_read", [])
    if "tests.py" in files_read:
        return False

    output = observation.get("output", "").lower()
    patches_applied = observation.get("patches_applied", 0)
    if patches_applied > 0:
        return False

    if "shipping" in output or "zone" in output or "zip" in output:
        return True
    if "coupon" in output or "apply_coupon" in output:
        return True
    if "username" in output or "handle" in output:
        return True

    for name in ("cart.py", "models.py", "utils.py"):
        if name in files_read and name in file_cache:
            content = file_cache[name].lower()
            if "get_shipping_zone" in content or "apply_coupon" in content or "normalize_username" in content:
                return True

    return False


def should_reinspect_after_failed_patch(observation: dict[str, Any], history: list[str]) -> bool:
    """After a zero-reward patch, gather more evidence instead of patching again immediately."""
    if not history:
        return False
    if '"type": "apply_patch"' not in history[-1]:
        return False
    return observation.get("patches_applied", 0) > 0 and "patch applied. all tests pass!" not in observation.get("output", "").lower()


def default_action(step: int, observation: dict[str, Any]) -> dict[str, Any]:
    tests_passing = observation.get("tests_passing", 0)
    tests_total = observation.get("tests_total", 0)
    if tests_total > 0 and tests_passing == tests_total:
        return {"type": "done"}
    if step == 1:
        return {"type": "run_tests"}
    return {"type": "run_tests"}


def last_action_was_patch(history: list[str]) -> bool:
    if not history:
        return False
    return '"type": "apply_patch"' in history[-1]


def get_action(
    client: OpenAI,
    step: int,
    observation: dict[str, Any],
    history: list[str],
    file_cache: dict[str, str],
) -> str:
    tests_passing = observation.get("tests_passing", 0)
    tests_total = observation.get("tests_total", 0)
    files_read = observation.get("files_read", [])
    patches_applied = observation.get("patches_applied", 0)

    if tests_total > 0 and tests_passing == tests_total:
        return json.dumps({"type": "done"}, ensure_ascii=True)

    history_block = "\n".join(history[-4:]) if history else "None"
    output = observation.get("output", "")
    steps_remaining = observation.get("steps_remaining", 0)
    visible_files = {name: file_cache[name] for name in files_read if name in file_cache}
    suggested_action = default_action(step, observation)

    user_prompt = textwrap.dedent(
        f"""\
        Step: {step}
        Output:
        {output}
        Tests passing: {tests_passing}/{tests_total}
        Files read: {files_read}
        File contents:
        {json.dumps(visible_files, ensure_ascii=True)}
        Patches applied: {patches_applied}
        Steps remaining: {steps_remaining}
        Suggested fallback action if unsure:
        {json.dumps(suggested_action, ensure_ascii=True)}
        Previous steps:
        {history_block}
        Choose the next action.
        """
    ).strip()

    followup_prompt = (
        "Your previous reply was not a valid single JSON action object. "
        "Reply again with only one valid JSON object using one of the allowed action shapes."
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for _ in range(2):
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=120,
            )
            raw = (completion.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            action = parse_action(raw)
            if action.get("type") in {"run_tests", "read_file", "apply_patch", "done"}:
                return json.dumps(action, ensure_ascii=True)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": followup_prompt})
        raise ValueError("Model did not return a valid action after retry")
    except Exception as exc:
        debug_log(f"[DEBUG] Model error: {exc}")
        return json.dumps(default_action(step, observation), ensure_ascii=True)


def finalize_task_ws(ws, rewards: list[float], steps_taken: int) -> tuple[int, bool, dict[str, Any]]:
    ws.send(json.dumps({"type": "step", "data": {"type": "done"}}))
    obs_data, reward, _ = unpack_ws_payload(ws.recv())
    rewards.append(reward)
    steps_taken += 1
    log_step(step=steps_taken, action='{"type": "done"}', reward=reward, done=True, error=None)
    return steps_taken, True, obs_data


def finalize_task_http(base_url: str, rewards: list[float], steps_taken: int) -> tuple[int, bool, dict[str, Any]]:
    import requests

    response = requests.post(f"{base_url}/step", json={"action": {"type": "done"}}, timeout=30)
    response.raise_for_status()
    done_data = response.json()
    reward = done_data.get("reward", 0.0) or 0.0
    rewards.append(reward)
    steps_taken += 1
    log_step(step=steps_taken, action='{"type": "done"}', reward=reward, done=True, error=None)
    final_obs_data = done_data.get("observation", {}) or done_data
    return steps_taken, True, final_obs_data


def wait_for_server(server_url: str) -> None:
    import requests

    for attempt in range(5):
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except Exception:
            pass

        try:
            response = requests.post(f"{server_url}/reset", json={}, timeout=10)
            if response.status_code == 200:
                return
        except Exception:
            pass

        debug_log(f"[DEBUG] Waiting for server... attempt {attempt + 1}/5")
        time.sleep(2)


def can_use_websocket(base_url: str) -> bool:
    try:
        import websocket

        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/") + "/ws"
        ws = websocket.create_connection(ws_url, timeout=5)
        ws.close()
        return True
    except Exception:
        return False


def unpack_ws_payload(message: str) -> tuple[dict[str, Any], float, bool]:
    payload = json.loads(message).get("data", {})
    observation = payload.get("observation", {})
    reward = payload.get("reward", 0.0) or 0.0
    done = payload.get("done", False)
    return observation, reward, done


def calculate_score(final_obs_data: dict[str, Any], steps_taken: int) -> tuple[float, bool]:
    tests_passing = final_obs_data.get("tests_passing", 0)
    tests_total = final_obs_data.get("tests_total", 0)

    if tests_total > 0 and tests_passing == tests_total:
        remaining = max(final_obs_data.get("steps_remaining", 0), 0)
        total_budget = steps_taken + remaining
        efficiency = (remaining / total_budget) if total_budget > 0 else 0.0
        score = 0.8 + (0.2 * efficiency)
        return round(min(max(score, 0.0), 1.0), 3), True

    if tests_total > 0 and tests_passing > 0:
        partial = (tests_passing / tests_total) * 0.6
        return round(min(max(partial, 0.0), 1.0), 3), False

    return 0.0, False


def run_task_ws(client: OpenAI, task_id: int, base_url: str) -> None:
    import websocket

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + "/ws"

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[str] = []
    final_obs_data: dict[str, Any] = {}
    file_cache: dict[str, str] = {}

    log_start(task=f"debug-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    ws = None
    try:
        ws = websocket.create_connection(ws_url, timeout=30)
        ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        obs_data, _, is_done = unpack_ws_payload(ws.recv())
        final_obs_data = obs_data

        for step in range(1, MAX_STEPS + 1):
            if is_done:
                break

            action_str = get_action(client, step, obs_data, history, file_cache)
            action_dict = parse_action(action_str)

            ws.send(json.dumps({"type": "step", "data": action_dict}))
            obs_data, reward, is_done = unpack_ws_payload(ws.recv())

            if action_dict.get("type") == "read_file" and action_dict.get("file"):
                file_cache[action_dict["file"]] = obs_data.get("output", "")

            rewards.append(reward)
            steps_taken = step
            final_obs_data = obs_data
            log_step(step=step, action=action_str, reward=reward, done=is_done, error=None)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if is_done:
                break

        if not is_done:
            steps_taken, is_done, final_obs_data = finalize_task_ws(ws, rewards, steps_taken)

        score, success = calculate_score(final_obs_data, steps_taken)
    except Exception as exc:
        debug_log(f"[DEBUG] Task {task_id} error: {exc}")
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_task_http(client: OpenAI, task_id: int, base_url: str) -> None:
    import requests

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[str] = []
    final_obs_data: dict[str, Any] = {}
    file_cache: dict[str, str] = {}

    log_start(task=f"debug-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    try:
        response = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
        response.raise_for_status()
        reset_data = response.json()
        obs_data = reset_data.get("observation", {})
        final_obs_data = obs_data
        is_done = reset_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if is_done:
                break

            action_str = get_action(client, step, obs_data, history, file_cache)
            action_dict = parse_action(action_str)

            response = requests.post(f"{base_url}/step", json={"action": action_dict}, timeout=30)
            response.raise_for_status()
            step_data = response.json()
            obs_data = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0) or 0.0
            is_done = step_data.get("done", False)

            if action_dict.get("type") == "read_file" and action_dict.get("file"):
                file_cache[action_dict["file"]] = obs_data.get("output", "")

            rewards.append(reward)
            steps_taken = step
            final_obs_data = obs_data
            log_step(step=step, action=action_str, reward=reward, done=is_done, error=None)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if is_done:
                break

        if not is_done:
            steps_taken, is_done, final_obs_data = finalize_task_http(base_url, rewards, steps_taken)

        score, success = calculate_score(final_obs_data, steps_taken)
    except Exception as exc:
        debug_log(f"[DEBUG] Task {task_id} HTTP error: {exc}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    server_url = os.getenv("HF_SPACE_URL", "http://localhost:8000")

    wait_for_server(server_url)
    use_ws = can_use_websocket(server_url)

    for task_id in [1, 2, 3, 4, 5]:
        if use_ws:
            run_task_ws(client, task_id, server_url)
        else:
            run_task_http(client, task_id, server_url)


if __name__ == "__main__":
    main()
