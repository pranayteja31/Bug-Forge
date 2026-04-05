import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from bugforge import BugforgeAction, BugforgeEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")
TASK_NAME = os.getenv("BUGFORGE_TASK", "debug")
BENCHMARK = "bugforge"
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
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

    Respond ONLY with a valid JSON action. Nothing else.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client, step, last_obs, history):
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
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
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return '{"type": "run_tests"}'


async def run_task(client, task_id):
    import json

    env = await BugforgeEnv.from_docker_image(IMAGE_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    history = []

    log_start(task=f"debug-task{task_id}", env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_obs = str(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str = get_action(client, step, last_obs, history)

            try:
                action_dict = json.loads(action_str)
                action = BugforgeAction(**action_dict)
            except:
                action = BugforgeAction(type="run_tests")

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = str(result.observation)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_STEPS
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in [1, 2, 3]:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())