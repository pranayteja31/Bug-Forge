import json
import shutil
import os

def inject_bug(task_id):
    task_dir = f"bugs/task{task_id}"
    config_path = f"{task_dir}/bug_config.json"

    with open(config_path) as f:
        config = json.load(f)

    # Copy clean files to working directory
    working_dir = f"bugs/task{task_id}/working"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    shutil.copytree(f"{task_dir}/clean", working_dir)

    # Inject the bug
    bug_file = f"{working_dir}/{config['bug_file']}"
    with open(bug_file, 'r') as f:
        content = f.read()

    content = content.replace(
        config['original_code'],
        config['buggy_code']
    )

    with open(bug_file, 'w') as f:
        f.write(content)

    return working_dir, config

def get_ground_truth(task_id):
    with open(f"bugs/task{task_id}/bug_config.json") as f:
        return json.load(f)