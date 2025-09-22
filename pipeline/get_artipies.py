# The repository structure of artipes has changed. This script pulls the repos from bump dataset.

import json
from pathlib import Path
import subprocess

HOME = Path.home()
DATA_ROOT = HOME / "breaking_updates_rl/" / "data"

COMMAND_TEMPLATE = [
    "apptainer", "exec",
    "--bind",
    str(DATA_ROOT) + "/dataset/{breakingCommit}:/out",
    "docker://ghcr.io/chains-project/breaking-updates:{breakingCommit}-breaking",
    "cp",
    "-r",
    "/{project}",
    "/out/"
]


breaking_commit_list = ["13fd75e233a5cb2771a6cb186c0decaed6d6545a",
                        "497b81f4446c257f693648cad7a64f62b23920a2",
                        "4aab2869639226035c999c282f31efba15648ea3",
                        "5fca04bd287baf1534baff8cc23cc3dc26dc680d",
                        "9836e07e553e29f16ee35b5d7e4d0370e1789ecd",
                        "a0ec50cb297c4202e138627859a89ef032fa78ab",
                        "ab85440ce7321d895c7a9621224ce8059162a26a",
                        "ae0a0bd1311451e4a5a185a8d96405cfe3e049c5",
                        "c311ee0a84b72b15ba64da3514181c2347912225",
                        "d38182a8a0fe1ec039aed97e103864fce717a0be",
                        "db02c6bcb989a5b0f08861c3344b532769530467",
                        "e36118e158965251089da1446777bd699d7473c1",
                        "fab0f8d4f7322fa6da914c8c9f30baf740d46b99"]



def repo_completion() -> None:

    for breaking_commit in breaking_commit_list:
        metadata_path = DATA_ROOT / "benchmark" / f"{breaking_commit}.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            project = metadata.get("project")
            
        command = COMMAND_TEMPLATE.copy()
        command[3] = command[3].format(breakingCommit=breaking_commit)
        command[4] = command[4].format(breakingCommit=breaking_commit)
        command[7] = command[7].format(project=project)

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {breaking_commit}: {e}")

if __name__ == "__main__":
    repo_completion()