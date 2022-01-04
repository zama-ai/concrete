"""Script to generate custom GitHub actions test matrices."""

import argparse
import itertools
import json
from pathlib import Path

WEEKLY = "weekly"
RELEASE = "release"
PR = "pr"

LINUX = "linux"
MACOS = "macos"

OSES = {LINUX, MACOS}

PR_OSES = {LINUX: "ubuntu-20.04"}
PR_PYTHON_VERSIONS = ["3.8"]
PR_CONF = {"os": PR_OSES, "python": PR_PYTHON_VERSIONS}

WEEKLY_OSES = {
    LINUX: "ubuntu-20.04",
    MACOS: "macos-10.15",
}
WEEKLY_PYTHON_VERSIONS = ["3.8", "3.9"]
WEEKLY_CONF = {"os": WEEKLY_OSES, "python": WEEKLY_PYTHON_VERSIONS}

RELEASE_OSES = {
    LINUX: "ubuntu-20.04",
    MACOS: "macos-10.15",
}
RELEASE_PYTHON_VERSIONS = ["3.8", "3.9"]
RELEASE_CONF = {"os": RELEASE_OSES, "python": RELEASE_PYTHON_VERSIONS}

CONFIGURATIONS = {PR: PR_CONF, WEEKLY: WEEKLY_CONF, RELEASE: RELEASE_CONF}


def main(args):
    """Entry point."""

    matrix_conf = CONFIGURATIONS[args.build_type]

    github_action_matrix = []

    for (os_kind, os_name), python_version in itertools.product(
        matrix_conf["os"].items(), matrix_conf["python"]
    ):
        github_action_matrix.append(
            {
                "os_kind": os_kind,
                "runs_on": os_name,
                "python_version": python_version,
            }
        )

    print(json.dumps(github_action_matrix, indent=4))

    output_json_path = Path(args.output_json).resolve()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(github_action_matrix, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate GHA test matrices", allow_abbrev=False)

    parser.add_argument(
        "--build-type",
        type=str,
        required=True,
        choices=[WEEKLY, RELEASE, PR],
        help="The type of build for which the matrix generation is required",
    )

    parser.add_argument(
        "--output-json", type=str, required=True, help="Where to output the matrix as json data"
    )

    cli_args = parser.parse_args()

    main(cli_args)
