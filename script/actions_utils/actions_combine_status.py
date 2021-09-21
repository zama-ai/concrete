# -*- coding: utf-8 -*-

"""Helper script for github actions to combine job statuses"""
import argparse
import json

RESULTS_TO_DISPLAY_LEVEL = {
    "failure": 0,
    "cancelled": 1,
    "success": 2,
    "skipped": 3,
}

DISPLAY_LEVEL_TO_RESULTS = {val: key for key, val in RESULTS_TO_DISPLAY_LEVEL.items()}


def main(args):
    """Entry point"""

    need_context_data = None
    with open(args.needs_context_json, encoding="utf-8") as f:
        need_context_data = json.load(f)

    display_level = min(
        RESULTS_TO_DISPLAY_LEVEL[job_object["result"]] for job_object in need_context_data.values()
    )

    print(DISPLAY_LEVEL_TO_RESULTS[display_level])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Combine github actions statuses", allow_abbrev=False)

    parser.add_argument(
        "--needs_context_json",
        type=str,
        help="Pass the json file path containing the workflow needs context",
    )

    cli_args = parser.parse_args()

    main(cli_args)
