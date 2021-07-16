# -*- coding: utf-8 -*-

"""Helper script for github actions"""
import argparse
import traceback
from pathlib import Path


def main(args):
    """Entry point"""
    diff_cover_file_path = Path(args.diff_cover_output).resolve().absolute()

    diff_cover_content = None

    with open(diff_cover_file_path, "r") as f:
        diff_cover_content = f.readlines()

    with open(diff_cover_file_path, "w", encoding="utf-8") as f:
        if args.diff_cover_exit_code == 0:
            f.write("## Coverage passed ✅\n\n")
        else:
            f.write("## Coverage failed ❌\n\n")

        # Open collapsible section
        f.write("<details><summary>Coverage details</summary>\n<p>\n\n")
        f.write("```\n")

        f.writelines(diff_cover_content)

        # Close collapsible section
        f.write("```\n\n")
        f.write("</p>\n</details>\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--diff-cover-exit-code", type=int, required=True)
    parser.add_argument("--diff-cover-output", type=str, required=True)

    cli_args = parser.parse_args()
    try:
        main(cli_args)
    except Exception as e:
        traceback.print_exc()
