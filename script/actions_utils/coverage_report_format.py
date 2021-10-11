# -*- coding: utf-8 -*-

"""Helper script for github actions"""
import argparse
import json
from pathlib import Path


def write_coverage_file(coverage_file_path: Path, exit_code: int, coverage_content):
    """Write the formatted coverage to file."""
    with open(coverage_file_path, "w", encoding="utf-8") as f:
        if exit_code == 0:
            f.write("## Coverage passed ✅\n\n")
        else:
            f.write("## Coverage failed ❌\n\n")

        # Open collapsible section
        f.write("<details><summary>Coverage details</summary>\n<p>\n\n")
        f.write("```\n")

        f.writelines(coverage_content)

        # Close collapsible section
        f.write("```\n\n")
        f.write("</p>\n</details>\n\n")


def diff_coverage(args):
    """diff-coverage entry point."""
    diff_cover_file_path = Path(args.diff_cover_output).resolve()
    diff_cover_content = None

    with open(diff_cover_file_path, "r", encoding="utf-8") as f:
        diff_cover_content = f.readlines()

    write_coverage_file(diff_cover_file_path, args.diff_cover_exit_code, diff_cover_content)


def global_coverage(args):
    """global-coverage entry point."""
    global_coverage_json_path = Path(args.global_coverage_json_file).resolve()
    global_coverage_infos = None
    with open(global_coverage_json_path, "r", encoding="utf-8") as f:
        global_coverage_infos = json.load(f)

    exit_code = global_coverage_infos["exit_code"]
    coverage_content = global_coverage_infos["content"]
    global_coverage_output_file_path = Path(args.global_coverage_output_file).resolve()
    write_coverage_file(global_coverage_output_file_path, exit_code, coverage_content)


def main(args):
    """Entry point"""
    args.entry_point(args)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(allow_abbrev=False)

    sub_parsers = main_parser.add_subparsers(dest="sub-command", required=True)

    parser_diff_coverage = sub_parsers.add_parser("diff-coverage")

    parser_diff_coverage.add_argument("--diff-cover-exit-code", type=int, required=True)
    parser_diff_coverage.add_argument("--diff-cover-output", type=str, required=True)
    parser_diff_coverage.set_defaults(entry_point=diff_coverage)

    parser_global_coverage = sub_parsers.add_parser("global-coverage")

    parser_global_coverage.add_argument("--global-coverage-output-file", type=str, required=True)
    parser_global_coverage.add_argument("--global-coverage-json-file", type=str, required=True)
    parser_global_coverage.set_defaults(entry_point=global_coverage)

    cli_args = main_parser.parse_args()

    main(cli_args)
