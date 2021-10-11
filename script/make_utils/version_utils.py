"""Tool to manage version in the project"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import tomlkit
from semver import VersionInfo


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible."""
    return version_str[1:] if version_str.startswith("v") else version_str


def islatest(args):
    """islatest command entry point."""
    print(args, file=sys.stderr)

    # This is the safest default
    result = {"is_latest": False, "is_prerelease": True}

    new_version_str = strip_leading_v(args.new_version)
    if VersionInfo.isvalid(new_version_str):
        new_version_info = VersionInfo.parse(new_version_str)
        if new_version_info.prerelease is None:
            # If it's an actual release
            all_versions_str = (
                strip_leading_v(version_str) for version_str in args.existing_versions
            )

            # Keep versions that are not release candidate
            all_non_prerelease_version_infos = [
                version_info
                for version_str in all_versions_str
                if VersionInfo.isvalid(version_str)
                and (version_info := VersionInfo.parse(version_str))
                and version_info.prerelease is None
            ]

            all_non_prerelease_version_infos.append(new_version_info)

            new_version_is_latest = max(all_non_prerelease_version_infos) == new_version_info
            result["is_latest"] = new_version_is_latest
            result["is_prerelease"] = False

    print(json.dumps(result))


def update_variable_in_py_file(file_path: Path, var_name: str, version_str: str):
    """Update the version in a .py file."""

    file_content = None
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()

    updated_file_content = re.sub(
        rf'{var_name} *[:=] *["\'](.+)["\']',
        rf'{var_name} = "{version_str}"',
        file_content,
    )

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated_file_content)


def update_variable_in_toml_file(file_path: Path, var_name: str, version_str: str):
    """Update the version in a .toml file."""
    toml_content = None
    with open(file_path, encoding="utf-8") as f:
        toml_content = tomlkit.loads(f.read())

    toml_keys = var_name.split(".")
    current_content = toml_content
    for toml_key in toml_keys[:-1]:
        current_content = current_content[toml_key]
    last_toml_key = toml_keys[-1]
    current_content[last_toml_key] = version_str

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(tomlkit.dumps(toml_content))


def load_file_vars_set(pyproject_path: os.PathLike, cli_file_vars: Optional[List[str]]):
    """Load files and their version variables set-up in pyproject.toml and passed as arguments."""

    file_vars_set = set()
    if cli_file_vars is not None:
        file_vars_set.update(cli_file_vars)

    pyproject_path = Path(pyproject_path).resolve()

    # Check if there is a semantic release configuration
    if pyproject_path.exists():
        pyproject_content = None
        with open(pyproject_path, encoding="utf-8") as f:
            pyproject_content = tomlkit.loads(f.read())

        try:
            sr_conf = pyproject_content["tool"]["semantic_release"]
            sr_version_toml: str = sr_conf.get("version_toml", "")
            file_vars_set.update(sr_version_toml.split(","))
            sr_version_variable: str = sr_conf.get("version_variable", "")
            file_vars_set.update(sr_version_variable.split(","))
        except KeyError:
            print("No configuration for semantic release in pyproject.toml")

    return file_vars_set


def set_version(args):
    """set-version command entry point."""

    version_str = strip_leading_v(args.version)
    if not VersionInfo.isvalid(version_str):
        raise RuntimeError(f"Unable to validate version: {args.version}")

    file_vars_set = load_file_vars_set(args.pyproject_file, args.file_vars)

    for file_var_str in sorted(file_vars_set):
        print(f"Processing {file_var_str}")
        file, var_name = file_var_str.split(":", 1)
        file_path = Path(file).resolve()

        if file_path.suffix == ".py":
            update_variable_in_py_file(file_path, var_name, version_str)
        elif file_path.suffix == ".toml":
            update_variable_in_toml_file(file_path, var_name, version_str)
        else:
            raise RuntimeError(f"Unsupported file extension: {file_path.suffix}")


def get_variable_from_py_file(file_path: Path, var_name: str):
    """Read variable value from a .py file."""
    file_content = None
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()

    variable_values_set = set()

    start_pos = 0
    while True:
        file_content = file_content[start_pos:]
        match = re.search(
            rf'{var_name} *[:=] *["\'](.+)["\']',
            file_content,
        )
        if match is None:
            break

        variable_values_set.add(match.group(1))
        start_pos = match.end()

    return variable_values_set


def get_variable_from_toml_file(file_path: Path, var_name: str):
    """Read variable value from a .toml file."""

    toml_content = None
    with open(file_path, encoding="utf-8") as f:
        toml_content = tomlkit.loads(f.read())

    toml_keys = var_name.split(".")
    current_content = toml_content
    for toml_key in toml_keys:
        current_content = current_content[toml_key]

    return current_content


def check_version(args):
    """check-version command entry point."""

    version_str_set = set()

    file_vars_set = load_file_vars_set(args.pyproject_file, args.file_vars)

    for file_var_str in sorted(file_vars_set):
        print(f"Processing {file_var_str}")
        file, var_name = file_var_str.split(":", 1)
        file_path = Path(file).resolve()

        if file_path.suffix == ".py":
            version_str_set.update(get_variable_from_py_file(file_path, var_name))
        elif file_path.suffix == ".toml":
            version_str_set.add(get_variable_from_toml_file(file_path, var_name))
        else:
            raise RuntimeError(f"Unsupported file extension: {file_path.suffix}")

    if len(version_str_set) == 0:
        raise RuntimeError(f"No versions found in {', '.join(sorted(file_vars_set))}")
    if len(version_str_set) > 1:
        raise RuntimeError(
            f"Found more than one version: {', '.join(sorted(version_str_set))}\n"
            "Re-run make set-version"
        )
    # Now version_str_set len == 1
    if not VersionInfo.isvalid((version := next(iter(version_str_set)))):
        raise RuntimeError(f"Unable to validate version: {version}")

    print(f"Found version {version} in all processed locations.")


def main(args):
    """Entry point"""
    args.entry_point(args)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser("Version utils", allow_abbrev=False)

    sub_parsers = main_parser.add_subparsers(dest="sub-command", required=True)

    parser_islatest = sub_parsers.add_parser("islatest")
    parser_islatest.add_argument(
        "--new-version", type=str, required=True, help="The new version to compare"
    )
    parser_islatest.add_argument(
        "--existing-versions",
        type=str,
        nargs="+",
        required=True,
        help="The list of existing versions",
    )
    parser_islatest.set_defaults(entry_point=islatest)

    parser_set_version = sub_parsers.add_parser("set-version")
    parser_set_version.add_argument("--version", type=str, required=True, help="The version to set")
    parser_set_version.add_argument(
        "--pyproject-file",
        type=str,
        default="pyproject.toml",
        help="The path to a project's pyproject.toml file, defaults to $pwd/pyproject.toml",
    )
    parser_set_version.add_argument(
        "--file-vars",
        type=str,
        nargs="+",
        help=(
            "A space separated list of file/path.{py, toml}:variable to update with the new version"
        ),
    )
    parser_set_version.set_defaults(entry_point=set_version)

    parser_check_version = sub_parsers.add_parser("check-version")
    parser_check_version.add_argument(
        "--pyproject-file",
        type=str,
        default="pyproject.toml",
        help="The path to a project's pyproject.toml file, defaults to $pwd/pyproject.toml",
    )
    parser_check_version.add_argument(
        "--file-vars",
        type=str,
        nargs="+",
        help=(
            "A space separated list of file/path.{py, toml}:variable to update with the new version"
        ),
    )
    parser_check_version.set_defaults(entry_point=check_version)

    cli_args = main_parser.parse_args()

    main(cli_args)
