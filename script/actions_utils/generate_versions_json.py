"""Tool to manage the versions.json file at the root of our docs sites."""

import argparse
import json
from json.decoder import JSONDecodeError
from pathlib import Path

from semver import VersionInfo


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible."""
    return version_str[1:] if version_str.startswith("v") else version_str


def main(args):
    """Entry point."""
    version = args.version
    latest = args.latest
    prerelease = args.prerelease

    if not VersionInfo.isvalid(strip_leading_v(version)):
        raise RuntimeError(f"Invalid version: {version}")

    version_json_file_path = Path(args.versions_json_file).resolve()
    try:
        with open(version_json_file_path, "r", encoding="utf-8") as f:
            version_json = json.loads(f.read())
    except JSONDecodeError as err:
        raise RuntimeError(
            f"An error occurred while trying to load {str(version_json_file_path)}"
        ) from err

    # Version json is composed by:
    #  all: list of all published versions
    #  menu: list of all available versions (if any entry is not included in "all",
    #        warning banner with DEV/PRE-RELEASE doc warning will be displayed)
    #  latest: latest version, if current doc != latest, warning banner is displayed
    if "version" not in version_json["menu"]:
        version_json["menu"].append(version)
    if not prerelease:
        version_json["all"].append(version)
        if latest:
            version_json["latest"] = version

    print(version_json)
    output_json_path = Path(args.output_json).resolve()
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(version_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("versions.json generator", allow_abbrev=False)

    parser.add_argument(
        "--add-version",
        type=str,
        required=True,
        dest="version",
        help="A single versions to add to versions.json. "
        "The link will point to '$VERSION/' and will have text '$VERSION'.",
    )
    parser.add_argument(
        "--versions-json-file", type=str, required=True, help="Path to the versions.json to update."
    )
    parser.add_argument(
        "--prerelease",
        action="store_true",
        dest="prerelease",
        help="set this version as a pre-release documentation.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        dest="latest",
        help="set this version as latest available documentation.",
    )
    parser.add_argument("--output-json", type=str, required=True, help="Output file path.")

    cli_args = parser.parse_args()
    main(cli_args)
