"""Tool to manage the versions.html file at the root of our docs sites."""

import argparse
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from semver import VersionInfo

VERSIONS_LIST_ID = "versions-list"


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible."""
    return version_str[1:] if version_str.startswith("v") else version_str


def create_list_element(soup: BeautifulSoup, contents: Tag) -> Tag:
    """Create a list element for links.

    Args:
        soup (BeautifulSoup): The soup to use to create the tag.

    Returns:
        Tag: tag containing <li class="toctree-l1"></li>.
    """
    new_list_element = soup.new_tag("li", **{"class": "toctree-l1"})
    new_list_element.contents.append(contents)
    return new_list_element


def create_link_tag_set_string(soup: BeautifulSoup, version_string: str) -> Tag:
    """Create a link tag on the given soup to version specified by version_string.

    Args:
        soup (BeautifulSoup): The soup to use to create the tag.
        version_string (str): The version string to use.

    Returns:
        Tag: tag containing <a class="reference internal" href="0.1.0/">{version_string}</a>.
    """
    new_tag = soup.new_tag(
        "a",
        **{
            "href": f"{version_string}/",
            "class": "reference internal",
        },
    )

    new_tag.string = version_string
    return new_tag


def main(args):
    """Entry point."""

    invalid_versions = [
        version
        for version in args.add_versions
        if not VersionInfo.isvalid(strip_leading_v(version))
    ]
    if len(invalid_versions) > 0:
        raise RuntimeError(f"Found invalid versions: {invalid_versions}")

    version_html = None
    version_html_file_path = Path(args.versions_html_file).resolve()
    with open(version_html_file_path, "r", encoding="utf-8") as f:
        version_html = BeautifulSoup(f, "html.parser")

    if version_html is None:
        raise RuntimeError(f"An error occured while trying to load {str(version_html_file_path)}")

    print(version_html)

    version_list = version_html.find(id=VERSIONS_LIST_ID)
    if version_list is None or version_list.name != "ul":
        raise RuntimeError(f"Could not find <ul> tag with id {VERSIONS_LIST_ID}")

    non_semver_versions = {}
    semver_versions = {}
    for list_entry in version_list.find_all("li"):
        version_tags = []
        version_is_valid_semver = False
        for potential_version_tag in list_entry.contents:
            if not isinstance(potential_version_tag, Tag):
                continue
            version_is_valid_semver = VersionInfo.isvalid(
                strip_leading_v(potential_version_tag.string)
            )
            version_tags.append(potential_version_tag.string)

        num_version_tags = len(version_tags)
        assert num_version_tags == 1, f"Can only have 1 version tag, got {num_version_tags}"

        version_tag = version_tags[0]

        if version_is_valid_semver:
            semver_versions[version_tag.string] = list_entry
        else:
            non_semver_versions[version_tag.string] = list_entry

    parsed_versions = [VersionInfo.parse(version) for version in args.add_versions]

    versions_already_in_html = set(parsed_versions).intersection(semver_versions.keys())
    if len(versions_already_in_html) > 0:
        raise RuntimeError(
            "The following versions are already in the html: "
            f"{', '.join(str(ver) for ver in sorted(versions_already_in_html))}"
        )

    semver_versions.update(
        (
            parsed_version,
            create_list_element(
                version_html, create_link_tag_set_string(version_html, str(parsed_version))
            ),
        )
        for parsed_version in parsed_versions
    )

    version_list.contents = []
    for sorted_non_semver_version in sorted(non_semver_versions.keys()):
        version_list.contents.append(non_semver_versions[sorted_non_semver_version])

    # We want the most recent versions at the top
    for sorted_semver_version in sorted(semver_versions.keys(), reverse=True):
        version_list.contents.append(semver_versions[sorted_semver_version])

    pretty_output = version_html.prettify()
    print(pretty_output)

    output_html_path = Path(args.output_html).resolve()
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(pretty_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("versions.html generator", allow_abbrev=False)

    parser.add_argument(
        "--add-versions",
        type=str,
        required=True,
        nargs="+",
        help="A list of versions to add to versions.html. "
        "The links will be sorted by versions with stable/main as the first entry. "
        "The link will point to '$VERSION/' and will have text '$VERSION'.",
    )
    parser.add_argument(
        "--versions-html-file",
        type=str,
        required=True,
        help="Path to the versions.html to update. "
        'It must have a <li> tag with id="versions-list".',
    )
    parser.add_argument("--output-html", type=str, required=True, help="Output file path.")

    cli_args = parser.parse_args()
    main(cli_args)
