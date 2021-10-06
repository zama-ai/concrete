"""Tool to bypass the insane logic of semantic-release and generate changelogs we want"""

import argparse
import subprocess
import sys
from collections import deque

from git.repo import Repo
from semantic_release.changelog import markdown_changelog
from semantic_release.errors import UnknownCommitMessageStyleError
from semantic_release.settings import config, current_commit_parser
from semantic_release.vcs_helpers import get_repository_owner_and_name
from semver import VersionInfo


def log_msg(*args, file=sys.stderr, **kwargs):
    """Shortcut to print to sys.stderr."""
    print(*args, file=file, **kwargs)


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible."""
    return version_str[1:] if version_str.startswith("v") else version_str


def get_poetry_project_version() -> VersionInfo:
    """Run poetry version and get the project version"""
    command = ["poetry", "version"]
    poetry_version_output = subprocess.check_output(command, text=True)
    return version_string_to_version_info(poetry_version_output.split(" ")[1])


def raise_exception_or_print_warning(is_error: bool, message_body: str):
    """Raise an exception if is_error is true else print a warning to stderr"""
    msg_start = "Error" if is_error else "Warning"
    msg = f"{msg_start}: {message_body}"
    if is_error:
        raise RuntimeError(msg)
    log_msg(msg)


def version_string_to_version_info(version_string: str) -> VersionInfo:
    """Convert git tag to VersionInfo."""
    return VersionInfo.parse(strip_leading_v(version_string))


def generate_changelog(repo: Repo, from_commit_excluded: str, to_commit_included: str) -> dict:
    """Recreate the functionality from semantic release with the from and to commits.

    Args:
        repo (Repo): the gitpython Repo object representing your git repository
        from_commit_excluded (str): the commit after which we want to collect commit messages for
            the changelog
        to_commit_included (str): the last commit included in the collected commit messages for the
            changelog.

    Returns:
        dict: the same formatted dict as the generate_changelog from semantic-release
    """
    # Additional sections will be added as new types are encountered
    changes: dict = {"breaking": []}

    rev = f"{from_commit_excluded}...{to_commit_included}"

    for commit in repo.iter_commits(rev):
        hash_ = commit.hexsha
        commit_message = (
            commit.message.replace("\r\n", "\n")
            if isinstance(commit.message, str)
            else commit.message.replace(b"\r\n", b"\n")
        )
        try:
            message = current_commit_parser()(commit_message)
            if message.type not in changes:
                log_msg(f"Creating new changelog section for {message.type} ")
                changes[message.type] = []

            # Capitalize the first letter of the message, leaving others as they were
            # (using str.capitalize() would make the other letters lowercase)
            formatted_message = message.descriptions[0][0].upper() + message.descriptions[0][1:]
            if config.get("changelog_capitalize") is False:
                formatted_message = message.descriptions[0]

            # By default, feat(x): description shows up in changelog with the
            # scope bolded, like:
            #
            # * **x**: description
            if config.get("changelog_scope") and message.scope:
                formatted_message = f"**{message.scope}:** {formatted_message}"

            changes[message.type].append((hash_, formatted_message))

            if message.breaking_descriptions:
                # Copy breaking change descriptions into changelog
                for paragraph in message.breaking_descriptions:
                    changes["breaking"].append((hash_, paragraph))
            elif message.bump == 3:
                # Major, but no breaking descriptions, use commit subject instead
                changes["breaking"].append((hash_, message.descriptions[0]))

        except UnknownCommitMessageStyleError as err:
            log_msg(f"Ignoring UnknownCommitMessageStyleError: {err}")

    return changes


def main(args):
    """Entry point"""

    repo = Repo(args.repo_root)

    sha1_to_tags = {tag.commit.hexsha: tag for tag in repo.tags}

    to_commit = repo.commit(args.to_ref)
    log_msg(f"To commit: {to_commit}")

    to_tag = sha1_to_tags.get(to_commit.hexsha, None)
    if to_tag is None:
        raise_exception_or_print_warning(
            is_error=args.to_ref_must_have_tag,
            message_body=f"to-ref {args.to_ref} has no tag associated to it",
        )

    to_version = (
        get_poetry_project_version()
        if to_tag is None
        else version_string_to_version_info(to_tag.name)
    )
    log_msg(f"Project version {to_version} taken from tag: {to_tag is not None}")

    from_commit = None
    if args.from_ref is None:
        tags_by_name = {strip_leading_v(tag.name): tag for tag in repo.tags}
        all_release_version_infos = {
            version_info: tags_by_name[tag_name]
            for tag_name in tags_by_name
            if VersionInfo.isvalid(tag_name)
            and (version_info := VersionInfo.parse(tag_name))
            and version_info.prerelease is None
        }
        log_msg(f"All release versions {all_release_version_infos}")

        versions_before_project_version = [
            version_info for version_info in all_release_version_infos if version_info < to_version
        ]
        if len(versions_before_project_version) > 0:
            highest_version_before_current_version = max(versions_before_project_version)
            highest_version_tag = all_release_version_infos[highest_version_before_current_version]
            from_commit = highest_version_tag.commit
        else:
            # No versions before, get the initial commit reachable from to_commit
            # from https://stackoverflow.com/a/48232574
            last_element_extractor = deque(repo.iter_commits(to_commit), 1)
            from_commit = last_element_extractor.pop()
    else:
        from_commit = repo.commit(args.from_ref)

    log_msg(f"From commit: {from_commit}")
    ancestor_commit = repo.merge_base(to_commit, from_commit)
    assert len(ancestor_commit) == 1
    ancestor_commit = ancestor_commit[0]
    log_msg(f"Common ancestor: {ancestor_commit}")

    if ancestor_commit != from_commit:
        do_not_change_from_ref = args.do_not_change_from_ref and args.from_ref is not None
        raise_exception_or_print_warning(
            is_error=do_not_change_from_ref,
            message_body=(
                f"the ancestor {ancestor_commit} for {from_commit} and {to_commit} "
                f"is not the same commit as the commit for '--from-ref' {from_commit}."
            ),
        )

    ancestor_tag = sha1_to_tags.get(ancestor_commit.hexsha, None)
    if ancestor_tag is None:
        raise_exception_or_print_warning(
            is_error=args.ancestor_must_have_tag,
            message_body=(
                f"the ancestor {ancestor_commit} for " f"{from_commit} and {to_commit} has no tag"
            ),
        )

    ancestor_version_str = (
        None if ancestor_tag is None else str(version_string_to_version_info(ancestor_tag.name))
    )

    log_msg(
        f"Collecting commits from \n{ancestor_commit} "
        f"(tag: {ancestor_tag} - parsed version "
        f"{str(ancestor_version_str)}) to \n{to_commit} "
        f"(tag: {to_tag} - parsed version {str(to_version)})"
    )

    log_dict = generate_changelog(repo, ancestor_commit.hexsha, to_commit.hexsha)

    owner, name = get_repository_owner_and_name()
    md_changelog = markdown_changelog(
        owner,
        name,
        str(to_version),
        log_dict,
        header=True,
        previous_version=ancestor_version_str,
    )

    print(md_changelog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Changelog helper", allow_abbrev=False)

    parser.add_argument("--repo-root", type=str, default=".", help="Path to the repo root")
    parser.add_argument(
        "--to-ref",
        type=str,
        help="Specify the git ref-like string (sha1, tag, HEAD~, etc.) that will mark the LAST "
        "included commit of the changelog. If this is not specified, the current project version "
        "will be used to create a changelog with the current commit as last commit.",
    )
    parser.add_argument(
        "--from-ref",
        type=str,
        help="Specify the git ref-like string (sha1, tag, HEAD~, etc.) that will mark the commit "
        "BEFORE the first included commit of the changelog. If this is not specified, the most "
        "recent actual release tag (no pre-releases) before the '--to-ref' argument will be used. "
        "If the tagged commit is not an ancestor of '--to-ref' then the most recent common ancestor"
        "(git merge-base) will be used unless '--do-not-change-from-ref' is specified.",
    )
    parser.add_argument(
        "--ancestor-must-have-tag",
        action="store_true",
        help="Set if the used ancestor must have a tag associated to it.",
    )
    parser.add_argument(
        "--to-ref-must-have-tag",
        action="store_true",
        help="Set if '--to-ref' must have a tag associated to it.",
    )
    parser.add_argument(
        "--do-not-change-from-ref",
        action="store_true",
        help="Specify to prevent selecting a different '--from-ref' than the one specified in cli. "
        "Will raise an exception if '--from-ref' is not a suitable ancestor for '--to-ref' and "
        "would otherwise use the most recent common ancestor (git merge-base) as '--from-ref'.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
