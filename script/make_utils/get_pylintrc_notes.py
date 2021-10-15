"""File to get pylintrc notes"""

import argparse
import configparser
from pathlib import Path


def main(args):
    """Entry point"""

    pylintrc_file_path = Path(args.pylintrc_path).resolve()
    config = configparser.ConfigParser()
    config.read(pylintrc_file_path)
    notes = sorted(map(lambda x: x.strip(), config["MISCELLANEOUS"]["notes"].split(",")))
    # Make sure we at least have todo in there without writing it otherwise we'll match
    notes.append("TO" + "DO")
    notes_for_grep_search = r"\|".join(notes)
    print(notes_for_grep_search)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse pylintrc notes", allow_abbrev=False)

    parser.add_argument(
        "--pylintrc-path", type=str, required=True, help="Path to pylintrc ini config"
    )

    cli_args = parser.parse_args()

    main(cli_args)
