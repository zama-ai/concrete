"""Helper script to be able to test python code in markdown files."""

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List

PYTHON_BLOCK_HINTS = ["py", "python", "python3"]
BLOCK_STARTS = tuple(f"```{hint}" for hint in PYTHON_BLOCK_HINTS)
BLOCK_END = "```"
DIRECTIVE_COMMENT_PATTERN = "<!--python-test:(.*)-->"
SKIP_DIRECTIVE = "skip"
CONT_DIRECTIVE = "cont"


def get_code_blocks_for_file(md_file: Path) -> Dict[int, List[str]]:
    """Function to process an md file and test the python code in it.

    Args:
        md_file (Path): The path to the md file to convert and test.

    Raises:
        SyntaxError: If EOF is reached before a code block is closed.
        SyntaxError: If a block is not closed and a new python block is opened.

    Returns:
        Dict[int, List[str]]: A dict containing the code blocks of the file.
    """
    file_content = None

    python_code_blocks: Dict[int, List[str]] = {}

    def get_code_block_container(line_idx):
        block_idx = line_idx
        python_code_blocks[block_idx] = []
        return python_code_blocks[block_idx]

    with open(md_file, encoding="utf-8") as f:
        file_content = f.readlines()

    file_content_iterator = iter(enumerate(file_content, 1))
    python_block_continues = False
    skip_next_python_block = False

    for line_idx, line in file_content_iterator:
        if line.startswith(BLOCK_STARTS):
            if skip_next_python_block:
                skip_next_python_block = False
                continue
            if not python_block_continues:
                current_python_code = get_code_block_container(line_idx)
            while True:
                line_idx, line = next(file_content_iterator)
                if line == "":
                    # Reached EOF
                    raise SyntaxError(
                        "Reached EOF before finding the end of the current python block in "
                        f"{str(md_file)}"
                    )

                if line.strip() == BLOCK_END:
                    break

                if line.startswith(BLOCK_STARTS):
                    raise SyntaxError(
                        f"Error at line {line_idx} in file {str(md_file)}, "
                        "python block was opened before the previous one was "
                        "closed (missing ``` ?)"
                    )
                current_python_code.append(line)
            python_block_continues = False
        else:
            match = re.match(DIRECTIVE_COMMENT_PATTERN, line)
            if match is not None:
                directive = match.group(1)
                if directive == SKIP_DIRECTIVE:
                    skip_next_python_block = True
                elif directive == CONT_DIRECTIVE:
                    python_block_continues = True

                python_block_continues = python_block_continues and not skip_next_python_block

    return python_code_blocks


def main(args):
    """The actual processing."""
    md_dir_path = Path(args.md_dir)
    md_files = sorted(md_dir_path.glob("**/*.md"))

    code_blocks_per_file: Dict[str, Dict[int, List[str]]] = {}

    err_msg = ""

    for md_file in md_files:
        md_file = md_file.resolve().absolute()
        md_file_str = str(md_file)
        # pylint: disable=broad-except
        try:
            code_blocks_per_file[md_file_str] = get_code_blocks_for_file(md_file)
        except Exception:
            err_msg += f"Error while converting {md_file_str}"
            err_msg += traceback.format_exc() + "\n"
        # pylint: enable=broad-except

    for md_file_str, code_blocks in code_blocks_per_file.items():
        for line_idx, python_code in code_blocks.items():
            # pylint: disable=broad-except,exec-used
            try:
                print(f"Testing block starting line #{line_idx} from {md_file_str}")
                python_code = "".join(python_code)
                compiled_code = compile(python_code, filename=md_file_str, mode="exec")
                exec(compiled_code, {"__MODULE__": "__main__"})
                print("Success")
            except Exception:
                print("Failed")
                err_msg += (
                    f"Error while testing block starting line #{line_idx} from {md_file_str}:\n"
                )
                err_msg += f"```\n{python_code}```\n"
                err_msg += traceback.format_exc() + "\n"
            # pylint: enable=broad-except,exec-used

    if err_msg != "":
        print(err_msg)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts md python blocks to python files", allow_abbrev=False
    )
    parser.add_argument(
        "--md_dir", type=str, help="The path to the dir containing md files to convert."
    )

    cli_args = parser.parse_args()

    main(cli_args)
