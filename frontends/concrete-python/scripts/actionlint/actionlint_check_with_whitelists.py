""" Check an actionlint log against some whitelists """

import sys
from typing import Set

# Exact lines which are whitelisted
whitelisted_lines: Set[str] = set()

# Pattern which are whitelisted
whitelisted_pattern: Set[str] = {
}


def main():
    """Do the test

    Raises:
        ValueError: if non whitelisted error occurred
    """
    status = 0
    bad_lines = []
    for line in sys.stdin:
        if line in whitelisted_lines:
            continue

        is_bad_line = True

        for pattern in whitelisted_pattern:
            if pattern in line:
                is_bad_line = False
                break

        if is_bad_line:
            print("->", line)
            status = 1
            bad_lines.append(line)

    if status:
        errors = "\n------\n".join(bad_lines)
        raise ValueError("Some non whitelisted errors, look at full log file:" f"{errors}")


if __name__ == "__main__":
    main()
