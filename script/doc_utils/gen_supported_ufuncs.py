"""Update list of supported functions in the doc."""

import argparse

from concrete.numpy.tracing import Tracer


def main(file_to_update):
    """Update list of supported functions in file_to_update"""
    supported_func = sorted(f.__name__ for f in Tracer.SUPPORTED_NUMPY_OPERATORS)

    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    newlines = []
    keep_line = True

    for line in lines:
        if line.startswith(
            "<!--- gen_supported_ufuncs.py: inject supported operations [BEGIN] -->"
        ):
            keep_line = False
            newlines.append(line)
            newlines.append(
                "<!--- do not edit, auto generated part by "
                "`python3 gen_supported_ufuncs.py` in docker -->\n"
            )
        elif line.startswith(
            "<!--- do not edit, auto generated part by "
            "`python3 gen_supported_ufuncs.py` in docker -->"
        ):
            pass
        elif line.startswith(
            "<!--- gen_supported_ufuncs.py: inject supported operations [END] -->"
        ):
            keep_line = True

            # Inject the supported functions
            newlines.append("List of supported functions:\n")

            newlines.extend(f"- {f}\n" for f in supported_func)

            newlines.append(line)
        else:
            assert "gen_supported_ufuncs.py" not in line, (
                f"Error: not expected to have 'gen_supported_ufuncs.py' at line {line} "
                f"of {file_to_update}"
            )

            if keep_line:
                newlines.append(line)

    if args.check:

        with open(file_to_update, "r", encoding="utf-8") as file:
            oldlines = file.readlines()

        assert (
            oldlines == newlines
        ), "List of supported functions is not up to date. Please run `make supported_functions`."

    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(newlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of supported functions in the doc")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    parser.add_argument("file_to_update", type=str, help=".md file to update")
    args = parser.parse_args()
    main(args.file_to_update)
