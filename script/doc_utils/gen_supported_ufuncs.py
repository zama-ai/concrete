"""Update list of supported functions in the doc."""
import argparse

from concrete.numpy import tracing


def main(file_to_update):
    """Update list of supported functions in file_to_update"""
    supported_unary_ufunc = sorted(
        f.__name__ for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 1
    )
    supported_binary_ufunc = sorted(
        f.__name__ for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 2
    )

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
            newlines.append("List of supported unary functions:\n")

            newlines.extend(f"- {f}\n" for f in supported_unary_ufunc)

            newlines.append("\n")
            newlines.append("## Binary operations\n")
            newlines.append("\n")

            newlines.append(
                "List of supported binary functions if one of the "
                "two operators is a constant scalar:\n"
            )

            newlines.extend(f"- {f}\n" for f in supported_binary_ufunc)

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
