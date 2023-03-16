"""Jupyter Notebook Sanitization."""

import argparse
import json
from pathlib import Path


def main():
    """Sanitize or check sanitization of Jupyter Notebooks."""

    parser = argparse.ArgumentParser(description="Sanitizer for Jupyter Notebooks")

    parser.add_argument("base", type=str, help="directory which contains the notebooks")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    args = parser.parse_args()

    base = Path(args.base)
    notebooks = base.glob("**/*.ipynb")

    for notebook in notebooks:
        path = str(notebook)
        if "_build" in path or ".ipynb_checkpoints" in path:
            continue

        with open(notebook, "r", encoding="utf-8") as f:
            content = json.load(f)

        if args.check:
            try:
                metadata = content["metadata"]
                assert len(metadata) == 1
                assert "execution" in metadata

                execution = metadata["execution"]
                assert len(execution) == 1
                assert "timeout" in execution

                timeout = execution["timeout"]
                assert timeout == 10800  # 3 hours
            except Exception:
                print("Notebooks are not sanitized. Please run `make conformance`.")
                raise
        else:
            content["metadata"] = {
                "execution": {
                    "timeout": 10800,  # 3 hours
                }
            }
            with open(notebook, "w", newline="\n", encoding="utf-8") as f:
                json.dump(content, f, indent=1, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    main()
