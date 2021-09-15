"""Sanitizer for Jupyter notebooks."""
import argparse
import json
from pathlib import Path


def main():
    """Sanitize"""
    parser = argparse.ArgumentParser(description="Sanitizer for Jupyter Notebooks")

    parser.add_argument("base", type=str, help="directory which contains the notebooks")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    args = parser.parse_args()

    base = Path(args.base)
    notebooks = base.glob("*.ipynb")

    for notebook in notebooks:
        with open(notebook, "r", encoding="utf-8") as f:
            content = json.load(f)

        if args.check:
            if len(content["metadata"]) != 0:
                print("Notebooks are not sanitized. Please run `make conformance`.")
                raise ValueError
        else:
            content["metadata"] = {}
            with open(notebook, "w", newline="\n", encoding="utf-8") as f:
                json.dump(content, f, indent=1, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    main()
