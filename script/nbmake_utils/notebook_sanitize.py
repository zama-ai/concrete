import json
import sys

from pathlib import Path


def main():
    path_to_glob = Path(sys.argv[1])
    notebooks = path_to_glob.glob("*.ipynb")

    for notebook_file in notebooks:
        with open(notebook_file, "r") as f:
            notebook_dict = json.load(f)
            notebook_dict["metadata"] = {}

        with open(notebook_file, "w", newline="\n") as f:
            json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
