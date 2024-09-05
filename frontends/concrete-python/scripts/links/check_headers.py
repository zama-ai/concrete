"""Check that headers linked do indeed exist in target markdown files"""

import os
from pathlib import Path

import mistletoe  # Markdown to AST


def ast_iterator(root):
    """Iterate on all children of a node

    Args:
        root: Base node of the ast on which to iterate

    Yields:
        Unknown
    """
    nodes = [root]
    while nodes:
        current_node = nodes.pop(0)
        yield current_node
        if hasattr(current_node, "children") and current_node.children is not None:
            nodes += current_node.children


def is_web_link(target: str) -> bool:
    """Check if the link points to http or https

    Arguments:
        target (str): string to check

    Returns:
        bool
    """
    return target.startswith("http://") or target.startswith("https://")


def is_mailto_link(target: str) -> bool:
    """Check if the link points to a mailto

    Arguments:
        target (str): string to check

    Returns:
        bool
    """
    return "mailto:" in target


def contains_header(ast, header) -> bool:
    """Check if the ast-represented document contains the header

    Arguments:
        ast: ast to check if contains the header
        header: header to check

    Returns:
        bool
    """

    for node in ast_iterator(ast):
        if isinstance(node, mistletoe.block_token.Heading):

            # Heading is list of tokens
            file_header = " ".join(
                [
                    str(elt.content)
                    for elt in ast_iterator(node)
                    if isinstance(elt, mistletoe.span_token.RawText)
                ]
            )
            # Needed to escape some characters
            # We might want to check with the markdown spec
            file_header = (
                "-".join(file_header.split())
                .replace("<kbd>", "")
                .replace("</kbd>", "")
                .replace(".", "")
                .replace("!", "")
                .replace("?", "")
                .replace("/", "")
                .replace("&", "")
                .lower()
            )

            if header == file_header:
                return True
    return False


# pylint: disable-next=too-many-branches
def main():
    """Main function that checks for all files that the header exists in the linked file

    Raises:
        ValueError: if a missing link is found
    """
    # Get files
    current_path = Path(os.getcwd())
    markdown_files = [
        path
        for path in current_path.rglob("*")
        if str(path).endswith(".md")
        if ".venv" not in set(map(str, path.parts))
    ]

    # We don't want to checks links from docs/_build
    markdown_files = [
        path for path in markdown_files if "../../docs/_build/" not in str(path.resolve())
    ]

    # Collect ASTs
    asts = {}
    for file_path in markdown_files:
        with open(file_path, mode="r", encoding="utf-8") as file:
            asts[file_path.resolve()] = mistletoe.Document(file)

    # Check links
    errors = []
    # For each document we check all links
    for document_path, document in asts.items():
        for node in ast_iterator(document):
            if isinstance(node, (mistletoe.span_token.Link)):
                # We don't verify external links
                if is_web_link(node.target):
                    continue
                if is_mailto_link(node.target):
                    continue

                # Split file and header
                splitted = node.target.split("#")
                if len(splitted) == 2:
                    file_path = Path(splitted[0]) if splitted[0] else Path(document_path)
                    header = splitted[1]
                elif len(splitted) == 1:
                    file_path, header = Path(splitted[0]), None
                else:
                    error_message = f"Could not parse {node.target}"
                    raise ValueError(error_message)

                # Get absolute path
                abs_file_path = (document_path.parent / file_path).resolve()

                # Check file exists
                if not abs_file_path.exists():
                    errors.append(f"Link to {abs_file_path} from {document_path} does not exist")
                    continue

                # Check header is contained
                if header:
                    if abs_file_path not in asts:
                        errors.append(
                            f"{abs_file_path} for {node.target} was not "
                            f"parsed into AST (from {document_path})"
                        )
                        continue
                    if header and not contains_header(asts[abs_file_path], header):
                        errors.append(
                            f"{header} from {document_path} does not exist in {abs_file_path}"
                        )
                        continue
    if errors:
        raise ValueError(
            "Errors:\n" + "\n".join([f"- {error}" for error in errors]) + f"\n{len(errors)} errors"
        )


if __name__ == "__main__":
    main()
