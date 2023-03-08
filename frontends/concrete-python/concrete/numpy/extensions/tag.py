"""
Declaration of `tag` context manager, to allow tagging certain nodes.
"""

import threading
from contextlib import contextmanager

tag_context = threading.local()
tag_context.stack = []


@contextmanager
def tag(name: str):
    """
    Introduce a new tag to the tag stack.

    Can be nested, and the resulting tag will be `tag1.tag2`.
    """

    tag_context.stack.append(name)
    try:
        yield
    finally:
        tag_context.stack.pop()
