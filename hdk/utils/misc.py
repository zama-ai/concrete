"""Misc. utils for hdk"""
from typing import Iterator


def get_unique_id() -> int:
    """Function to get a unique ID"""

    if not hasattr(get_unique_id, "generator"):

        def generator() -> Iterator[int]:
            current_id = 0
            while True:
                yield current_id
                current_id += 1

        setattr(get_unique_id, "generator", generator())

    return next(getattr(get_unique_id, "generator"))
