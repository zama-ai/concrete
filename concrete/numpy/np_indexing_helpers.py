"""Helpers for indexing with numpy values functionality."""

from typing import Any

import numpy


def should_sanitize(indexing_element: Any) -> bool:
    """Decide whether to sanitize an indexing element or not.

    Sanitizing in this context means converting supported numpy values into python values.

    Args:
        indexing_element (Any): the indexing element to decide sanitization.

    Returns:
        bool: True if indexing element should be sanitized otherwise False.
    """

    return isinstance(indexing_element, numpy.integer) or (
        isinstance(indexing_element, numpy.ndarray)
        and issubclass(indexing_element.dtype.type, numpy.integer)
        and indexing_element.shape == ()
    )


def process_indexing_element(indexing_element: Any) -> Any:
    """Process an indexing element.

    Processing in this context means converting supported numpy values into python values.
    (if they are decided to be sanitized)

    Args:
        indexing_element (Any): the indexing element to sanitize.

    Returns:
        Any: the sanitized indexing element.
    """

    if isinstance(indexing_element, slice):

        start = indexing_element.start
        if should_sanitize(start):
            start = int(start)

        stop = indexing_element.stop
        if should_sanitize(stop):
            stop = int(stop)

        step = indexing_element.step
        if should_sanitize(step):
            step = int(step)

        indexing_element = slice(start, stop, step)

    elif should_sanitize(indexing_element):
        indexing_element = int(indexing_element)

    return indexing_element
