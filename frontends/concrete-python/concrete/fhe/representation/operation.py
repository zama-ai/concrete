"""
Declaration of `Operation` enum.
"""

from enum import Enum


class Operation(Enum):
    """
    Operation enum, to distinguish nodes within a computation graph.
    """

    # pylint: disable=invalid-name

    Constant = "constant"
    Generic = "generic"
    Input = "input"

    # pylint: enable=invalid-name
