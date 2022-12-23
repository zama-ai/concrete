"""
Declaration of various functions and constants related to the entire project.
"""


def assert_that(condition: bool, message: str = ""):
    """
    Assert a condition.

    Args:
        condition (bool):
            condition to assert

        message (str):
            message to give to `AssertionError` if the condition does not hold

    Raises:
        AssertionError:
            if the condition does not hold
    """

    if not condition:
        raise AssertionError(message)


def unreachable():
    """
    Raise a RuntimeError to indicate unreachable code is entered.
    """

    message = "Entered unreachable code"
    raise RuntimeError(message)
