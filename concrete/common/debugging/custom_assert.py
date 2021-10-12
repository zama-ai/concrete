"""Provide some variants of assert."""


def _custom_assert(condition: bool, on_error_msg: str = "") -> None:
    """Provide a custom assert which is kept even if the optimized python mode is used.

    See https://docs.python.org/3/reference/simple_stmts.html#assert for the documentation
    on the classical assert function

    Args:
        condition(bool): the condition. If False, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error

    """

    if not condition:
        raise AssertionError(on_error_msg)


def assert_true(condition: bool, on_error_msg: str = ""):
    """Provide a custom assert to check that the condition is True.

    Args:
        condition(bool): the condition. If False, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error

    """
    return _custom_assert(condition, on_error_msg)


def assert_false(condition: bool, on_error_msg: str = ""):
    """Provide a custom assert to check that the condition is False.

    Args:
        condition(bool): the condition. If True, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error

    """
    return _custom_assert(not condition, on_error_msg)


def assert_not_reached(on_error_msg: str):
    """Provide a custom assert to check that a piece of code is never reached.

    Args:
        on_error_msg(str): message for precising the error

    """
    return _custom_assert(False, on_error_msg)
