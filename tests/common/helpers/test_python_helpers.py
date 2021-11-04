"""Test file for common python helpers"""

from concrete.common.helpers.python_helpers import catch


def test_catch_failure():
    """Test case for when the function called with catch raises an exception."""

    def f_fail():
        return 1 / 0

    assert catch(f_fail) is None


def test_catch():
    """Test case for catch"""

    def f(*args, **kwargs):
        return *args, dict(**kwargs)

    assert catch(f, (1, 2, 3,), **{"one": 1, "two": 2, "three": 3}) == (
        (1, 2, 3),
        {"one": 1, "two": 2, "three": 3},
    )
