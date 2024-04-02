"""
Tests of utilities related to the entire project.
"""

import pytest

from concrete.fhe.internal.utils import assert_that, unreachable
from concrete.fhe import _is_cpu_compatible


def test_assert_that():
    """
    Test `assert_that` function.
    """

    with pytest.raises(AssertionError) as excinfo:
        assert_that(2 + 2 == 3, "no")

    assert str(excinfo.value) == "no"


def test_unreachable():
    """
    Test `unreachable` function.
    """

    with pytest.raises(RuntimeError) as excinfo:
        unreachable()

    assert str(excinfo.value) == "Entered unreachable code"


def test_cpu_compatibility():
    """
    Test `_is_cpu_compatible` function.
    """
    assert _is_cpu_compatible() == True
