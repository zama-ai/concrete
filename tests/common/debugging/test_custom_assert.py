"""Test custom assert functions."""
import pytest

from concrete.common.debugging.custom_assert import assert_false, assert_not_reached, assert_true


def test_assert_not_functions():
    """Test custom assert functions"""
    assert_true(True, "one check")
    assert_false(False, "another check")

    with pytest.raises(AssertionError) as excinfo:
        assert_not_reached("yet another one")

    assert "yet another one" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        assert_true(False, "one failing check")

    assert "one failing check" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        assert_false(True, "another failing check")

    assert "another failing check" in str(excinfo.value)
