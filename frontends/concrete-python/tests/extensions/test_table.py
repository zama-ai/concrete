"""
Tests of 'LookupTable' extension.
"""

import pytest

from concrete.fhe import LookupTable


@pytest.mark.parametrize(
    "table, expected_result",
    [
        pytest.param(
            LookupTable([1, 2, 3]),
            "[1, 2, 3]",
        ),
        pytest.param(
            LookupTable([LookupTable([1, 2, 3]), LookupTable([4, 5, 6])]),
            "[[1, 2, 3], [4, 5, 6]]",
        ),
    ],
)
def test_lookup_table_repr(table, expected_result):
    """
    Test `__repr__` method of `LookupTable` class.
    """

    assert repr(table) == expected_result
