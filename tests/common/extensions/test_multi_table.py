"""Test file for direct multi table lookups"""

import random

import numpy
import pytest

from concrete.common.data_types.integers import Integer
from concrete.common.extensions.multi_table import MultiLookupTable
from concrete.common.extensions.table import LookupTable

table_2b_to_2b = LookupTable([1, 2, 0, 3])
table_2b_to_1b = LookupTable([1, 0, 0, 1])
table_2b_to_3b = LookupTable([5, 2, 7, 0])

table_3b_to_2b = LookupTable([1, 2, 0, 3, 0, 3, 1, 2])
table_3b_to_1b = LookupTable([1, 0, 0, 1, 1, 1, 1, 0])
table_3b_to_3b = LookupTable([5, 2, 7, 0, 4, 1, 6, 2])

tables_2b = [table_2b_to_1b, table_2b_to_2b, table_2b_to_3b]
tables_3b = [table_3b_to_1b, table_3b_to_2b, table_3b_to_3b]


def test_multi_lookup_table_creation_and_indexing():
    """Test function for creating and indexing multi lookup tables"""
    tables = [
        [tables_2b[random.randint(0, 2)], tables_2b[random.randint(0, 2)]],
        [tables_2b[random.randint(0, 2)], tables_2b[random.randint(0, 2)]],
        [tables_2b[random.randint(0, 2)], tables_2b[random.randint(0, 2)]],
    ]
    multitable = MultiLookupTable(tables)

    assert multitable.input_shape == (3, 2)

    assert isinstance(multitable.output_dtype, Integer)
    assert multitable.output_dtype.bit_width <= 3

    index = numpy.random.randint(0, 2 ** 2, size=multitable.input_shape).tolist()
    result = multitable[index]

    for i in range(3):
        for j in range(2):
            assert result[i][j] == multitable.tables[i][j][index[i][j]], f"i={i}, j={j}"


@pytest.mark.parametrize(
    "tables,match",
    [
        pytest.param(
            [
                [],
                [table_2b_to_2b, table_2b_to_3b],
            ],
            "MultiLookupTable cannot have an empty array within it",
        ),
        pytest.param(
            [
                [table_2b_to_1b, 42.0],
                [table_2b_to_2b, table_2b_to_3b],
            ],
            "MultiLookupTable should have been made out of LookupTables "
            "but it had an object of type float within it",
        ),
        pytest.param(
            [
                [table_2b_to_2b],
                [table_2b_to_2b, table_2b_to_3b],
                [table_2b_to_2b, table_2b_to_1b],
            ],
            "MultiLookupTable should have the shape (3, 1) but it does not "
            "(an array on dimension 1 has the size 2 but its size should have been 1 "
            "as the expected shape is (3, 1))",
        ),
        pytest.param(
            [
                [table_2b_to_2b, table_3b_to_3b],
                [table_2b_to_2b, table_3b_to_1b],
            ],
            "LookupTables within a MultiLookupTable should have the same size but they do not "
            "(there was a table with the size of 4 and another with the size of 8)",
        ),
    ],
)
def test_multi_lookup_table_creation_failure(tables, match):
    """Test function for failing to create multi lookup tables"""

    with pytest.raises(ValueError) as excinfo:
        MultiLookupTable(tables)

    assert str(excinfo.value) == match


@pytest.mark.parametrize(
    "tables,index,match",
    [
        pytest.param(
            [
                [table_2b_to_2b, table_2b_to_1b, table_2b_to_3b],
                [table_2b_to_1b, table_2b_to_2b, table_2b_to_3b],
            ],
            [
                [1, 2],
                [3, 0],
            ],
            "Multiple Lookup Table of shape (2, 3) cannot be looked up with [[1, 2], [3, 0]] "
            "(you should check your inputset)",
        ),
    ],
)
def test_multi_lookup_table_indexing_failure(tables, index, match):
    """Test function for failing to index multi lookup tables"""

    table = MultiLookupTable(tables)

    with pytest.raises(ValueError) as excinfo:
        table.__getitem__(index)

    assert str(excinfo.value) == match
