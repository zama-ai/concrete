"""Test file for hnumpy numpy dtype helpers"""

import numpy
import pytest

from hdk.common.data_types.floats import Float
from hdk.common.data_types.integers import Integer
from hdk.hnumpy.np_dtypes_helpers import convert_numpy_dtype_to_common_dtype


@pytest.mark.parametrize(
    "numpy_dtype,expected_common_type",
    [
        pytest.param(numpy.int32, Integer(32, is_signed=True)),
        pytest.param("int32", Integer(32, is_signed=True)),
        pytest.param(numpy.int64, Integer(64, is_signed=True)),
        pytest.param("int64", Integer(64, is_signed=True)),
        pytest.param(numpy.uint32, Integer(32, is_signed=False)),
        pytest.param("uint32", Integer(32, is_signed=False)),
        pytest.param(numpy.uint64, Integer(64, is_signed=False)),
        pytest.param("uint64", Integer(64, is_signed=False)),
        pytest.param(numpy.float32, Float(32)),
        pytest.param("float32", Float(32)),
        pytest.param(numpy.float64, Float(64)),
        pytest.param("float64", Float(64)),
        pytest.param("complex64", None, marks=pytest.mark.xfail(strict=True, raises=ValueError)),
    ],
)
def test_convert_numpy_dtype_to_common_dtype(numpy_dtype, expected_common_type):
    """Test function for convert_numpy_dtype_to_common_dtype"""
    assert convert_numpy_dtype_to_common_dtype(numpy_dtype) == expected_common_type
