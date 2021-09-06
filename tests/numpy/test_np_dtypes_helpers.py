"""Test file for numpy dtype helpers"""

import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.numpy.np_dtypes_helpers import (
    convert_base_data_type_to_numpy_dtype,
    convert_numpy_dtype_to_base_data_type,
)


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
def test_convert_numpy_dtype_to_base_data_type(numpy_dtype, expected_common_type):
    """Test function for convert_numpy_dtype_to_base_data_type"""
    assert convert_numpy_dtype_to_base_data_type(numpy_dtype) == expected_common_type


@pytest.mark.parametrize(
    "common_dtype,expected_numpy_dtype",
    [
        pytest.param(Integer(7, is_signed=False), numpy.uint32),
        pytest.param(Integer(7, is_signed=True), numpy.int32),
        pytest.param(Integer(32, is_signed=True), numpy.int32),
        pytest.param(Integer(64, is_signed=True), numpy.int64),
        pytest.param(Integer(32, is_signed=False), numpy.uint32),
        pytest.param(Integer(64, is_signed=False), numpy.uint64),
        pytest.param(Float(32), numpy.float32),
        pytest.param(Float(64), numpy.float64),
        pytest.param(
            Integer(128, is_signed=True),
            None,
            marks=pytest.mark.xfail(strict=True, raises=NotImplementedError),
        ),
    ],
)
def test_convert_common_dtype_to_numpy_dtype(common_dtype, expected_numpy_dtype):
    """Test function for convert_common_dtype_to_numpy_dtype"""
    assert expected_numpy_dtype == convert_base_data_type_to_numpy_dtype(common_dtype)
