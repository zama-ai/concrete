"""Test file for numpy dtype helpers"""

import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.numpy.np_dtypes_helpers import (
    convert_base_data_type_to_numpy_dtype,
    convert_numpy_dtype_to_base_data_type,
    get_base_value_for_numpy_or_python_constant_data,
    get_constructor_for_numpy_or_python_constant_data,
)


@pytest.mark.parametrize(
    "numpy_dtype,expected_common_type",
    [
        pytest.param(numpy.int8, Integer(8, is_signed=True)),
        pytest.param("int8", Integer(8, is_signed=True)),
        pytest.param(numpy.int16, Integer(16, is_signed=True)),
        pytest.param("int16", Integer(16, is_signed=True)),
        pytest.param(numpy.int32, Integer(32, is_signed=True)),
        pytest.param("int32", Integer(32, is_signed=True)),
        pytest.param(numpy.int64, Integer(64, is_signed=True)),
        pytest.param("int64", Integer(64, is_signed=True)),
        pytest.param(numpy.uint8, Integer(8, is_signed=False)),
        pytest.param("uint8", Integer(8, is_signed=False)),
        pytest.param(numpy.uint16, Integer(16, is_signed=False)),
        pytest.param("uint16", Integer(16, is_signed=False)),
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


@pytest.mark.parametrize(
    "constant_data,expected_constructor",
    [
        (10, int),
        (42.0, float),
        (numpy.int32(10), numpy.int32),
    ],
)
def test_get_constructor_for_numpy_or_python_constant_data(constant_data, expected_constructor):
    """Test function for get_constructor_for_numpy_or_python_constant_data"""

    assert expected_constructor == get_constructor_for_numpy_or_python_constant_data(constant_data)


def test_get_constructor_for_numpy_arrays(test_helpers):
    """Test function for get_constructor_for_numpy_or_python_constant_data for numpy arrays."""

    arrays = [
        numpy.array([[0, 1], [3, 4]], dtype=numpy.uint64),
        numpy.array([[0, 1], [3, 4]], dtype=numpy.float64),
    ]

    def get_expected_constructor(array: numpy.ndarray):
        return lambda x: numpy.full(array.shape, x, dtype=array.dtype)

    expected_constructors = [get_expected_constructor(array) for array in arrays]

    for array, expected_constructor in zip(arrays, expected_constructors):
        assert test_helpers.python_functions_are_equal_or_equivalent(
            expected_constructor, get_constructor_for_numpy_or_python_constant_data(array)
        )


def test_get_base_value_for_numpy_or_python_constant_data_with_list():
    """Test function for get_base_value_for_numpy_or_python_constant_data called with list"""

    with pytest.raises(
        AssertionError,
        match="Unsupported constant data of type list "
        "\\(if you meant to use a list as an array, please use numpy\\.array instead\\)",
    ):
        get_base_value_for_numpy_or_python_constant_data([1, 2, 3])
