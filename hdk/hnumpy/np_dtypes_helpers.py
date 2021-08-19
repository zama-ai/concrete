"""File to hold code to manage package and numpy dtypes."""

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.base import BaseDataType
from ..common.data_types.dtypes_helpers import (
    BASE_DATA_TYPES,
    get_base_data_type_for_python_constant_data,
    get_base_value_for_python_constant_data,
)
from ..common.data_types.floats import Float
from ..common.data_types.integers import Integer
from ..common.data_types.values import BaseValue, ScalarValue

NUMPY_TO_HDK_DTYPE_MAPPING: Dict[numpy.dtype, BaseDataType] = {
    numpy.dtype(numpy.int32): Integer(32, is_signed=True),
    numpy.dtype(numpy.int64): Integer(64, is_signed=True),
    numpy.dtype(numpy.uint32): Integer(32, is_signed=False),
    numpy.dtype(numpy.uint64): Integer(64, is_signed=False),
    numpy.dtype(numpy.float32): Float(32),
    numpy.dtype(numpy.float64): Float(64),
}

SUPPORTED_NUMPY_DTYPES = tuple(NUMPY_TO_HDK_DTYPE_MAPPING)
SUPPORTED_NUMPY_DTYPES_CLASS_TYPES = tuple(dtype.type for dtype in NUMPY_TO_HDK_DTYPE_MAPPING)

SUPPORTED_DTYPE_MSG_STRING = ", ".join(sorted(str(dtype) for dtype in SUPPORTED_NUMPY_DTYPES))


def convert_numpy_dtype_to_base_data_type(numpy_dtype: DTypeLike) -> BaseDataType:
    """Helper function to get the corresponding BaseDataType from a numpy dtype.

    Args:
        numpy_dtype (DTypeLike): Any python object that can be translated to a numpy.dtype

    Raises:
        ValueError: If the numpy_dtype is not supported

    Returns:
        BaseDataType: The corresponding data type corresponding to the input numpy_dtype
    """
    # Normalize numpy_dtype
    normalized_numpy_dtype = numpy.dtype(numpy_dtype)
    corresponding_hdk_dtype = NUMPY_TO_HDK_DTYPE_MAPPING.get(normalized_numpy_dtype, None)

    if corresponding_hdk_dtype is None:
        raise ValueError(
            f"Unsupported numpy type: {numpy_dtype} ({normalized_numpy_dtype}), "
            f"supported numpy types: "
            f"{SUPPORTED_DTYPE_MSG_STRING}"
        )

    # deepcopy to avoid having the value from the dict modified
    return deepcopy(corresponding_hdk_dtype)


def convert_base_data_type_to_numpy_dtype(common_dtype: BaseDataType) -> numpy.dtype:
    """Convert a BaseDataType to corresponding numpy.dtype.

    Args:
        common_dtype (BaseDataType): dtype to convert to numpy.dtype

    Returns:
        numpy.dtype: The resulting numpy.dtype
    """
    assert isinstance(
        common_dtype, BASE_DATA_TYPES
    ), f"Unsupported common_dtype: {type(common_dtype)}"
    type_to_return: numpy.dtype

    if isinstance(common_dtype, Float):
        assert common_dtype.bit_width in (
            32,
            64,
        ), "Only converting Float(32) or Float(64) is supported"
        type_to_return = (
            numpy.dtype(numpy.float64)
            if common_dtype.bit_width == 64
            else numpy.dtype(numpy.float32)
        )
    elif isinstance(common_dtype, Integer):
        signed = common_dtype.is_signed
        if common_dtype.bit_width <= 32:
            type_to_return = numpy.dtype(numpy.int32) if signed else numpy.dtype(numpy.uint32)
        elif common_dtype.bit_width <= 64:
            type_to_return = numpy.dtype(numpy.int64) if signed else numpy.dtype(numpy.uint64)
        else:
            raise NotImplementedError(
                f"Conversion to numpy dtype only supports Integers with bit_width <= 64, "
                f"got {common_dtype!r}"
            )

    return type_to_return


def get_base_data_type_for_numpy_or_python_constant_data(constant_data: Any) -> BaseDataType:
    """Helper function to determine the BaseDataType to hold the input constant data.

    Args:
        constant_data (Any): The constant data for which to determine the
            corresponding BaseDataType.

    Returns:
        BaseDataType: The corresponding BaseDataType
    """
    base_dtype: BaseDataType
    assert isinstance(
        constant_data, (int, float, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)
    ), f"Unsupported constant data of type {type(constant_data)}"
    if isinstance(constant_data, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES):
        base_dtype = convert_numpy_dtype_to_base_data_type(constant_data)
    else:
        base_dtype = get_base_data_type_for_python_constant_data(constant_data)
    return base_dtype


def get_base_value_for_numpy_or_python_constant_data(
    constant_data: Any,
) -> Callable[..., BaseValue]:
    """Helper function to determine the BaseValue and BaseDataType to hold the input constant data.

    This function is able to handle numpy types

    Args:
        constant_data (Any): The constant data for which to determine the
            corresponding BaseValue and BaseDataType.

    Raises:
        AssertionError: If `constant_data` is of an unsupported type.

    Returns:
        Callable[..., BaseValue]: A partial object that will return the proper BaseValue when called
            with `encrypted` as keyword argument (forwarded to the BaseValue `__init__` method).
    """
    constant_data_value: Callable[..., BaseValue]
    assert isinstance(
        constant_data, (int, float, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)
    ), f"Unsupported constant data of type {type(constant_data)}"

    base_dtype = get_base_data_type_for_numpy_or_python_constant_data(constant_data)
    if isinstance(constant_data, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES):
        constant_data_value = partial(ScalarValue, data_type=base_dtype)
    else:
        constant_data_value = get_base_value_for_python_constant_data(constant_data)
    return constant_data_value


def get_ufunc_numpy_output_dtype(
    ufunc: numpy.ufunc,
    input_dtypes: List[BaseDataType],
) -> List[numpy.dtype]:
    """Function to record the output dtype of a numpy.ufunc given some input types.

    Args:
        ufunc (numpy.ufunc): The numpy.ufunc whose output types need to be recorded
        input_dtypes (List[BaseDataType]): Common dtypes in the same order as they will be used with
            the ufunc inputs

    Returns:
        List[numpy.dtype]: The ordered numpy dtypes of the ufunc outputs
    """
    assert (
        len(input_dtypes) == ufunc.nin
    ), f"Expected {ufunc.nin} types, got {len(input_dtypes)}: {input_dtypes}"

    input_numpy_dtypes = [convert_base_data_type_to_numpy_dtype(dtype) for dtype in input_dtypes]

    # Store numpy old error settings and ignore all errors in this function
    # We ignore errors as we may call functions with invalid inputs just to get the proper output
    # dtypes
    old_numpy_err_settings = numpy.seterr(all="ignore")

    dummy_inputs = tuple(
        dtype.type(1000.0 * numpy.random.random_sample()) for dtype in input_numpy_dtypes
    )

    outputs = ufunc(*dummy_inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    # Restore numpy error settings
    numpy.seterr(**old_numpy_err_settings)

    return [output.dtype for output in outputs]
