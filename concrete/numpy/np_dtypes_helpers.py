"""File to hold code to manage package and numpy dtypes."""

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.base import BaseDataType
from ..common.data_types.dtypes_helpers import (
    BASE_DATA_TYPES,
    find_type_to_hold_both_lossy,
    get_base_data_type_for_python_constant_data,
    get_base_value_for_python_constant_data,
    get_constructor_for_python_constant_data,
)
from ..common.data_types.floats import Float
from ..common.data_types.integers import Integer
from ..common.debugging.custom_assert import assert_true
from ..common.tracing import BaseTracer
from ..common.values import BaseValue, TensorValue

NUMPY_TO_COMMON_DTYPE_MAPPING: Dict[numpy.dtype, BaseDataType] = {
    numpy.dtype(numpy.byte): Integer(numpy.byte(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.short): Integer(numpy.short(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.intc): Integer(numpy.intc(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.int_): Integer(numpy.int_(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.longlong): Integer(numpy.longlong(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.int8): Integer(numpy.int8(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.int16): Integer(numpy.int16(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.int32): Integer(numpy.int32(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.int64): Integer(numpy.int64(0).nbytes * 8, is_signed=True),
    numpy.dtype(numpy.ubyte): Integer(numpy.ubyte(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.ushort): Integer(numpy.ushort(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uintc): Integer(numpy.uintc(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uint): Integer(numpy.uint(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.ulonglong): Integer(numpy.ulonglong(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uint8): Integer(numpy.uint8(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uint16): Integer(numpy.uint16(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uint32): Integer(numpy.uint32(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.uint64): Integer(numpy.uint64(0).nbytes * 8, is_signed=False),
    numpy.dtype(numpy.float16): Float(16),
    numpy.dtype(numpy.float32): Float(32),
    numpy.dtype(numpy.float64): Float(64),
    numpy.dtype(bool): Integer(8, is_signed=False),
}

SUPPORTED_NUMPY_DTYPES = tuple(NUMPY_TO_COMMON_DTYPE_MAPPING)
SUPPORTED_NUMPY_DTYPES_CLASS_TYPES = tuple(dtype.type for dtype in NUMPY_TO_COMMON_DTYPE_MAPPING)

SUPPORTED_DTYPE_MSG_STRING = ", ".join(sorted(str(dtype) for dtype in SUPPORTED_NUMPY_DTYPES))


def convert_numpy_dtype_to_base_data_type(numpy_dtype: DTypeLike) -> BaseDataType:
    """Get the corresponding BaseDataType from a numpy dtype.

    Args:
        numpy_dtype (DTypeLike): Any python object that can be translated to a numpy.dtype

    Raises:
        ValueError: If the numpy_dtype is not supported

    Returns:
        BaseDataType: The corresponding data type corresponding to the input numpy_dtype
    """
    # Normalize numpy_dtype
    normalized_numpy_dtype = numpy.dtype(numpy_dtype)
    corresponding_common_dtype = NUMPY_TO_COMMON_DTYPE_MAPPING.get(normalized_numpy_dtype, None)

    if corresponding_common_dtype is None:
        raise ValueError(
            f"Unsupported numpy type: {numpy_dtype} ({normalized_numpy_dtype}), "
            f"supported numpy types: "
            f"{SUPPORTED_DTYPE_MSG_STRING}"
        )

    # deepcopy to avoid having the value from the dict modified
    return deepcopy(corresponding_common_dtype)


def convert_base_data_type_to_numpy_dtype(common_dtype: BaseDataType) -> numpy.dtype:
    """Convert a BaseDataType to corresponding numpy.dtype.

    Args:
        common_dtype (BaseDataType): dtype to convert to numpy.dtype

    Returns:
        numpy.dtype: The resulting numpy.dtype
    """
    assert_true(
        isinstance(common_dtype, BASE_DATA_TYPES), f"Unsupported common_dtype: {type(common_dtype)}"
    )
    type_to_return: numpy.dtype

    if isinstance(common_dtype, Float):
        assert_true(
            (bit_width := common_dtype.bit_width)
            in (
                16,
                32,
                64,
            ),
            "Only converting Float(16), Float(32) or Float(64) is supported",
        )
        if bit_width == 64:
            type_to_return = numpy.dtype(numpy.float64)
        elif bit_width == 32:
            type_to_return = numpy.dtype(numpy.float32)
        else:
            type_to_return = numpy.dtype(numpy.float16)
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
    """Determine the BaseDataType to hold the input constant data.

    Args:
        constant_data (Any): The constant data for which to determine the
            corresponding BaseDataType.

    Returns:
        BaseDataType: The corresponding BaseDataType
    """
    base_dtype: BaseDataType
    assert_true(
        isinstance(
            constant_data, (int, float, list, numpy.ndarray, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)
        ),
        f"Unsupported constant data of type {type(constant_data)}",
    )
    if isinstance(constant_data, (numpy.ndarray, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)):
        native_type = float if (constant_data.dtype in (numpy.float32, numpy.float64)) else int

        min_value = native_type(constant_data.min())
        max_value = native_type(constant_data.max())

        min_value_dtype = get_base_data_type_for_python_constant_data(min_value)
        max_value_dtype = get_base_data_type_for_python_constant_data(max_value)

        # numpy
        base_dtype = find_type_to_hold_both_lossy(min_value_dtype, max_value_dtype)
    else:
        # python
        base_dtype = get_base_data_type_for_python_constant_data(constant_data)
    return base_dtype


def get_base_value_for_numpy_or_python_constant_data(
    constant_data: Any,
) -> Callable[..., BaseValue]:
    """Determine the BaseValue and BaseDataType to hold the input constant data.

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
    assert_true(
        not isinstance(constant_data, list),
        "Unsupported constant data of type list "
        "(if you meant to use a list as an array, please use numpy.array instead)",
    )
    assert_true(
        isinstance(
            constant_data,
            (int, float, numpy.ndarray, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES),
        ),
        f"Unsupported constant data of type {type(constant_data)}",
    )

    base_dtype = get_base_data_type_for_numpy_or_python_constant_data(constant_data)
    if isinstance(constant_data, numpy.ndarray):
        constant_data_value = partial(TensorValue, dtype=base_dtype, shape=constant_data.shape)
    elif isinstance(constant_data, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES):
        constant_data_value = partial(TensorValue, dtype=base_dtype, shape=())
    else:
        constant_data_value = get_base_value_for_python_constant_data(constant_data)
    return constant_data_value


def get_numpy_function_output_dtype_and_shape_from_input_dtypes(
    function: Union[numpy.ufunc, Callable],
    input_dtypes: List[BaseDataType],
    input_shapes: List[Tuple[int, ...]],
) -> List[Tuple[numpy.dtype, Tuple[int, ...]]]:
    """Record the output dtype of a numpy function given some input types.

    Args:
        function (Union[numpy.ufunc, Callable]): The numpy function whose output types need to
            be recorded
        input_dtypes (List[BaseDataType]): BaseDataTypes in the same order as they will be used with
            the function inputs
        input_shapes (List[Tuple[int, ...]]): Shapes in the same order as they will be used with
            the function inputs

    Returns:
        List[Tuple[numpy.dtype, Tuple[int, ...]]]: appropriate (numpy.dtype, shape) tuple for each
            output of the function
    """
    if isinstance(function, numpy.ufunc):
        assert_true(
            (len(input_dtypes) == function.nin),
            f"Expected {function.nin} types, got {len(input_dtypes)}: {input_dtypes}",
        )

    input_numpy_dtypes = [convert_base_data_type_to_numpy_dtype(dtype) for dtype in input_dtypes]

    dummy_inputs = tuple(
        (
            dtype.type(10.0 * numpy.random.random_sample())
            if shape == ()
            else numpy.abs(numpy.random.randn(*shape) * 10.0).astype(dtype)
        )
        for dtype, shape in zip(input_numpy_dtypes, input_shapes)
    )

    # We ignore errors as we may call functions with invalid inputs just to get the proper output
    # dtypes
    with numpy.errstate(all="ignore"):
        outputs = function(*dummy_inputs)

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    return [(output.dtype, output.shape) for output in outputs]


def get_numpy_function_output_dtype_and_shape_from_input_tracers(
    func: Union[numpy.ufunc, Callable],
    *input_tracers: BaseTracer,
) -> List[Tuple[BaseDataType, Tuple[int, ...]]]:
    """Determine output dtypes and shapes for a numpy function.

    This function is responsible for determining the output dtype
    of a numpy function after inputs with specific dtypes are passed to it.

    Args:
        func (Union[numpy.ufunc, Callable]): function that is being managed
        *input_tracers (BaseTracer): inputs to the function

    Returns:
        List[Tuple[BaseDataType, Tuple[int, ...]]]: appropriate (BaseDataType, shape) tuple for each
            output of the function
    """

    input_shapes = [
        input_tracer.output.shape if isinstance(input_tracer.output, TensorValue) else ()
        for input_tracer in input_tracers
    ]
    output_dtypes_and_shapes = get_numpy_function_output_dtype_and_shape_from_input_dtypes(
        func,
        [input_tracer.output.dtype for input_tracer in input_tracers],
        input_shapes,
    )
    common_output_dtypes = [
        (convert_numpy_dtype_to_base_data_type(dtype), shape)
        for dtype, shape in output_dtypes_and_shapes
    ]
    return common_output_dtypes


def get_constructor_for_numpy_or_python_constant_data(constant_data: Any):
    """Get the constructor for the numpy constant data or python dtype.

    Args:
        constant_data (Any): The data for which we want to determine the type constructor.
    """

    assert_true(
        isinstance(
            constant_data, (int, float, list, numpy.ndarray, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)
        ),
        f"Unsupported constant data of type {type(constant_data)}",
    )

    if isinstance(constant_data, list):
        # this is required because some operations return python lists from their evaluate function
        # an example of such operation is evaluation of multi tlu during bound measurements
        constant_data = numpy.array(constant_data)

    if isinstance(constant_data, (numpy.ndarray, SUPPORTED_NUMPY_DTYPES_CLASS_TYPES)):
        if isinstance(constant_data, numpy.ndarray):
            return lambda x: numpy.full(constant_data.shape, x, dtype=constant_data.dtype)
        return constant_data.dtype.type

    return get_constructor_for_python_constant_data(constant_data)
