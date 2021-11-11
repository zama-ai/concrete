"""File to hold helper functions for data types related stuff."""

from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Tuple, Union, cast

from ..debugging.custom_assert import assert_true
from ..values import BaseValue, ClearTensor, EncryptedTensor, TensorValue
from .base import BaseDataType
from .floats import Float
from .integers import Integer, get_bits_to_represent_value_as_integer

INTEGER_TYPES = (Integer,)
FLOAT_TYPES = (Float,)
BASE_DATA_TYPES = INTEGER_TYPES + FLOAT_TYPES


def value_is_encrypted_scalar_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is an encrypted scalar of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted scalar of type Integer
    """
    return value_is_scalar_integer(value_to_check) and value_to_check.is_encrypted


def value_is_encrypted_scalar_unsigned_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is an encrypted scalar of type unsigned Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted scalar of type Integer and
            unsigned
    """
    return (
        value_is_encrypted_scalar_integer(value_to_check)
        and not cast(Integer, value_to_check.dtype).is_signed
    )


def value_is_clear_scalar_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is a clear scalar of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a clear scalar of type Integer
    """
    return value_is_scalar_integer(value_to_check) and value_to_check.is_clear


def value_is_scalar_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is a scalar of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a scalar of type Integer
    """
    return (
        isinstance(value_to_check, TensorValue)
        and value_to_check.is_scalar
        and isinstance(value_to_check.dtype, INTEGER_TYPES)
    )


def value_is_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is of type Integer
    """

    return isinstance(value_to_check.dtype, INTEGER_TYPES)


def value_is_unsigned_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is of type Integer and is unsigned.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is of type Integer and is unsigned
    """

    return (
        isinstance(value_to_check.dtype, INTEGER_TYPES)
        and not cast(Integer, value_to_check.dtype).is_signed
    )


def value_is_encrypted_tensor_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is an encrypted TensorValue of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted TensorValue of type Integer
    """
    return value_is_tensor_integer(value_to_check) and value_to_check.is_encrypted


def value_is_clear_tensor_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is a clear TensorValue of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a clear TensorValue of type Integer
    """
    return value_is_tensor_integer(value_to_check) and value_to_check.is_clear


def value_is_tensor_integer(value_to_check: BaseValue) -> bool:
    """Check that a value is a TensorValue of type Integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a TensorValue of type Integer
    """
    return (
        isinstance(value_to_check, TensorValue)
        and not value_to_check.is_scalar
        and isinstance(value_to_check.dtype, INTEGER_TYPES)
    )


def find_type_to_hold_both_lossy(
    dtype1: BaseDataType,
    dtype2: BaseDataType,
) -> BaseDataType:
    """Determine the type that can represent both dtype1 and dtype2 separately.

    This is lossy with floating point types.

    Args:
        dtype1 (BaseDataType): first dtype to hold
        dtype2 (BaseDataType): second dtype to hold

    Raises:
        NotImplementedError: Raised if one of the two input dtypes is not an Integer as they are the
            only type supported for now

    Returns:
        BaseDataType: The dtype able to hold (potentially lossy) dtype1 and dtype2
    """
    assert_true(isinstance(dtype1, BASE_DATA_TYPES), f"Unsupported dtype1: {type(dtype1)}")
    assert_true(isinstance(dtype2, BASE_DATA_TYPES), f"Unsupported dtype2: {type(dtype2)}")

    type_to_return: BaseDataType

    if isinstance(dtype1, Integer) and isinstance(dtype2, Integer):
        d1_signed = dtype1.is_signed
        d2_signed = dtype2.is_signed
        max_bits = max(dtype1.bit_width, dtype2.bit_width)

        if d1_signed and d2_signed:
            type_to_return = Integer(max_bits, is_signed=True)
        elif not d1_signed and not d2_signed:
            type_to_return = Integer(max_bits, is_signed=False)
        elif d1_signed and not d2_signed:
            # 2 is unsigned, if it has the bigger bit_width, we need a signed integer that can hold
            # it, so add 1 bit of sign to its bit_width
            if dtype2.bit_width >= dtype1.bit_width:
                new_bit_width = dtype2.bit_width + 1
                type_to_return = Integer(new_bit_width, is_signed=True)
            else:
                type_to_return = Integer(dtype1.bit_width, is_signed=True)
        elif not d1_signed and d2_signed:
            # Same as above, with 1 and 2 switched around
            if dtype1.bit_width >= dtype2.bit_width:
                new_bit_width = dtype1.bit_width + 1
                type_to_return = Integer(new_bit_width, is_signed=True)
            else:
                type_to_return = Integer(dtype2.bit_width, is_signed=True)
    elif isinstance(dtype1, Float) and isinstance(dtype2, Float):
        max_bits = max(dtype1.bit_width, dtype2.bit_width)
        type_to_return = Float(max_bits)
    elif isinstance(dtype1, Float):
        type_to_return = deepcopy(dtype1)
    elif isinstance(dtype2, Float):
        type_to_return = deepcopy(dtype2)

    return type_to_return


def mix_tensor_values_determine_holding_dtype(
    value1: TensorValue,
    value2: TensorValue,
) -> TensorValue:
    """Return mixed TensorValue with data type able to hold both value1 and value2 dtypes.

    Returns a TensorValue that would result from computation on both value1 and value2 while
    determining the data type able to hold both value1 and value2 data type (this can be lossy
    with floats).

    Args:
        value1 (TensorValue): first TensorValue to mix.
        value2 (TensorValue): second TensorValue to mix.

    Returns:
        TensorValue: The resulting mixed TensorValue with data type able to hold both value1 and
            value2 dtypes.
    """

    assert_true(
        isinstance(value1, TensorValue), f"Unsupported value1: {value1}, expected TensorValue"
    )
    assert_true(
        isinstance(value2, TensorValue), f"Unsupported value2: {value2}, expected TensorValue"
    )

    resulting_shape = broadcast_shapes(value1.shape, value2.shape)
    assert_true(
        resulting_shape is not None,
        (
            f"Tensors have incompatible shapes which is not supported.\n"
            f"value1: {value1.shape}, value2: {value2.shape}"
        ),
    )
    assert resulting_shape is not None  # this is to make mypy happy

    holding_type = find_type_to_hold_both_lossy(value1.dtype, value2.dtype)
    if value1.is_encrypted or value2.is_encrypted:
        mixed_value = EncryptedTensor(dtype=holding_type, shape=resulting_shape)
    else:
        mixed_value = ClearTensor(dtype=holding_type, shape=resulting_shape)

    return mixed_value


def mix_values_determine_holding_dtype(value1: BaseValue, value2: BaseValue) -> BaseValue:
    """Return mixed BaseValue with data type able to hold both value1 and value2 dtypes.

    Returns a BaseValue that would result from computation on both value1 and value2 while
    determining the data type able to hold both value1 and value2 data type (this can be lossy
    with floats). Supports only mixing instances from the same class.

    Args:
        value1 (BaseValue): first BaseValue to mix.
        value2 (BaseValue): second BaseValue to mix.

    Raises:
        ValueError: raised if the BaseValue is not one of (TensorValue)

    Returns:
        BaseValue: The resulting mixed BaseValue with data type able to hold both value1 and value2
            dtypes.
    """

    assert_true(
        (value1.__class__ == value2.__class__),
        f"Cannot mix values of different types: value 1:{type(value1)}, value2: {type(value2)}",
    )

    if isinstance(value1, TensorValue) and isinstance(value2, TensorValue):
        return mix_tensor_values_determine_holding_dtype(value1, value2)

    raise ValueError(
        f"{mix_values_determine_holding_dtype.__name__} does not support value {type(value1)}"
    )


def get_base_data_type_for_python_constant_data(constant_data: Union[int, float]) -> BaseDataType:
    """Determine the BaseDataType to hold the input constant data.

    Args:
        constant_data (Union[int, float]): The constant data for which to determine the
            corresponding BaseDataType.

    Returns:
        BaseDataType: The corresponding BaseDataType
    """
    constant_data_type: BaseDataType
    assert_true(
        isinstance(constant_data, (int, float)),
        f"Unsupported constant data of type {type(constant_data)}",
    )

    if isinstance(constant_data, int):
        is_signed = constant_data < 0
        constant_data_type = Integer(
            get_bits_to_represent_value_as_integer(constant_data, is_signed), is_signed
        )
    elif isinstance(constant_data, float):
        constant_data_type = Float(64)

    return constant_data_type


def get_base_value_for_python_constant_data(
    constant_data: Union[int, float]
) -> Callable[..., BaseValue]:
    """Wrap the BaseDataType to hold the input constant data in BaseValue partial.

    The returned object can then be instantiated as an Encrypted or Clear version
    by calling it with the proper arguments forwarded to the BaseValue `__init__` function

    Args:
        constant_data (Union[int, float]): The constant data for which to determine the
            corresponding Value.

    Returns:
        Callable[..., BaseValue]: A partial object that will return the proper BaseValue when
            called with `is_encrypted` as keyword argument (forwarded to the BaseValue `__init__`
            method).
    """

    constant_data_type = get_base_data_type_for_python_constant_data(constant_data)
    return partial(TensorValue, dtype=constant_data_type, shape=())


def get_constructor_for_python_constant_data(constant_data: Union[int, float]):
    """Get the constructor for the passed python constant data.

    Args:
        constant_data (Any): The data for which we want to determine the type constructor.
    """
    return type(constant_data)


def is_data_type_compatible_with(
    dtype: BaseDataType,
    other: BaseDataType,
) -> bool:
    """Determine whether dtype is compatible with other.

    `dtype` being compatible with `other` means `other` can hold every value of `dtype`
    (e.g., uint2 is compatible with uint4 and int4)
    (e.g., int2 is compatible with int4 but not with uint4)

    Note that this function is not symetric.
    (e.g., uint2 is compatible with uint4, but uint4 is not compatible with uint2)

    Args:
        dtype (BaseDataType): dtype to check compatiblity
        other (BaseDataType): dtype to check compatiblity against

    Returns:
        bool: Whether the dtype is compatible with other or not
    """

    combination = find_type_to_hold_both_lossy(dtype, other)
    return other == combination


def broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Broadcast two shapes into a single shape.

    We are mimicing the exact semantics of broadcasting in numpy.
    You can learn more about it here: https://numpy.org/doc/stable/user/theory.broadcasting.html

    Args:
        shape1 (Tuple[int, ...]): first shape to broadcast
        shape2 (Tuple[int, ...]): second shape to broadcast

    Returns:
        Optional[Tuple[int, ...]]: None if the shapes are not broadcastable else broadcasted shape
    """

    result = []
    for size1, size2 in zip(shape1[::-1], shape2[::-1]):
        if size1 != size2 and size1 != 1 and size2 != 1 and size1 != 0 and size2 != 0:
            return None

        if size1 == 0 or size2 == 0:
            result.append(0)
        else:
            result.append(max(size1, size2))

    if len(result) < len(shape1):
        for i in reversed(range(len(shape1) - len(result))):
            result.append(shape1[i])
    elif len(result) < len(shape2):
        for i in reversed(range(len(shape2) - len(result))):
            result.append(shape2[i])

    return tuple(reversed(result))
