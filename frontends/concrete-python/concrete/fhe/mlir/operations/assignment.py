"""
Conversion of assignment operation.
"""

# pylint: disable=import-error,no-name-in-module

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
from concrete.lang.dialects import fhelinalg
from mlir.dialects import tensor
from mlir.ir import DenseI64ArrayAttr as MlirDenseI64ArrayAttr
from mlir.ir import ShapedType as MlirShapedType

from ..context import Context
from ..conversion import Conversion, ConversionType
from .indexing import generate_fancy_indices, process_indexing_element

# pylint: enable=import-error,no-name-in-module


def fancy_assignment(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
) -> Conversion:
    """
    Convert fancy assignment operation.

    Args:
        ctx (Context):
            conversion context

        resulting_type (ConversionType):
            resulting type of the operation

        x (Conversion):
            tensor to assign to

        y (Conversion):
            tensor to assign

        index (Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]]):
            fancy index to use

    Returns:
        Conversion:
            x after fancy assignment
    """

    sample_index = []
    for indexing_element in index:
        sample_index.append(
            np.zeros(indexing_element.shape, dtype=np.int64)
            if isinstance(indexing_element, Conversion)
            else indexing_element
        )

    indexing_element_shape = np.zeros(resulting_type.shape, dtype=np.int8)[
        tuple(sample_index)
    ].shape

    indices = generate_fancy_indices(
        ctx,
        indexing_element_shape,
        x,
        index,
        check_out_of_bounds=ctx.configuration.dynamic_assignment_check_out_of_bounds,
    )

    if y.shape != indexing_element_shape:
        y = ctx.broadcast_to(y, indexing_element_shape)

    return ctx.operation(
        fhelinalg.FancyAssignOp,
        resulting_type,
        x.result,
        indices.result,
        y.result,
    )


def assignment(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
) -> Conversion:
    """
    Convert assignment operation.

    Args:
        ctx (Context):
            conversion context

        resulting_type (ConversionType):
            resulting type of the operation

        x (Conversion):
            tensor to assign to

        y (Conversion):
            tensor to assign

        index (Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]]):
            index to use

    Returns:
        Conversion:
            x after assignment
    """

    if x.is_clear and y.is_encrypted:
        highlights = {
            x.origin: "tensor is clear",
            y.origin: "assigned value is encrypted",
            ctx.converting: "but encrypted values cannot be assigned to clear tensors",
        }
        ctx.error(highlights)

    assert ctx.is_bit_width_compatible(resulting_type, x, y)

    index = list(index)
    while len(index) < len(x.shape):
        index.append(slice(None, None, None))

    if x.is_encrypted and y.is_clear:
        encrypted_type = ctx.tensor(ctx.element_typeof(x), y.shape)
        y = ctx.encrypt(encrypted_type, y)

    is_fancy = any(
        (
            isinstance(indexing_element, (list, np.ndarray))
            or (isinstance(indexing_element, Conversion) and indexing_element.is_tensor)
        )
        for indexing_element in index
    )
    if is_fancy:
        return fancy_assignment(ctx, resulting_type, x, y, index)

    static_offsets: list[Any] = []
    static_sizes: list[Any] = []
    static_strides: list[Any] = []

    dynamic_offsets: list[Any] = []

    for indexing_element, dimension_size in zip(index, x.shape):
        offset: Any
        size: Any
        stride: Any

        if isinstance(indexing_element, slice):
            size = int(np.zeros(dimension_size)[indexing_element].shape[0])
            stride = int(indexing_element.step if indexing_element.step is not None else 1)
            offset = int(
                process_indexing_element(
                    ctx,
                    indexing_element.start,  # type: ignore
                    dimension_size,
                    check_out_of_bounds=ctx.configuration.dynamic_assignment_check_out_of_bounds,
                )
                if indexing_element.start is not None
                else (0 if stride > 0 else dimension_size - 1)
            )
        else:
            assert isinstance(indexing_element, (int, np.integer)) or (
                isinstance(indexing_element, Conversion) and indexing_element.is_scalar
            )

            size = 1
            stride = 1
            offset = process_indexing_element(
                ctx,
                indexing_element,
                dimension_size,
                check_out_of_bounds=ctx.configuration.dynamic_assignment_check_out_of_bounds,
            )

            if isinstance(offset, Conversion):
                dynamic_offsets.append(offset)
                offset = MlirShapedType.get_dynamic_size()

        static_offsets.append(offset)
        static_sizes.append(size)
        static_strides.append(stride)

    required_y_shape_list = []
    for i, indexing_element in enumerate(index):
        if isinstance(indexing_element, slice):
            n = len(np.zeros(x.shape[i])[indexing_element])
            required_y_shape_list.append(n)
        else:
            required_y_shape_list.append(1)

    required_y_shape = tuple(required_y_shape_list)
    try:
        np.reshape(np.zeros(y.shape), required_y_shape)
        y = ctx.reshape(y, required_y_shape)
    except Exception:  # pylint: disable=broad-except
        np.broadcast_to(np.zeros(y.shape), required_y_shape)
        y = ctx.broadcast_to(y, required_y_shape)

    x = ctx.to_signedness(x, of=resulting_type)
    y = ctx.to_signedness(y, of=resulting_type)

    return ctx.operation(
        tensor.InsertSliceOp,
        resulting_type,
        y.result,
        x.result,
        tuple(item.result for item in dynamic_offsets),
        (),
        (),
        MlirDenseI64ArrayAttr.get(static_offsets),
        MlirDenseI64ArrayAttr.get(static_sizes),
        MlirDenseI64ArrayAttr.get(static_strides),
        original_bit_width=x.original_bit_width,
    )
