"""
Conversion of indexing operation.
"""

# pylint: disable=import-error,no-name-in-module

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from concrete.lang.dialects import fhelinalg, tracing
from mlir.dialects import arith, tensor
from mlir.ir import ArrayAttr as MlirArrayAttr
from mlir.ir import DenseI64ArrayAttr as MlirDenseI64ArrayAttr
from mlir.ir import InsertionPoint as MlirInsertionPoint
from mlir.ir import IntegerAttr as MlirIntegerAttr
from mlir.ir import ShapedType as MlirShapedType

from ...dtypes import Integer
from ...internal.utils import unreachable
from ..context import Context
from ..conversion import Conversion, ConversionType

# pylint: enable=import-error,no-name-in-module


def check_out_of_bounds_in_runtime(
    ctx: Context,
    indexing_element: Conversion,
    dimension_size: int,
):
    """
    Add out of bounds checks in runtime.

    Args:
        ctx (Context):
            conversion context

        indexing_element (Conversion):
            indexing element to check

        dimension_size (int):
            size of the indexed dimension
    """

    dimension_size_variable_type = ctx.i(
        max(
            Integer.that_can_represent(dimension_size).bit_width + 1,
            indexing_element.bit_width,
        )
    )
    dimension_size_variable = ctx.constant(dimension_size_variable_type, dimension_size)

    if indexing_element.bit_width < dimension_size_variable.bit_width:
        indexing_element = ctx.operation(
            arith.ExtSIOp,
            ctx.tensor(dimension_size_variable_type, shape=indexing_element.shape),
            indexing_element.result,
        )

    def warn_out_of_memory_access():
        pred_ids = [pred.properties["id"] for pred in ctx.graph.ordered_preds_of(ctx.converting)]
        operation_string = ctx.converting.format(pred_ids)
        tracing.TraceMessageOp(
            msg=(
                f"Runtime Warning: Index out of range on "
                f"\"{ctx.converting.properties['id']} = {operation_string}\"\n"
            )
        )

    if indexing_element.is_scalar:
        indexing_element_too_large_condition = ctx.operation(
            arith.CmpIOp,
            ctx.i(1),
            ctx.attribute(ctx.i(64), 5),  # signed greater than or equal
            indexing_element.result,
            dimension_size_variable.result,
        )
        ctx.conditional(
            None,
            indexing_element_too_large_condition,
            warn_out_of_memory_access,
        )

        indexing_element_too_small_condition = ctx.operation(
            arith.CmpIOp,
            ctx.i(1),
            ctx.attribute(ctx.i(64), 2),  # signed less than
            indexing_element.result,
            ctx.constant(indexing_element.type, 0).result,
        )
        ctx.conditional(
            None,
            indexing_element_too_small_condition,
            warn_out_of_memory_access,
        )

    else:

        def create_for_body(indexing_element_shape, indices):
            remaining_shape = indexing_element_shape[1:]

            def body(i):
                new_indices = indices + (i,)
                if remaining_shape != ():
                    ctx.for_loop(
                        0,
                        remaining_shape[0],
                        create_for_body(remaining_shape, new_indices),
                    )
                    return

                element = indexing(
                    ctx,
                    ctx.element_typeof(indexing_element),
                    indexing_element,
                    new_indices,
                )
                check_out_of_bounds_in_runtime(ctx, element, dimension_size)

            return body

        ctx.for_loop(
            0,
            indexing_element.shape[0],
            create_for_body(indexing_element.shape, ()),
        )


def process_indexing_element(
    ctx: Context,
    indexing_element: Union[int, np.integer, slice, np.ndarray, list, Conversion],
    dimension_size: int,
    check_out_of_bounds: bool,
) -> Union[int, np.integer, slice, np.ndarray, list, Conversion]:
    """
    Process indexing element.

    - Variables would be checked for of bounds and converted to index type.
    - Constants will be sanitized to be in range(0, dimension_size).
    - Slices are not supported.

    Args:
        ctx (Context):
            conversion context

        indexing_element (Conversion):
            indexing element to process

        dimension_size (int):
            size of the indexed dimension

        check_out_of_bounds (int):
            whether to check for out of bounds access in runtime

    Returns:
        Union[int, np.integer, slice, np.ndarray, list, Conversion]:
            processed indexing element
    """

    result: Union[int, np.integer, slice, np.ndarray, list, Conversion]

    if isinstance(indexing_element, (int, np.integer)):
        result = int(indexing_element)
        if result < 0:
            result += dimension_size

        assert 0 <= result < dimension_size
        return result

    if isinstance(indexing_element, (list, np.ndarray)):
        indexing_element = np.array(indexing_element, dtype=np.int64)
        result = np.where(indexing_element < 0, indexing_element + dimension_size, indexing_element)

        assert 0 <= result.min() <= result.max() < dimension_size
        return result

    if isinstance(indexing_element, Conversion):
        assert indexing_element.is_clear
        if indexing_element.type.is_index:
            return indexing_element

        indexing_element_node = indexing_element.origin
        assert isinstance(indexing_element_node.output.dtype, Integer)

        dimension_size_variable: Optional[Conversion] = None
        dimension_size_variable_type = ctx.i(
            max(
                Integer.that_can_represent(dimension_size).bit_width + 1,
                indexing_element.bit_width,
            )
        )

        if indexing_element_node.output.dtype.is_signed:
            dimension_size_variable = ctx.constant(dimension_size_variable_type, dimension_size)
            if indexing_element.bit_width < dimension_size_variable.bit_width:
                indexing_element = ctx.operation(
                    arith.ExtSIOp,
                    ctx.tensor(dimension_size_variable_type, shape=indexing_element.shape),
                    indexing_element.result,
                )

            if indexing_element.is_scalar:
                offset_condition = ctx.operation(
                    arith.CmpIOp,
                    ctx.i(1),
                    ctx.attribute(ctx.i(64), 2),  # signed less than
                    indexing_element.result,
                    ctx.constant(indexing_element.type, 0).result,
                )

                assert isinstance(indexing_element, Conversion)
                new_indexing_element = ctx.conditional(
                    indexing_element.type,
                    offset_condition,
                    lambda: ctx.operation(
                        arith.AddIOp,
                        indexing_element.type,  # type: ignore
                        indexing_element.result,  # type: ignore
                        dimension_size_variable.result,  # type: ignore
                    ),
                    lambda: indexing_element,  # type: ignore
                )

                assert new_indexing_element is not None
                indexing_element = new_indexing_element

                if check_out_of_bounds:
                    check_out_of_bounds_in_runtime(ctx, indexing_element, dimension_size)
            else:
                element_type = ctx.element_typeof(indexing_element)

                sanitized_indexing_element = tensor.GenerateOp(
                    indexing_element.type.mlir,
                    (),
                )
                sanitized_indexing_element.body.blocks.append(
                    *[ctx.index_type().mlir for _ in range(len(indexing_element.shape))],
                )

                block = sanitized_indexing_element.regions[0].blocks[0]
                indices = tuple(
                    Conversion(ctx.converting, block.arguments[i])
                    for i in range(len(block.arguments))
                )

                with MlirInsertionPoint.at_block_begin(block):
                    index = indexing(ctx, element_type, indexing_element, indices)

                    offset_condition = ctx.operation(
                        arith.CmpIOp,
                        ctx.i(1),
                        ctx.attribute(ctx.i(64), 2),  # signed less than
                        index.result,
                        ctx.constant(index.type, 0, use_cache=False).result,
                        use_cache=False,
                    )
                    sanitized_index = ctx.conditional(
                        index.type,
                        offset_condition,
                        lambda: ctx.operation(
                            arith.AddIOp,
                            index.type,
                            index.result,
                            dimension_size_variable.result,  # type: ignore
                        ),
                        lambda: index,
                    )
                    assert sanitized_index is not None

                    if check_out_of_bounds:
                        check_out_of_bounds_in_runtime(ctx, sanitized_index, dimension_size)

                    assert sanitized_index is not None
                    ctx.operation(
                        tensor.YieldOp,
                        element_type,
                        sanitized_index.result,
                        use_cache=False,
                    )

                indexing_element = Conversion(
                    ctx.converting,
                    sanitized_indexing_element.result,
                )

        elif check_out_of_bounds:
            check_out_of_bounds_in_runtime(ctx, indexing_element, dimension_size)

        return ctx.operation(
            arith.IndexCastOp,
            ctx.tensor(ctx.index_type(), shape=indexing_element.shape),
            indexing_element.result,
        )

    unreachable()  # pragma: no cover
    return 0  # pragma: no cover


def generate_fancy_indices(
    ctx: Context,
    indexing_element_shape: tuple[int, ...],
    x: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
    check_out_of_bounds: bool,
) -> Conversion:
    """
    Generate indices to use for fancy indexing.

    Args:
        ctx (Context):
            conversion context

        indexing_element_shape (Tuple[int, ...]):
            individual shape of indexing elements

        x (Conversion):
            tensor to fancy index

        index (Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]]):
            fancy index to use

        check_out_of_bounds (int):
            whether to check for out of bounds access in runtime

    Returns:
        Conversion:
            indices to use for fancy indexing operation
    """

    # refer to
    # - https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    # - https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # for semantics

    just_seen_slice = False
    fancy_indices_start_positions = []

    for i in range(len(index)):
        if isinstance(index[i], slice):
            just_seen_slice = True
        elif just_seen_slice or i == 0:
            fancy_indices_start_positions.append(i)
            just_seen_slice = False

    processed_index = []
    for dimension_size, indexing_element in zip(x.shape, index):
        if isinstance(indexing_element, slice):
            indexing_element = np.array(list(range(dimension_size))[indexing_element])

        indexing_element = process_indexing_element(
            ctx,
            indexing_element,
            dimension_size,
            check_out_of_bounds,
        )

        if isinstance(indexing_element, Conversion):
            processed_index.append(indexing_element)
            continue

        if isinstance(indexing_element, (int, np.integer)):
            processed_index.append(ctx.constant(ctx.index_type(), indexing_element))
            continue

        if isinstance(indexing_element, (list, np.ndarray)):
            processed_index.append(
                ctx.constant(
                    ctx.tensor(ctx.index_type(), np.array(indexing_element).shape),
                    indexing_element,
                )
            )
            continue

        message = f"invalid indexing element of type {type(indexing_element)}"  # pragma: no cover
        raise AssertionError(message)  # pragma: no cover

    if len(processed_index) == 1:
        assert len(x.shape) == 1
        indices = processed_index[0]
    else:
        expanded_indexing_element_shape = indexing_element_shape + (1,)

        to_concat = []
        for dimension, indexing_element in enumerate(processed_index):
            if indexing_element.is_scalar:
                to_concat.append(
                    ctx.broadcast_to(indexing_element, expanded_indexing_element_shape)
                )

            elif indexing_element.shape == indexing_element_shape:
                to_concat.append(ctx.reshape(indexing_element, expanded_indexing_element_shape))

            else:
                if len(fancy_indices_start_positions) == 1:
                    broadcast_shape: tuple[int, ...] = (1,)
                    for original_indexing_element in index:
                        if isinstance(original_indexing_element, (list, np.ndarray)) or (
                            isinstance(original_indexing_element, Conversion)
                            and original_indexing_element.is_tensor
                        ):
                            if isinstance(original_indexing_element, list):
                                original_indexing_element = np.array(original_indexing_element)
                            broadcast_shape = np.broadcast_shapes(
                                broadcast_shape,
                                original_indexing_element.shape,
                            )

                    extra_dimensions = 1
                    if isinstance(index[dimension], slice):
                        if dimension < fancy_indices_start_positions[0]:
                            extra_dimensions += len(broadcast_shape)

                    for original_indexing_element in index[(dimension + 1) :]:
                        if isinstance(original_indexing_element, slice):
                            extra_dimensions += 1
                else:
                    extra_dimensions = 1
                    if isinstance(index[dimension], slice):
                        for original_indexing_element in index[(dimension + 1) :]:
                            if isinstance(original_indexing_element, slice):
                                extra_dimensions += 1
                    else:
                        removed_dimensions = 0
                        for original_indexing_element in index:
                            if isinstance(original_indexing_element, (int, np.integer)) or (
                                isinstance(original_indexing_element, Conversion)
                                and original_indexing_element.is_scalar
                            ):
                                removed_dimensions += 1

                        extra_dimensions = 1
                        for original_indexing_element in index:
                            if isinstance(original_indexing_element, slice):
                                extra_dimensions += 1

                indexing_element = ctx.reshape(
                    indexing_element,
                    indexing_element.shape + (1,) * extra_dimensions,
                )
                to_concat.append(
                    ctx.broadcast_to(indexing_element, expanded_indexing_element_shape)
                )

        indices = ctx.concatenate(
            ctx.tensor(ctx.index_type(), indexing_element_shape + (len(to_concat),)),
            to_concat,
            axis=-1,
        )

    return indices


def fancy_indexing(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
) -> Conversion:
    """
    Convert fancy indexing operation.

    Args:
        ctx (Context):
            conversion context

        resulting_type (ConversionType):
            resulting type of the operation

        x (Conversion):
            tensor to fancy index

        index (Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]]):
            fancy index to use

    Returns:
        Conversion:
            result of fancy indexing operation
    """

    indices = generate_fancy_indices(
        ctx,
        resulting_type.shape,
        x,
        index,
        check_out_of_bounds=ctx.configuration.dynamic_indexing_check_out_of_bounds,
    )

    return ctx.operation(
        fhelinalg.FancyIndexOp,
        resulting_type,
        x.result,
        indices.result,
        original_bit_width=x.original_bit_width,
    )


def indexing(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
) -> Conversion:
    """
    Convert indexing operation.

    Args:
        ctx (Context):
            conversion context

        resulting_type (ConversionType):
            resulting type of the operation

        x (Conversion):
            tensor to index

        index (Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]]):
            index to use

    Returns:
        Conversion:
            result of indexing operation
    """

    assert resulting_type.is_encrypted == x.is_encrypted
    assert ctx.is_bit_width_compatible(resulting_type, x)

    index = list(index)
    while len(index) < len(x.shape):
        index.append(slice(None, None, None))

    is_fancy = any(
        (
            isinstance(indexing_element, (list, np.ndarray))
            or (isinstance(indexing_element, Conversion) and indexing_element.is_tensor)
        )
        for indexing_element in index
    )
    if is_fancy:
        return fancy_indexing(ctx, resulting_type, x, index)

    if resulting_type.shape == ():
        processed_index = []
        for indexing_element, dimension_size in zip(index, x.shape):
            assert isinstance(indexing_element, (int, np.integer, Conversion))

            indexing_element = process_indexing_element(
                ctx,
                indexing_element,
                dimension_size,
                check_out_of_bounds=ctx.configuration.dynamic_indexing_check_out_of_bounds,
            )
            if not isinstance(indexing_element, Conversion):
                indexing_element = ctx.constant(
                    ctx.index_type(),
                    indexing_element,
                )

            processed_index.append(indexing_element)

        return ctx.operation(
            tensor.ExtractOp,
            resulting_type,
            x.result,
            tuple(indexing_element.result for indexing_element in processed_index),
            original_bit_width=x.original_bit_width,
        )

    static_offsets: list[Any] = []
    static_sizes: list[Any] = []
    static_strides: list[Any] = []

    dynamic_offsets: list[Any] = []

    destroyed_dimensions = []
    for dimension, (indexing_element, dimension_size) in enumerate(zip(index, x.shape)):
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
                    check_out_of_bounds=ctx.configuration.dynamic_indexing_check_out_of_bounds,
                )
                if indexing_element.start is not None
                else (0 if stride > 0 else dimension_size - 1)
            )
        else:
            assert isinstance(indexing_element, (int, np.integer)) or (
                isinstance(indexing_element, Conversion) and indexing_element.is_scalar
            )
            destroyed_dimensions.append(dimension)

            size = 1
            stride = 1
            offset = process_indexing_element(
                ctx,
                indexing_element,
                dimension_size,
                check_out_of_bounds=ctx.configuration.dynamic_indexing_check_out_of_bounds,
            )

            if isinstance(offset, Conversion):
                dynamic_offsets.append(offset)
                offset = MlirShapedType.get_dynamic_size()

        static_offsets.append(offset)
        static_sizes.append(size)
        static_strides.append(stride)

    if len(destroyed_dimensions) == 0:
        return ctx.operation(
            tensor.ExtractSliceOp,
            resulting_type,
            x.result,
            tuple(item.result for item in dynamic_offsets),
            (),
            (),
            MlirDenseI64ArrayAttr.get(static_offsets),
            MlirDenseI64ArrayAttr.get(static_sizes),
            MlirDenseI64ArrayAttr.get(static_strides),
            original_bit_width=x.original_bit_width,
        )

    intermediate_shape = list(resulting_type.shape)
    for dimension in destroyed_dimensions:
        intermediate_shape.insert(dimension, 1)

    intermediate = ctx.operation(
        tensor.ExtractSliceOp,
        ctx.tensor(ctx.element_typeof(x), tuple(intermediate_shape)),
        x.result,
        tuple(item.result for item in dynamic_offsets),
        (),
        (),
        MlirDenseI64ArrayAttr.get(static_offsets),
        MlirDenseI64ArrayAttr.get(static_sizes),
        MlirDenseI64ArrayAttr.get(static_strides),
        original_bit_width=x.original_bit_width,
    )

    reassociaton = []

    current_intermediate_dimension = 0
    for _ in range(len(resulting_type.shape)):
        indices = [current_intermediate_dimension]
        while current_intermediate_dimension in destroyed_dimensions:
            current_intermediate_dimension += 1
            indices.append(current_intermediate_dimension)

        reassociaton.append(indices)
        current_intermediate_dimension += 1
    while current_intermediate_dimension < len(intermediate_shape):
        reassociaton[-1].append(current_intermediate_dimension)
        current_intermediate_dimension += 1

    return ctx.operation(
        tensor.CollapseShapeOp,
        resulting_type,
        intermediate.result,
        MlirArrayAttr.get(
            [
                MlirArrayAttr.get(
                    [MlirIntegerAttr.get(ctx.i(64).mlir, index) for index in indices],
                )
                for indices in reassociaton
            ],
        ),
        original_bit_width=x.original_bit_width,
    )
