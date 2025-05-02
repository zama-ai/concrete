"""
Conversion of min and max operations.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Union

import numpy as np

from ..context import Context
from ..conversion import Conversion, ConversionType


def min_max(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    axes: Union[int, np.integer, Sequence[Union[int, np.integer]]] = (),
    keep_dims: bool = False,
    *,
    operation: str,
) -> Conversion:
    """
    Convert min or max operation.

    Args:
        ctx (Context):
            conversion context

        resulting_type (ConversionType):
            resulting type of the operation

        x (Conversion):
            input of the operation

        axes (Union[int, np.integer, Sequence[Union[int, np.integer]]], default = ()):
            axes to reduce over

        keep_dims (bool, default = False):
            whether to keep the reduced axes

        operation (str):
            "min" or "max"

    Returns:
        Conversion:
            np.min or np.max on x depending on operation
    """

    # if the input is clear
    if x.is_clear:
        # raise error as computing min/max of clear values is not supported
        highlights = {
            x.origin: "value is clear",
            ctx.converting: f"but computing {operation} of clear values is not supported",
        }
        ctx.error(highlights)

    # if the value is scalar
    if x.is_scalar:
        # return it as it's the min/max
        return x

    # if axes is not specified, use all axes
    if axes is None:
        axes = []

    # compute the list of unique axes to reduce
    # empty list means reduce all axes
    axes = list(
        set(
            # if axes was a single integer, only it will be reduced
            [int(axes)]
            if isinstance(axes, (int, np.integer))
            # if axes was a sequence, every axis in it will be reduced
            else [int(axis) for axis in axes]
        )
    )

    # sanitize negative axis
    # `axis=-1` is the same as `axis=(-1 + x.ndim))`
    input_dimensions = len(x.shape)
    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] += input_dimensions
        assert 0 <= axes[i] < input_dimensions

    # if all axes are reduced
    if len(axes) == 0 or len(axes) == len(x.shape):
        # we flatten the input to use the `reduce` implementation
        x = ctx.flatten(x)

    # if the input is a vector
    if len(x.shape) == 1:
        # we reduce the vector to its min/max value
        result = reduce(ctx, ctx.element_typeof(resulting_type), x, operation=operation)

        # if the user wants to keep the reduced dimensions
        if keep_dims:
            # reshape the result into the resulting shape
            result = ctx.reshape(result, shape=resulting_type.shape)

        # return the result
        return result

    # if the reduce implementation is not used
    # mock implementation will be used instead
    #
    # the idea is to let numpy compute the indices that will be compared to obtain the result
    #
    # for example, if input.shape == (2, 3) and axis == 1
    # we'll be computing
    # [
    #     [
    #         {(0, 2, 0), (0, 0, 0), (0, 3, 0), (0, 1, 0)}
    #         {(0, 2, 1), (0, 0, 1), (0, 3, 1), (0, 1, 1)}
    #         {(0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 0, 2)}
    #     ]
    #     [
    #         {(1, 0, 0), (1, 3, 0), (1, 1, 0), (1, 2, 0)}
    #         {(1, 2, 1), (1, 0, 1), (1, 1, 1), (1, 3, 1)}
    #         {(1, 3, 2), (1, 2, 2), (1, 0, 2), (1, 1, 2)}
    #     ]
    # ]
    #
    # this means to compute output[0, 0] we need to compare
    # - input[0, 2, 0]
    # - input[0, 0, 0]
    # - input[0, 3, 0]
    # - input[0, 1, 0]
    #
    # or to compute output[1, 2] we need to compare
    # - input[1, 3, 2]
    # - input[1, 2, 2]
    # - input[1, 0, 2]
    # - input[1, 1, 2]
    #
    # notice that number of comparisons (say n) are always the same in each output cell
    # this opens up the possibility to use fancy indexing to extract n slices from the input
    # and compare the slices in a tree-like fashion to obtain the result
    #
    # in the end, we'll be performing
    #
    # slice1 = [
    #     [input[0, 2, 0], input[0, 2, 1], input[0, 1, 2]],
    #     [input[1, 0, 0], input[1, 2, 1], input[1, 3, 2]],
    # ]
    # slice2 = [
    #     [input[0, 0, 0], input[0, 0, 1], input[0, 2, 2]],
    #     [input[1, 3, 0], input[1, 0, 1], input[1, 2, 2]],
    # ]
    # slice3 = [
    #     [input[0, 3, 0], input[0, 3, 1], input[0, 3, 2]],
    #     [input[1, 1, 0], input[1, 1, 1], input[1, 0, 2]],
    # ]
    # slice4 = [
    #     [input[0, 1, 0], input[0, 1, 1], input[0, 0, 2]],
    #     [input[1, 2, 0], input[1, 3, 1], input[1, 1, 2]],
    # ]
    #
    # minimum_slice1_slice2 = np.minimum(slice1, slice2)
    # minimum_slice3_slice4 = np.minimum(slice3, slice4)
    #
    # result = np.minimum(minimum_slice1_slice2, minimum_slice3_slice4)
    #
    # notice that all slices have the same shape as the result

    class Mock:
        """
        Class to track accumulation of the operation.
        """

        # list of indices that have been accumulated
        indices: set[tuple[int, ...]]

        # initialize the mock with a starting index
        def __init__(self, index: tuple[int, ...]):
            self.indices = {index}

        # get the representation of the mock
        def __repr__(self) -> str:
            return f"{self.indices}"  # pragma: no cover

        # combine the indices of the mock with another mock into a new mock
        def combine(self, other: "Mock") -> "Mock":
            result = deepcopy(self)
            for index in other.indices:
                result.indices.add(index)
            return result

    # create the mock input
    #
    # [[[{(0, 0, 0)} {(0, 0, 1)} {(0, 0, 2)}]
    #   [{(0, 1, 0)} {(0, 1, 1)} {(0, 1, 2)}]
    #   [{(0, 2, 0)} {(0, 2, 1)} {(0, 2, 2)}]
    #   [{(0, 3, 0)} {(0, 3, 1)} {(0, 3, 2)}]]
    #
    #  [[{(1, 0, 0)} {(1, 0, 1)} {(1, 0, 2)}]
    #   [{(1, 1, 0)} {(1, 1, 1)} {(1, 1, 2)}]
    #   [{(1, 2, 0)} {(1, 2, 1)} {(1, 2, 2)}]
    #   [{(1, 3, 0)} {(1, 3, 1)} {(1, 3, 2)}]]]

    mock_input = []
    for index in np.ndindex(x.shape):
        mock_input.append(Mock(index))
    mock_input = np.array(mock_input).reshape(x.shape)

    # use numpy reduction to compute the mock output
    #
    # [[{(0, 2, 0), (0, 0, 0), (0, 3, 0), (0, 1, 0)}
    #   {(0, 2, 1), (0, 0, 1), (0, 3, 1), (0, 1, 1)}
    #   {(0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 0, 2)}]
    #  [{(1, 0, 0), (1, 3, 0), (1, 1, 0), (1, 2, 0)}
    #   {(1, 2, 1), (1, 0, 1), (1, 1, 1), (1, 3, 1)}
    #   {(1, 3, 2), (1, 2, 2), (1, 0, 2), (1, 1, 2)}]]

    mock_output = np.frompyfunc(lambda mock1, mock2: mock1.combine(mock2), 2, 1).reduce(
        mock_input,
        axis=tuple(axes),
        keepdims=keep_dims,
    )

    # extract a sample mock from the mock output
    sample_mock = mock_output.flat[0]

    # extract the indices of the sample mock
    sample_mock_indices = sample_mock.indices

    # compute number of comparisons
    number_of_comparisons = len(sample_mock_indices)

    # extract a sample index from the sample mock indices
    sample_mock_index = next(iter(sample_mock_indices))

    # compute the number of indices
    number_of_indices = len(sample_mock_index)

    # compute the shape of fancy indexing indices
    index_shape = resulting_type.shape

    # compute the fancy indices to extract for comparison
    to_compare = []
    for _ in range(number_of_comparisons):
        indices = []
        for _ in range(number_of_indices):
            index = np.zeros(index_shape, dtype=np.int64)  # type: ignore
            indices.append(index)
        to_compare.append(tuple(indices))
    for position in np.ndindex(mock_output.shape):
        mock_indices = list(mock_output[position].indices)
        for i in range(number_of_comparisons):
            for j in range(number_of_indices):
                to_compare[i][j][position] = mock_indices[i][j]  # type: ignore

    # to_compare will look like
    # [
    #     # for the first slice
    #     (
    #         [[0, 0, 0], [1, 1, 1]],  # i
    #         [[2, 2, 1], [0, 2, 3]],  # j
    #         [[0, 1, 2], [0, 1, 2]],  # k
    #     ),
    #     # for the second slice
    #     (
    #         [[0, 0, 0], [1, 1, 1]],  # i
    #         [[0, 0, 2], [3, 0, 2]],  # j
    #         [[0, 1, 2], [0, 1, 2]],  # k
    #     ),
    #     # for the third slice
    #     (
    #         [[0, 0, 0], [1, 1, 1]],  # i
    #         [[3, 3, 3], [1, 1, 0]],  # j
    #         [[0, 1, 2], [0, 1, 2]],  # k
    #     ),
    #     # for the fourth slice
    #     (
    #         [[0, 0, 0], [1, 1, 1]],  # i
    #         [[1, 1, 0], [2, 3, 1]],  # j
    #         [[0, 1, 2], [0, 1, 2]],  # k
    #     ),
    # ]

    # find the type of the slices
    slices_type = ctx.tensor(ctx.element_typeof(x), shape=resulting_type.shape)

    # extract the slices
    slices = []
    for index in to_compare:
        slices.append(ctx.index(slices_type, x, index))  # type: ignore

    # while there are more than 1 slices
    while len(slices) > 1:
        # pop the last two slices
        a = slices.pop()
        b = slices.pop()

        # compare the last two slices
        if operation == "min":
            c = ctx.minimum(resulting_type, a, b)
        else:
            c = ctx.maximum(resulting_type, a, b)

        # we need to set the original bit width manually
        # as minimum/maximum doesn't constraint their output bit width.
        c.set_original_bit_width(x.original_bit_width)

        # insert the slice back at the beginning of the slice queue
        slices.insert(0, c)

    # return the result
    return slices[0]


def reduce(
    ctx: Context,
    resulting_type: ConversionType,
    values: Conversion,
    *,
    operation: str,
) -> Conversion:
    """
    Reduce a vector of values to its min/max value.
    """

    # make sure the operation is valid
    assert operation in {"min", "max"}

    # make sure the value is valid
    assert values.is_tensor
    assert len(values.shape) == 1
    assert values.is_encrypted

    # make sure the resulting type is valid
    assert resulting_type.is_scalar
    assert resulting_type.is_encrypted

    # let's say the vector was [1, 4, 2, 3, 0]
    # and we're computing np.min(vector)

    # find the element type of the vector = fhe.uint3
    values_element_type = ctx.element_typeof(values)

    # find the middle of the vector = 2
    middle = values.shape[0] // 2

    # we'll be splitting the array into two halves
    # [1, 4] and [2, 3] in our case
    # then we'll compute np.minimum(first_half, second_half)
    # [1, 3] in our case
    # if the reduction is a scalar, we'll return it
    # otherwise, we'll reduce recursively until we obtain a scalar
    # 1 in our case

    # find the half type of the vector which is
    # fhe.tensor[fhe.uint3, 2] in the first iteration
    # fhe.uint3 in the last iteration
    half_type = (
        ctx.tensor(values_element_type, shape=(middle,)) if middle != 1 else values_element_type
    )

    # find the accumulated type of the vector which is
    # fhe.tensor[fhe.uint3, 2] in the first iteration
    # fhe.uint3 in the last iteration
    accumulated_type = (
        ctx.tensor(resulting_type, shape=(middle,)) if middle != 1 else resulting_type
    )

    # if there is only one element in each half (e.g., vector = [1, 3], halfs = [1], [3])
    if middle == 1:
        # extract the first element in the vector
        first_half = ctx.index(half_type, values, index=[0])
        # extract the second element in the vector
        second_half = ctx.index(half_type, values, index=[1])
    else:
        # extract the elements from 0 to middle as the first half
        first_half = ctx.index(half_type, values, index=[slice(0, middle)])
        # extract the elements from middle to 2*middle as the second half
        second_half = ctx.index(half_type, values, index=[slice(middle, 2 * middle)])

    # compare halfs
    if operation == "min":
        # [1, 3] in the first iteration
        # 1 in the last iteration
        reduced = ctx.minimum(accumulated_type, first_half, second_half)
    else:
        reduced = ctx.maximum(accumulated_type, first_half, second_half)

    # set the original bit width of the reduced so the following operation work as intended
    # this is required here since ctx.minimum and ctx.maximum does not constraint output bit width
    reduced.set_original_bit_width(values.original_bit_width)

    result = (
        # if reduced value is a scalar, we end the recursion
        reduced
        if reduced.is_scalar
        # otherwise, we reduce the result of comparison of halfs
        else reduce(ctx, resulting_type, reduced, operation=operation)
    )

    # if we have one more element that wasn't in the halfs
    if values.shape[0] % 2 == 1:
        # we extract it
        last_value = ctx.index(values_element_type, values, index=[-1])
        # and compare it with the result we obtained from the halfs
        result = (
            ctx.minimum(resulting_type, result, last_value)
            if operation == "min"
            else ctx.maximum(resulting_type, result, last_value)
        )
        # again, we need to set the original bit width
        result.set_original_bit_width(values.original_bit_width)

    # here is the visualization of the algorithm
    #
    # [ 1, 4, 2, 3, 0 ]
    #
    #  [1, 4][2, 3],0
    #     \   /     |
    #     [1, 3]    |
    #        \     /
    #        1    /
    #         \  /
    #          0
    #
    # it has O(log(n)) - 1 number of tensor comparisons (of sizes n/2, n/4, ...)
    # and up to 1 + O(log(n)) number of scalar comparisons (depending on the oddity of n/2, n/4, ..)

    return result
