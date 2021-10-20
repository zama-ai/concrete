"""Helpers for indexing functionality."""

from typing import Tuple, Union


def format_indexing_element(indexing_element: Union[int, slice]) -> str:
    """Format an indexing element.

    This is required mainly for slices. The reason is that string representation of slices
    are very long and verbose. To give an example, `x[:, 2:]` will have the following index
    `[slice(None, None, None), slice(2, None, None)]` if printed naively. With this helper,
    it will be formatted as `[:, 2:]`.

    Args:
        indexing_element (Union[int, slice]): indexing element to be formatted

    Returns:
        str: formatted element
    """

    result = ""
    if isinstance(indexing_element, slice):
        if indexing_element.start is not None:
            result += str(indexing_element.start)
        result += ":"
        if indexing_element.stop is not None:
            result += str(indexing_element.stop)
        if indexing_element.step is not None:
            result += ":"
            result += str(indexing_element.step)
    else:
        result += str(indexing_element)
    return result.replace("\n", " ")


def validate_index(
    index: Union[int, slice, Tuple[Union[int, slice], ...]],
) -> Tuple[Union[int, slice], ...]:
    """Make sure index is valid and convert it to the tuple form.

    For example in `x[2]`, `index` is passed as `2`.
    To make it easier to work with, this function converts index to `(2,)`.

    Args:
        index (Union[int, slice, Tuple[Union[int, slice], ...]]): index to validate, improve
            and return

    Returns:
        Tuple[Union[int, slice], ...]: validated and improved index
    """

    if not isinstance(index, tuple):
        index = (index,)

    for indexing_element in index:
        valid = isinstance(indexing_element, (int, slice))

        if isinstance(indexing_element, slice):
            if (
                not (indexing_element.start is None or isinstance(indexing_element.start, int))
                or not (indexing_element.stop is None or isinstance(indexing_element.stop, int))
                or not (indexing_element.step is None or isinstance(indexing_element.step, int))
            ):
                valid = False

        if not valid:
            raise TypeError(
                f"Only integers and integer slices can be used for indexing "
                f"but you tried to use {format_indexing_element(indexing_element)} for indexing"
            )

    return index


def determine_output_shape(
    input_shape: Tuple[int, ...],
    index: Tuple[Union[int, slice], ...],
) -> Tuple[int, ...]:
    """Determine the output shape from the input shape and the index.

    e.g., for `input_shape=(3, 2)` and `index=(:, 0)`, returns `(3,)`
          for `input_shape=(4, 3, 2)` and `index=(2:,)`, returns `(2, 3, 2)`

    Args:
        input_shape (Tuple[int, ...]): shape of the input tensor that is indexed
        index (Tuple[Union[int, slice], ...]): desired and validated index

    Returns:
        Tuple[int, ...]: shape of the result of indexing
    """

    indexing_elements = [format_indexing_element(indexing_element) for indexing_element in index]
    index_str = f"[{', '.join(indexing_elements)}]"

    if len(index) > len(input_shape):
        raise ValueError(
            f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
            f"as the index has more elements than the number of dimensions of the tensor"
        )

    # indexing (3, 4, 5) with [1] is the same as indexing it with [1, :, :]
    # indexing (3, 4, 5) with [1, 2] is the same as indexing it with [1, 2, :]

    # so let's replicate that behavior to make the rest of the code generic
    index += (slice(None, None, None),) * (len(input_shape) - len(index))

    output_shape = []
    for dimension, (indexing_element, dimension_size) in enumerate(zip(index, input_shape)):
        if isinstance(indexing_element, int):  # indexing removes the dimension
            indexing_element = (
                indexing_element if indexing_element >= 0 else indexing_element + dimension_size
            )
            if not 0 <= indexing_element < dimension_size:
                raise ValueError(
                    f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
                    f"because index is out of range for dimension {dimension}"
                )
        elif isinstance(indexing_element, slice):  # indexing possibly shrinks the dimension
            output_shape.append(
                determine_new_dimension_size(
                    indexing_element,
                    dimension_size,
                    dimension,
                    input_shape,
                    index_str,
                )
            )

    return tuple(output_shape)


def sanitize_start_index(
    start: int,
    dimension_size: int,
    # the rest is used for detailed exception message
    dimension: int,
    input_shape: Tuple[int, ...],
    index_str: str,
) -> int:
    """Sanitize and check start index of a slice.

    Args:
        start (int): start index being sanitized
        dimension_size (int): size of the dimension the slice is applied to
        dimension (int): index of the dimension being sliced (for better messages)
        input_shape (Tuple[int, ...]): shape of the whole input (for better messages)
        index_str (str): string representation of the whole index (for better messages)

    Returns:
        int: sanitized start index
    """

    start = start if start >= 0 else start + dimension_size
    if not 0 <= start < dimension_size:
        raise ValueError(
            f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
            f"because start index is out of range for dimension {dimension}"
        )
    return start


def sanitize_stop_index(
    stop: int,
    dimension_size: int,
    # the rest is used for detailed exception message
    dimension: int,
    input_shape: Tuple[int, ...],
    index_str: str,
) -> int:
    """Sanitize and check stop index of a slice.

    Args:
        stop (int): stop index being sanitized
        dimension_size (int): size of the dimension the slice is applied to
        dimension (int): index of the dimension being sliced (for better messages)
        input_shape (Tuple[int, ...]): shape of the whole input (for better messages)
        index_str (str): string representation of the whole index (for better messages)

    Returns:
        int: sanitized stop index
    """

    stop = stop if stop >= 0 else stop + dimension_size
    if not 0 <= stop <= dimension_size:
        raise ValueError(
            f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
            f"because stop index is out of range for dimension {dimension}"
        )
    return stop


def determine_new_dimension_size(
    slice_: slice,
    dimension_size: int,
    # the rest is used for detailed exception message
    dimension: int,
    input_shape: Tuple[int, ...],
    index_str: str,
) -> int:
    """Determine the new size of a dimension from the old size and the slice applied to it.

    e.g., for `slice_=1:4` and `dimension_size=5`, returns `3`
          for `slice_=::-1` and `dimension_size=5`, returns `5`

    You may want to check this page to learn more about how this function works
    https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing

    Args:
        slice_ (slice): slice being applied to the dimension
        dimension_size (int): size of the dimension the slice is applied to
        dimension (int): index of the dimension being sliced (for better messages)
        input_shape (Tuple[int, ...]): shape of the whole input (for better messages)
        index_str (str): string representation of the whole index (for better messages)

    Returns:
        int: new size of the dimension
    """

    step = slice_.step if slice_.step is not None else 1

    if step > 0:
        start = slice_.start if slice_.start is not None else 0
        stop = slice_.stop if slice_.stop is not None else dimension_size

        start = sanitize_start_index(start, dimension_size, dimension, input_shape, index_str)
        stop = sanitize_stop_index(stop, dimension_size, dimension, input_shape, index_str)

        if start >= stop:
            raise ValueError(
                f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
                f"because start index is not less than stop index for dimension {dimension}"
            )

        size_before_stepping = stop - start
    elif step < 0:
        start = slice_.start if slice_.start is not None else dimension_size - 1
        stop = slice_.stop

        start = sanitize_start_index(start, dimension_size, dimension, input_shape, index_str)

        if stop is None:
            # this is a weird case but it works as expected
            # the issue is that it's impossible to slice whole vector reversed
            # with a stop value different than none

            # if `x.shape == (6,)` the only one that works is `x[::-1].shape == (6,)`
            # here is what doesn't work (and this is expected it's just weird)
            #
            # ...
            # `x[:-2:-1].shape == (1,)`
            # `x[:-1:-1].shape == (0,)` (note that this is a hard error for us)
            # `x[:0:-1].shape == (5,)`
            # `x[:1:-1].shape == (4,)`
            # ...

            size_before_stepping = start + 1
        else:
            stop = sanitize_stop_index(stop, dimension_size, dimension, input_shape, index_str)

            if stop >= start:
                raise ValueError(
                    f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
                    f"because step is negative and "
                    f"stop index is not less than start index for dimension {dimension}"
                )

            size_before_stepping = start - stop
    else:
        raise ValueError(
            f"Tensor of shape {input_shape} cannot be indexed with {index_str} "
            f"because step is zero for dimension {dimension}"
        )

    quotient = size_before_stepping // abs(step)
    remainder = size_before_stepping % abs(step)

    return quotient + (remainder != 0)
