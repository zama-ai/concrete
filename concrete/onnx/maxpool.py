"""
Tracing and evaluation of maxpool function.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..numpy.internal.utils import assert_that
from ..numpy.representation import Node
from ..numpy.tracing import Tracer
from ..numpy.values import Value

# pylint: disable=too-many-branches,too-many-statements


AVAILABLE_AUTO_PAD = {
    "NOTSET",
    "SAME_UPPER",
    "SAME_LOWER",
    "VALID",
}

AVAILABLE_CEIL_MODE = {
    0,
    1,
}

AVAILABLE_STORAGE_ORDER = {
    0,
    1,
}


SUPPORTED_AUTO_PAD = {
    "NOTSET",
}

SUPPORTED_CEIL_MODE = {
    0,
}

SUPPORTED_STORAGE_ORDER = {
    0,
}


# pylint: disable=no-member

_EVALUATORS = {
    1: torch.max_pool1d,
    2: torch.max_pool2d,
    3: torch.max_pool3d,
}

# pylint: enable=no-member


def maxpool(
    x: Union[np.ndarray, Tracer],
    kernel_shape: Union[Tuple[int, ...], List[int]],
    strides: Optional[Union[Tuple[int, ...], List[int]]] = None,
    auto_pad: str = "NOTSET",
    pads: Optional[Union[Tuple[int, ...], List[int]]] = None,
    dilations: Optional[Union[Tuple[int, ...], List[int]]] = None,
    ceil_mode: int = 0,
    storage_order: int = 0,
) -> Union[np.ndarray, Tracer]:
    """
    Evaluate or trace MaxPool operation.

    Refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool for more info.

    Args:
        x (Union[np.ndarray, Tracer]):
            input of shape (N, C, D1, ..., DN)

        kernel_shape (Union[Tuple[int, ...], List[int]]):
            shape of the kernel

        strides (Optional[Union[Tuple[int, ...], List[int]]]):
            stride along each spatial axis
            set to 1 along each spatial axis if not set

        auto_pad (str, default = "NOTSET"):
            padding strategy

        pads (Optional[Union[Tuple[int, ...], List[int]]]):
            padding for the beginning and ending along each spatial axis
            (D1_begin, D2_begin, ..., D1_end, D2_end, ...)
            set to 0 along each spatial axis if not set

        dilations (Optional[Union[Tuple[int, ...], List[int]]]):
            dilation along each spatial axis
            set to 1 along each spatial axis if not set

        ceil_mode (int, default = 1):
            ceiling mode

        storage_order (int, default = 0):
            storage order, 0 for row major, 1 for column major

    Raises:
        TypeError:
            if arguments are inappropriately typed

        ValueError:
            if arguments are inappropriate

        NotImplementedError:
            if desired operation is not supported yet

    Returns:
        Union[np.ndarray, Tracer]:
            maxpool over the input or traced computation
    """

    def check_value_is_a_tuple_or_list_of_ints_of_size(value_name, value, size) -> Tuple[int, ...]:
        if isinstance(value, list):
            value = tuple(value)

        if not isinstance(value, tuple):
            message = (
                f"Expected {value_name} to be a tuple or a list but it's {type(value).__name__}"
            )
            raise TypeError(message)

        for element in value:
            if not isinstance(element, int):
                message = (
                    f"Expected {value_name} to consist of integers "
                    f"but it has an element of type {type(element).__name__}"
                )
                raise TypeError(message)

        if len(value) != size:
            message = f"Expected {value_name} to have {size} elements but it has {len(value)}"
            raise ValueError(message)

        return value

    # check x

    if isinstance(x, list):  # pragma: no cover
        try:
            x = np.array(x)
        except Exception:  # pylint: disable=broad-except
            pass

    if isinstance(x, np.ndarray):
        if not (
            np.issubdtype(x.dtype, np.integer)
            or np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.bool_)
        ):
            message = (
                f"Expected input elements to be of type np.integer, np.floating, or np.bool_ "
                f"but it's {type(x.dtype).__name__}"
            )
            raise TypeError(message)
    elif not isinstance(x, Tracer):
        message = (
            f"Expected input to be of type np.ndarray or Tracer "
            f"but it's {type(auto_pad).__name__}"
        )
        raise TypeError(message)

    if x.ndim < 3:
        message = (
            f"Expected input to have at least 3 dimensions (N, C, D1, ...) "
            f"but it only has {x.ndim}"
        )
        raise ValueError(message)

    if x.ndim > 5:
        message = f"{x.ndim - 2}D maximum pooling is not supported yet"
        raise NotImplementedError(message)

    # check kernel_shape

    kernel_shape = check_value_is_a_tuple_or_list_of_ints_of_size(
        "kernel_shape", kernel_shape, x.ndim - 2
    )

    # check strides

    if strides is None:
        strides = (1,) * (x.ndim - 2)

    strides = check_value_is_a_tuple_or_list_of_ints_of_size("strides", strides, x.ndim - 2)

    # check auto_pad

    if not isinstance(auto_pad, str):
        message = f"Expected auto_pad to be of type str but it's {type(auto_pad).__name__}"
        raise TypeError(message)

    if auto_pad not in AVAILABLE_AUTO_PAD:
        message = (
            f"Expected auto_pad to be one of "
            f"{', '.join(sorted(AVAILABLE_AUTO_PAD))} "
            f"but it's {auto_pad}"
        )
        raise ValueError(message)

    if auto_pad not in SUPPORTED_AUTO_PAD:
        message = f"Desired auto_pad of {auto_pad} is not supported yet"
        raise NotImplementedError(message)

    # check pads

    if pads is None:
        pads = (0,) * (2 * (x.ndim - 2))

    pads = check_value_is_a_tuple_or_list_of_ints_of_size("pads", pads, 2 * (x.ndim - 2))

    for i in range(len(pads) // 2):
        pad_begin = pads[i]
        pad_end = pads[i + len(pads) // 2]
        if pad_begin != pad_end:
            message = f"Desired pads of {pads} is not supported yet because of uneven padding"
            raise NotImplementedError(message)

    # check dilations

    if dilations is None:
        dilations = (1,) * (x.ndim - 2)

    dilations = check_value_is_a_tuple_or_list_of_ints_of_size("dilations", dilations, x.ndim - 2)

    # check ceil_mode

    if not isinstance(ceil_mode, int):
        message = f"Expected ceil_mode to be of type int but it's {type(ceil_mode).__name__}"
        raise TypeError(message)

    if ceil_mode not in AVAILABLE_CEIL_MODE:
        message = (
            f"Expected ceil_mode to be one of "
            f"{', '.join(sorted(str(x) for x in AVAILABLE_CEIL_MODE))} "
            f"but it's {ceil_mode}"
        )
        raise ValueError(message)

    if ceil_mode not in SUPPORTED_CEIL_MODE:
        message = f"Desired ceil_mode of {ceil_mode} is not supported yet"
        raise NotImplementedError(message)

    # check storage_order

    if not isinstance(storage_order, int):
        message = (
            f"Expected storage_order to be of type int but it's {type(storage_order).__name__}"
        )
        raise TypeError(message)

    if storage_order not in AVAILABLE_STORAGE_ORDER:
        message = (
            f"Expected storage_order to be one of "
            f"{', '.join(sorted(str(x) for x in AVAILABLE_STORAGE_ORDER))} "
            f"but it's {storage_order}"
        )
        raise ValueError(message)

    if storage_order not in SUPPORTED_STORAGE_ORDER:
        message = f"Desired storage_order of {storage_order} is not supported yet"
        raise NotImplementedError(message)

    # trace or evaluate
    return _trace_or_evaluate(x, kernel_shape, strides, pads, dilations, ceil_mode == 1)


def _trace_or_evaluate(
    x: Union[np.ndarray, Tracer],
    kernel_shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    pads: Tuple[int, ...],
    dilations: Tuple[int, ...],
    ceil_mode: bool,
):
    if not isinstance(x, Tracer):
        return _evaluate(x, kernel_shape, strides, pads, dilations, ceil_mode == 1)

    result = _evaluate(np.zeros(x.shape), kernel_shape, strides, pads, dilations, ceil_mode == 1)
    resulting_value = Value.of(result)

    resulting_value.is_encrypted = x.output.is_encrypted
    resulting_value.dtype = x.output.dtype

    computation = Node.generic(
        "maxpool",
        [x.output],
        resulting_value,
        _evaluate,
        kwargs={
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "ceil_mode": ceil_mode,
        },
    )
    return Tracer(computation, [x])


def _evaluate(
    x: np.ndarray,
    kernel_shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    pads: Tuple[int, ...],
    dilations: Tuple[int, ...],
    ceil_mode: bool,
) -> np.ndarray:
    # pylint: disable=no-member

    dims = x.ndim - 2
    assert_that(dims in {1, 2, 3})

    evaluator = _EVALUATORS[dims]
    result = (
        evaluator(
            torch.from_numpy(x.astype(np.float64)),  # torch only supports float maxpools
            kernel_shape,
            strides,
            pads[: len(pads) // 2],
            dilations,
            ceil_mode,
        )
        .numpy()
        .astype(x.dtype)
    )

    # pylint: enable=no-member

    return result
