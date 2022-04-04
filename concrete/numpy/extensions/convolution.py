"""
Declaration of `conv2d` function.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..representation import Node
from ..tracing import Tracer
from ..values import EncryptedTensor

SUPPORTED_AUTO_PAD = {
    "NOTSET",
}


def conv2d(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]] = None,
    pads: Union[Tuple[int, int, int, int], List[int]] = (0, 0, 0, 0),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    dilations: Union[Tuple[int, int], List[int]] = (1, 1),
    auto_pad: str = "NOTSET",
) -> Union[np.ndarray, Tracer]:
    """
    Trace or evaluate 2D convolution.

    Args:
        x (Union[np.ndarray, Tracer]):
            input of shape (N, C, H, W)

        weight (Union[np.ndarray, Tracer]):
            kernel of shape (F, C, H, W)

        bias (Optional[Union[np.ndarray, Tracer]], default = None):
            bias of shape (F,)

        pads (Union[Tuple[int, int, int, int], List[int]], default = (0, 0, 0, 0)):
            padding over each height and width (H_beg, W_beg, H_end, W_end)

        strides (Union[Tuple[int, int], List[int]], default = (1, 1)):
            stride over height and width

        dilations (Union[Tuple[int, int], List[int]], default = (1, 1)):
            dilation over height and width

        auto_pad (str, default = "NOTSET"):
            padding strategy

    Returns:
        Union[np.ndarray, Tracer]:
            evaluation result or traced computation

    Raises:
        ValueError:
            if arguments are not appropriate
    """

    if auto_pad not in SUPPORTED_AUTO_PAD:
        raise ValueError(f"Auto pad should be in {SUPPORTED_AUTO_PAD} but it's {repr(auto_pad)}")

    if len(pads) != 4:
        raise ValueError(
            f"Pads should be of form "
            f"(height_begin_pad, width_begin_pad, height_end_pad, width_end_pad) "
            f"but it's {pads}"
        )
    if len(strides) != 2:
        raise ValueError(
            f"Strides should be of form (height_stride, width_stride) but it's {strides}"
        )
    if len(dilations) != 2:
        raise ValueError(
            f"Dilations should be of form "
            f"(height_dilation, width_dilation) "
            f"but it's {dilations}"
        )

    if isinstance(x, Tracer):
        return _trace_conv2d(x, weight, bias, pads, strides, dilations)

    if not isinstance(weight, np.ndarray):
        raise ValueError("Weight should be of type np.ndarray for evaluation")

    if bias is not None and not isinstance(bias, np.ndarray):
        raise ValueError("Bias should be of type np.ndarray for evaluation")

    bias = np.zeros(weight.shape[0]) if bias is None else bias
    return _evaluate_conv2d(x, weight, bias, pads, strides, dilations)


def _trace_conv2d(
    x: Tracer,
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, int, int, int], List[int]],
    strides: Union[Tuple[int, int], List[int]],
    dilations: Union[Tuple[int, int], List[int]],
) -> Tracer:
    """
    Trace 2D convolution.

    Args:
        x (Tracer):
            input of shape (N, C, H, W)

        weight (Union[np.ndarray, Tracer]):
            kernel of shape (F, C, H, W)

        bias (Optional[Union[np.ndarray, Tracer]]):
            bias of shape (F,)

        pads (Union[Tuple[int, int, int, int], List[int]]):
            padding over each axis (H_beg, W_beg, H_end, W_end)

        strides (Union[Tuple[int, int], List[int]]):
            stride over height and width

        dilations (Union[Tuple[int, int], List[int]]):
            dilation over height and width

    Returns:
        Tracer:
            traced computation
    """

    if x.output.ndim != 4:
        raise ValueError(
            f"Input should be of shape (N, C, H, W) but it's of shape {x.output.shape}",
        )

    weight = weight if isinstance(weight, Tracer) else Tracer(Node.constant(weight), [])

    if weight.output.ndim != 4:
        raise ValueError(
            f"Weight should be of shape (F, C, H, W) but it's of shape {weight.output.shape}",
        )

    input_values = [x.output, weight.output]
    inputs = [x, weight]

    if bias is not None:
        bias = bias if isinstance(bias, Tracer) else Tracer(Node.constant(bias), [])
        input_values.append(bias.output)
        inputs.append(bias)
        if bias.output.ndim != 1:
            raise ValueError(
                f"Bias should be of shape (F,) but it's of shape {bias.output.shape}",
            )

    input_n, _, input_h, input_w = x.output.shape
    weight_f, _, weight_h, weight_w = weight.output.shape

    pads_h = pads[0] + pads[2]
    pads_w = pads[1] + pads[3]

    output_h = math.floor((input_h + pads_h - dilations[0] * (weight_h - 1) - 1) / strides[0]) + 1
    output_w = math.floor((input_w + pads_w - dilations[1] * (weight_w - 1) - 1) / strides[1]) + 1

    output_shape = (input_n, weight_f, output_h, output_w)
    output_value = EncryptedTensor(dtype=x.output.dtype, shape=output_shape)

    computation = Node.generic(
        "conv2d",
        input_values,
        output_value,
        _evaluate_conv2d,
        args=() if bias is not None else (np.zeros(weight.output.shape[0], dtype=np.int64),),
        kwargs={"pads": pads, "strides": strides, "dilations": dilations},
    )
    return Tracer(computation, inputs)


def _evaluate_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, int, int, int], List[int]],  # pylint: disable=unused-argument
    strides: Union[Tuple[int, int], List[int]],
    dilations: Union[Tuple[int, int], List[int]],
) -> np.ndarray:
    """
    Evaluate 2D convolution.

    Args:
        x (np.ndarray):
            input of shape (N, C, H, W)

        weight (np.ndarray):
            kernel of shape (F, C, H, W)

        bias (np.ndarray):
            bias of shape (F,)

        pads (Union[Tuple[int, int, int, int], List[int]]):
            padding over each axis (H_beg, W_beg, H_end, W_end)

        strides (Union[Tuple[int, int], List[int]]):
            stride over height and width

        dilations (Union[Tuple[int, int], List[int]]):
            dilation over height and width

    Returns:
        np.ndarray:
            result of the convolution
    """

    # pylint: disable=no-member

    dtype = (
        torch.float64
        if np.issubdtype(x.dtype, np.floating)
        or np.issubdtype(weight.dtype, np.floating)
        or np.issubdtype(bias.dtype, np.floating)
        else torch.long
    )

    return torch.conv2d(
        torch.tensor(x, dtype=dtype),
        torch.tensor(weight, dtype=dtype),
        torch.tensor(bias, dtype=dtype),
        stride=strides,
        dilation=dilations,
    ).numpy()

    # pylint: enable=no-member
