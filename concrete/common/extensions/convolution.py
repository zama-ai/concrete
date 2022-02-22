"""This file contains tracers for convolution operations."""

from typing import List, Optional, Tuple, Union, cast

import numpy as np

from ...numpy.tracing import NPConstant, NPTracer
from ..representation.intermediate import Conv2D
from ..tracing.base_tracer import BaseTracer

SUPPORTED_AUTO_PAD = [
    "NOTSET",
]


def conv2d(
    x: Union[np.ndarray, BaseTracer],
    weight: Union[np.ndarray, BaseTracer],
    bias: Optional[Union[np.ndarray, BaseTracer]] = None,
    pads: Union[Tuple[int, int, int, int], List[int]] = (0, 0, 0, 0),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    dilations: Union[Tuple[int, int], List[int]] = (1, 1),
    auto_pad: str = "NOTSET",
) -> Union[np.ndarray, NPTracer]:
    """Trace or evaluate 2D convolution.

    Args:
        x (Union[np.ndarray, BaseTracer]): Input of shape (NxCxHxW)
        weight (Union[np.ndarray, BaseTracer]): Weight (kernel) of shape (FxCxHxW)
        bias (Optional[Union[np.ndarray, BaseTracer]], optional): Bias vector of size (F).
            Defaults to None.
        pads (Union[Tuple[int, int, int, int], List[int]], optional): Padding over each axis
            (H_beg, W_beg, H_end, W_end). Defaults to (0, 0, 0, 0).
        strides (Union[Tuple[int, int], List[int]], optional): Stride over each axis
            (height and width). Defaults to (1, 1).
        dilations (Union[Tuple[int, int], List[int]], optional): Dilation over each axis
            (height and width). Defaults to (1, 1).
        auto_pad (str, optional): Padding strategy. Defaults to "NOTSET".

    Raises:
        ValueError: If one argument isn't in the range of expected values.
        TypeError: If one argument isn't of the appropriate type.

    Returns:
        Union[np.ndarray, BaseTracer]: Evaluation result, or traced computation
    """
    if auto_pad not in SUPPORTED_AUTO_PAD:
        raise ValueError("invalid auto_pad is specified")

    if not isinstance(x, (np.ndarray, BaseTracer)):
        raise TypeError(f"input x must be an ndarray, or a BaseTracer, not a {type(x)}")
    if not isinstance(weight, (np.ndarray, BaseTracer)):
        raise TypeError(f"weight must be an ndarray, or a BaseTracer, not a {type(weight)}")
    if not isinstance(bias, (np.ndarray, BaseTracer, type(None))):
        raise TypeError(f"bias must be an ndarray, a BaseTracer, or None, not a {type(bias)}")
    if not isinstance(pads, (tuple, list)):
        raise TypeError(f"padding must be a tuple, or list, not a {type(pads)}")
    if not isinstance(strides, (tuple, list)):
        raise TypeError(f"strides must be a tuple, or list, not a {type(strides)}")
    if not isinstance(dilations, (tuple, list)):
        raise TypeError(f"dilations must be a tuple, or list, not a {type(dilations)}")

    if len(pads) != 4:
        raise ValueError(
            f"padding should be of the form (pad_height_begin, pad_width_begin, pad_height_end, "
            f" pad_width_end), but got {type(pads)} of length {len(pads)}"
        )
    if len(strides) != 2:
        raise ValueError(
            f"strides should be of the form (stride_height, stride_width), but got {type(strides)}"
            f" of length {len(strides)}"
        )
    if len(dilations) != 2:
        raise ValueError(
            f"dilations should be of the form (dilation_height, dilation_width), but got"
            f" {type(dilations)} of length {len(dilations)}"
        )

    assert len(x.shape) == 4, f"input x should have size (N x C x H x W), not {x.shape}"
    assert len(weight.shape) == 4, f"weight should have size (F x C x H x W), not {weight.shape}"
    if bias is not None:
        assert len(bias.shape) == 1, f"bias should have size (F), not {bias.shape}"

    if isinstance(x, BaseTracer):
        return _trace_conv2d(x, weight, bias, pads, strides, dilations)
    # X is an ndarray
    bias = np.zeros(weight.shape[0]) if bias is None else bias
    # For mypy
    weight = cast(np.ndarray, weight)
    bias = cast(np.ndarray, bias)
    return _evaluate_conv2d(x, weight, bias, pads, strides, dilations)


def _trace_conv2d(
    x: BaseTracer,
    weight: Union[np.ndarray, BaseTracer],
    bias: Optional[Union[np.ndarray, BaseTracer]],
    pads: Union[Tuple[int, int, int, int], List[int]],
    strides: Union[Tuple[int, int], List[int]],
    dilations: Union[Tuple[int, int], List[int]],
) -> NPTracer:
    """Trace 2D convolution.

    Args:
        x (BaseTracer): Input of shape (NxCxHxW)
        weight (Union[np.ndarray, BaseTracer]): Weight (kernel) of shape (FxCxHxW)
        bias (Optional[Union[np.ndarray, BaseTracer]]): Bias vector of size (F)
        pads (Union[Tuple[int, int, int, int], List[int]]): Padding over each
            axis (H_beg, W_beg, H_end, W_end)
        strides (Union[Tuple[int, int], List[int]]): Stride over each
            axis (height and width)
        dilations (Union[Tuple[int, int], List[int]]): Dilation over each
            axis (height and width)

    Returns:
        BaseTracer: Traced computation
    """
    weight_tracer = (
        weight if isinstance(weight, BaseTracer) else NPTracer([], NPConstant(weight), 0)
    )
    inputs = [x.output, weight_tracer.output]
    output_tracer_inputs = [x, weight_tracer]
    if bias is not None:
        bias_tracer = bias if isinstance(bias, BaseTracer) else NPTracer([], NPConstant(bias), 0)
        inputs.append(bias_tracer.output)
        # For mypy
        bias = cast(BaseTracer, bias_tracer)
        output_tracer_inputs.append(bias)

    traced_computation = Conv2D(inputs, x.output.dtype, pads, strides, dilations)
    output_tracer = x.__class__(
        output_tracer_inputs, traced_computation=traced_computation, output_idx=0
    )
    # For mypy
    assert isinstance(output_tracer, NPTracer)
    return output_tracer


def _evaluate_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, int, int, int], List[int]],
    strides: Union[Tuple[int, int], List[int]],
    dilations: Union[Tuple[int, int], List[int]],
) -> np.ndarray:
    """Evaluate 2D convolution.

    Args:
        x (np.ndarray): Input of shape (NxCxHxW)
        weight (np.ndarray): Weight (kernel) of shape (FxCxHxW)
        bias (np.ndarray): Bias vector of size (F)
        pads (Union[Tuple[int, int, int, int], List[int]]): Padding over each
            axis (H_beg, W_beg, H_end, W_end)
        strides (Union[Tuple[int, int], List[int]]): Stride over each axis (height and width)
        dilations (Union[Tuple[int, int], List[int]]): Dilation over each axis (height and width)

    Returns:
        np.ndarray: Result of the convolution of shape (NxCxHxW)
    """
    return Conv2D.evaluate_conv2d(x, weight, bias, pads, strides, dilations)
