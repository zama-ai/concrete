"""
Convolution operations' tracing and evaluation.
"""

import math
from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np
import torch

from ..numpy.internal.utils import assert_that
from ..numpy.representation import Node
from ..numpy.tracing import Tracer
from ..numpy.values import EncryptedTensor

SUPPORTED_AUTO_PAD = {
    "NOTSET",
}


# pylint: disable=too-many-branches,too-many-statements


def conv(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]] = None,
    pads: Optional[Union[Tuple[int, ...], List[int]]] = None,
    strides: Optional[Union[Tuple[int, ...], List[int]]] = None,
    dilations: Optional[Union[Tuple[int, ...], List[int]]] = None,
    kernel_shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
    group: int = 1,
    auto_pad: str = "NOTSET",
) -> Union[np.ndarray, Tracer]:
    """
    Trace and evaluate convolution operations.

    Refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv for more info.

    Args:
        x (Union[np.ndarray, Tracer]): input of shape (N, C, D1, ..., DN)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C / group, K1, ..., KN)
        bias (Optional[Union[np.ndarray, Tracer]], optional): bias of shape (F,). Defaults to None.
        pads (Optional[Union[Tuple[int, ...], List[int]]], optional):
            padding for the beginning and ending along each spatial axis
            (D1_begin, D2_begin, ..., D1_end, D2_end, ...).
            Will be set to 0 along each spatial axis if not set.
        strides (Optional[Union[Tuple[int, ...], List[int]]], optional):
            stride along each spatial axis. Will be set to 1 along each spatial axis if not set.
        dilations (Optional[Union[Tuple[int, ...], List[int]]], optional):
            dilation along each spatial axis. Will be set to 1 along each spatial axis if not set.
        kernel_shape (Optional[Union[Tuple[int, ...], List[int]]], optional):
            shape of the convolution kernel. Inferred from input weight if not present
        group (int, optional):
            number of groups input channels and output channels are divided into. Defaults to 1.
        auto_pad (str, optional): padding strategy. Defaults to "NOTSET".

    Raises:
        ValueError: if arguments are not appropriate
        TypeError: unexpected types
        NotImplementedError: a convolution that we don't support

    Returns:
        Union[np.ndarray, Tracer]: evaluation result or traced computation
    """
    if kernel_shape is not None and (
        (weight.ndim - 2) != len(kernel_shape) or not np.all(weight.shape[2:] == kernel_shape)
    ):
        message = f"expected kernel_shape to be {weight.shape[2:]}, but got {kernel_shape}"
        raise ValueError(message)

    if isinstance(x, np.ndarray):
        if not isinstance(weight, np.ndarray):
            message = "expected weight to be of same type as x"
            raise TypeError(message)
        if bias is not None and not isinstance(bias, np.ndarray):
            message = "expected bias to be of same type as x"
            raise TypeError(message)
    elif isinstance(x, Tracer):
        if not isinstance(weight, (Tracer, np.ndarray)):
            message = "expected weight to be of type Tracer or ndarray"
            raise TypeError(message)
        if bias is not None and not isinstance(bias, (Tracer, np.ndarray)):
            message = "expected bias to be of type Tracer or ndarray"
            raise TypeError(message)

    if x.ndim <= 2:
        message = (
            f"expected input x to have at least 3 dimensions (N, C, D1, ...), but got {x.ndim}"
        )
        raise ValueError(message)

    if weight.ndim <= 2:
        message = (
            f"expected weight to have at least 3 dimensions (F, C / group, K1, ...), but got "
            f"{weight.ndim}"
        )
        raise ValueError(message)

    if bias is not None and bias.ndim != 1:
        message = f"expected bias to have a single dimension (F,), but got {bias.ndim}"
        raise ValueError(message)

    if not isinstance(group, int) or group <= 0:
        message = f"expected group to be an integer > 0, but got {group}"
        raise ValueError(message)

    if auto_pad not in SUPPORTED_AUTO_PAD:
        message = f"auto_pad should be in {SUPPORTED_AUTO_PAD}, but got {repr(auto_pad)}"
        raise ValueError(message)

    n_channels = x.shape[1]
    if weight.shape[1] != n_channels / group:
        message = (
            f"expected number of channel in weight to be {n_channels / group} (C / group), but got "
            f"{weight.shape[1]}"
        )
        raise ValueError(message)

    if weight.shape[0] % group != 0:
        message = (
            f"expected number of feature maps ({weight.shape[0]}) to be a multiple of group "
            f"({group})"
        )
        raise ValueError(message)

    dims = x.ndim - 2
    if dims == 1:
        pads = (0, 0) if pads is None else pads
        strides = (1,) if strides is None else strides
        dilations = (1,) if dilations is None else dilations
        return _conv1d(
            x,
            weight,
            bias=bias,
            pads=pads,
            strides=strides,
            dilations=dilations,
            group=group,
            auto_pad=auto_pad,
        )
    if dims == 2:
        pads = (0, 0, 0, 0) if pads is None else pads
        strides = (1, 1) if strides is None else strides
        dilations = (1, 1) if dilations is None else dilations
        return _conv2d(
            x,
            weight,
            bias=bias,
            pads=pads,
            strides=strides,
            dilations=dilations,
            group=group,
            auto_pad=auto_pad,
        )
    if dims == 3:
        pads = (0, 0, 0, 0, 0, 0) if pads is None else pads
        strides = (1, 1, 1) if strides is None else strides
        dilations = (1, 1, 1) if dilations is None else dilations
        return _conv3d(
            x,
            weight,
            bias=bias,
            pads=pads,
            strides=strides,
            dilations=dilations,
            group=group,
            auto_pad=auto_pad,
        )

    message = "only 1D, 2D, and 3D convolutions are supported"
    raise NotImplementedError(message)


# pylint: enable=too-many-branches


def _conv1d(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
    auto_pad: str,  # pylint: disable=unused-argument
) -> Union[np.ndarray, Tracer]:
    """
    Trace or evaluate 1D convolution.

    Args:
        x (Union[np.ndarray, Tracer]): input of shape (N, C, D)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C, D)
        bias (Optional[Union[np.ndarray, Tracer]]): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over dimension D (D_beg, D_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over dimension D
        dilations (Union[Tuple[int, ...], List[int]]): dilation over dimension D
        group (int, optional):
            number of groups input channels and output channels are divided into.
        auto_pad (str, optional): padding strategy.

    Raises:
        ValueError: if arguments are not appropriate

    Returns:
        Union[np.ndarray, Tracer]: evaluation result or traced computation
    """

    assert_that(
        x.ndim == 3,
        f"expected input x to be of shape (N, C, D) when performing 1D convolution, but "
        f"got {x.shape}",
    )

    assert_that(
        weight.ndim == 3,
        f"expected weight to be of shape (F, C, D) when performing 1D convolution, but "
        f"got {weight.shape}",
    )

    if len(pads) != 2:
        message = (
            f"pads should be of form "
            f"(D_begin_pad, D_end_pad) when performing "
            f"1D convolution, but it's {pads}"
        )
        raise ValueError(message)
    if len(strides) != 1:
        message = (
            f"strides should be of form (D_stride,) when performing 1D "
            f"convolution, but it's {strides}"
        )
        raise ValueError(message)
    if len(dilations) != 1:
        message = (
            f"dilations should be of form (D_dilation,) when performing 1D "
            f"convolution, but it's {dilations}"
        )
        raise ValueError(message)

    return _trace_or_eval(x, weight, bias, pads, strides, dilations, group)


def _conv2d(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
    auto_pad: str,  # pylint: disable=unused-argument
) -> Union[np.ndarray, Tracer]:
    """
    Trace or evaluate 2D convolution.

    Args:
        x (Union[np.ndarray, Tracer]): input of shape (N, C, H, W)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C, H, W)
        bias (Optional[Union[np.ndarray, Tracer]]): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over each height and width (H_beg, W_beg, H_end, W_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over height and width
        dilations (Union[Tuple[int, ...], List[int]]): dilation over height and width
        group (int, optional):
            number of groups input channels and output channels are divided into.
        auto_pad (str, optional): padding strategy.

    Raises:
        ValueError: if arguments are not appropriate

    Returns:
        Union[np.ndarray, Tracer]: evaluation result or traced computation
    """

    assert_that(
        x.ndim == 4,
        f"expected input x to be of shape (N, C, H, W) when performing 2D convolution, but "
        f"got {x.shape}",
    )

    assert_that(
        weight.ndim == 4,
        f"expected weight to be of shape (F, C, H, W) when performing 2D convolution, but "
        f"got {weight.shape}",
    )

    if len(pads) != 4:
        message = (
            f"pads should be of form "
            f"(height_begin_pad, width_begin_pad, height_end_pad, width_end_pad) when performing "
            f"2D convolution, but it's {pads}"
        )
        raise ValueError(message)
    if len(strides) != 2:
        message = (
            f"strides should be of form (height_stride, width_stride) when performing 2D "
            f"convolution, but it's {strides}"
        )
        raise ValueError(message)
    if len(dilations) != 2:
        message = (
            f"dilations should be of form (height_dilation, width_dilation) when performing 2D "
            f"convolution, but it's {dilations}"
        )
        raise ValueError(message)

    return _trace_or_eval(x, weight, bias, pads, strides, dilations, group)


def _conv3d(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
    auto_pad: str,  # pylint: disable=unused-argument
) -> Union[np.ndarray, Tracer]:
    """
    Trace or evaluate 3D convolution.

    Args:
        x (Union[np.ndarray, Tracer]): input of shape (N, C, D, H, W)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C, D, H, W)
        bias (Optional[Union[np.ndarray, Tracer]]): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over each spatial axis (D_beg, H_beg, W_beg, D_end, H_end, W_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over each spatial axis
        dilations (Union[Tuple[int, ...], List[int]]): dilation over each spatial axis
        group (int, optional):
            number of groups input channels and output channels are divided into.
        auto_pad (str, optional): padding strategy.

    Raises:
        ValueError: if arguments are not appropriate

    Returns:
        Union[np.ndarray, Tracer]: evaluation result or traced computation
    """

    assert_that(
        x.ndim == 5,
        f"expected input x to be of shape (N, C, D, H, W) when performing 3D convolution, but "
        f"got {x.shape}",
    )

    assert_that(
        weight.ndim == 5,
        f"expected weight to be of shape (F, C, D, H, W) when performing 3D convolution, but "
        f"got {weight.shape}",
    )

    if len(pads) != 6:
        message = (
            f"pads should be of form "
            f"(D_begin_pad, height_begin_pad, width_begin_pad, "
            f"D_end_pad, height_end_pad, width_end_pad) when performing "
            f"3D convolution, but it's {pads}"
        )
        raise ValueError(message)
    if len(strides) != 3:
        message = (
            f"strides should be of form (D_stride, height_stride, width_stride) when performing "
            f"3D convolution, but it's {strides}"
        )
        raise ValueError(message)
    if len(dilations) != 3:
        message = (
            f"dilations should be of form (D_dilation, height_dilation, width_dilation) when "
            f"performing 3D convolution, but it's {dilations}"
        )
        raise ValueError(message)

    return _trace_or_eval(x, weight, bias, pads, strides, dilations, group)


def _trace_or_eval(
    x: Union[np.ndarray, Tracer],
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
) -> Union[np.ndarray, Tracer]:
    """
    Trace or evaluate convolution.

    Args:
        x (Union[np.ndarray, Tracer]): input of shape (N, C, D1, ..., DN)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C / group, K1, ..., KN)
        bias (Optional[Union[np.ndarray, Tracer]]): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding for the beginning and ending along each spatial axis
            (D1_begin, D2_begin, ..., D1_end, D2_end, ...).
        strides (Union[Tuple[int, ...], List[int]]): stride along each spatial axis.
        dilations (Union[Tuple[int, ...], List[int]]): dilation along each spatial axis.
        group (int, optional):
            number of groups input channels and output channels are divided into.

    Returns:
        Union[np.ndarray, Tracer]: evaluation result or traced computation
    """
    assert_that(x.ndim in [3, 4, 5], "only support 1D, 2D, and 3D conv")
    if x.ndim == 3:
        conv_func = "conv1d"
    elif x.ndim == 4:
        conv_func = "conv2d"
    else:  # x.ndim == 5
        conv_func = "conv3d"

    if isinstance(x, Tracer):
        return _trace_conv(x, weight, bias, pads, strides, dilations, group, conv_func)

    assert isinstance(x, np.ndarray)
    assert isinstance(weight, np.ndarray)

    dtype = (
        np.float64
        if np.issubdtype(x.dtype, np.floating) or np.issubdtype(weight.dtype, np.floating)
        else np.int64
    )
    bias = np.zeros(weight.shape[0], dtype=dtype) if bias is None else bias

    assert isinstance(bias, np.ndarray)

    return _evaluate_conv(x, weight, bias, pads, strides, dilations, group, conv_func)


def _trace_conv(
    x: Tracer,
    weight: Union[np.ndarray, Tracer],
    bias: Optional[Union[np.ndarray, Tracer]],
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
    conv_func: str,
) -> Tracer:
    """
    Trace convolution.

    Args:
        x (Tracer): input of shape (N, C, D1, ..., DN)
        weight (Union[np.ndarray, Tracer]): kernel of shape (F, C / group, K1, ..., KN)
        bias (Optional[Union[np.ndarray, Tracer]]): bias of shape (F,)
        pads (Union[Tuple[int, int, int, int], List[int]]):
            padding for the beginning and ending along each spatial axis
            (D1_begin, D2_begin, ..., D1_end, D2_end, ...).
        strides (Union[Tuple[int, int], List[int]]): stride along each spatial axis.
        dilations (Union[Tuple[int, int], List[int]]): dilation along each spatial axis.
        group (int, optional):
            number of groups input channels and output channels are divided into.
        conv_func (str): convolution to apply, should be one of {conv1d,conv2d,conv3d}

    Returns:
        Tracer:
            traced computation
    """

    conv_eval_funcs = {
        "conv1d": _evaluate_conv1d,
        "conv2d": _evaluate_conv2d,
        "conv3d": _evaluate_conv3d,
    }
    eval_func = conv_eval_funcs.get(conv_func, None)
    assert_that(
        eval_func is not None,
        f"expected conv_func to be one of {list(conv_eval_funcs.keys())}, but got {conv_func}",
    )
    eval_func = cast(Callable, eval_func)

    weight = weight if isinstance(weight, Tracer) else Tracer(Node.constant(weight), [])

    input_values = [x.output, weight.output]
    inputs = [x, weight]

    if bias is not None:
        bias = bias if isinstance(bias, Tracer) else Tracer(Node.constant(bias), [])
        input_values.append(bias.output)
        inputs.append(bias)

    batch_size = x.output.shape[0]
    n_filters = weight.output.shape[0]

    n_dim = x.ndim - 2  # remove batch_size and channel dims
    total_pads_per_dim = []
    for dim in range(n_dim):
        total_pads_per_dim.append(pads[dim] + pads[n_dim + dim])

    output_shape = [batch_size, n_filters]
    for dim in range(n_dim):
        input_dim_at_dim = x.output.shape[dim + 2]
        weight_dim_at_dim = weight.output.shape[dim + 2]
        output_shape.append(
            math.floor(
                (
                    input_dim_at_dim
                    + total_pads_per_dim[dim]
                    - dilations[dim] * (weight_dim_at_dim - 1)
                    - 1
                )
                / strides[dim]
            )
            + 1
        )
    output_value = EncryptedTensor(dtype=x.output.dtype, shape=tuple(output_shape))

    computation = Node.generic(
        conv_func,  # "conv1d" or "conv2d" or "conv3d"
        input_values,
        output_value,
        eval_func,
        args=() if bias is not None else (np.zeros(n_filters, dtype=np.int64),),
        kwargs={"pads": pads, "strides": strides, "dilations": dilations, "group": group},
    )
    return Tracer(computation, inputs)


def _evaluate_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
) -> np.ndarray:
    """
    Evaluate 1D convolution.

    Args:
        x (np.ndarray): input of shape (N, C, D)
        weight (np.ndarray): kernel of shape (F, C / group, D)
        bias (np.ndarray): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over each axis (D_beg, D_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over dimension D
        dilations (Union[Tuple[int, ...], List[int]]): dilation over dimension D
        group (int, optional):
            number of groups input channels and output channels are divided into.

    Returns:
        np.ndarray: result of the convolution
    """
    return _evaluate_conv(x, weight, bias, pads, strides, dilations, group, "conv1d")


def _evaluate_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
) -> np.ndarray:
    """
    Evaluate 2D convolution.

    Args:
        x (np.ndarray): input of shape (N, C, H, W)
        weight (np.ndarray): kernel of shape (F, C / group, H, W)
        bias (np.ndarray): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over each axis (H_beg, W_beg, H_end, W_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over height and width
        dilations (Union[Tuple[int, ...], List[int]]): dilation over height and width
        group (int, optional):
            number of groups input channels and output channels are divided into.

    Returns:
        np.ndarray: result of the convolution
    """
    return _evaluate_conv(x, weight, bias, pads, strides, dilations, group, "conv2d")


def _evaluate_conv3d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, ...], List[int]],
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
) -> np.ndarray:
    """
    Evaluate 3D convolution.

    Args:
        x (np.ndarray): input of shape (N, C, D, H, W)
        weight (np.ndarray): kernel of shape (F, C / group, D, H, W)
        bias (np.ndarray): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding over each axis (D_beg, H_beg, W_beg, D_end, H_end, W_end)
        strides (Union[Tuple[int, ...], List[int]]): stride over D, height, and width
        dilations (Union[Tuple[int, ...], List[int]]): dilation over D, height, and width
        group (int, optional):
            number of groups input channels and output channels are divided into.

    Returns:
        np.ndarray: result of the convolution
    """
    return _evaluate_conv(x, weight, bias, pads, strides, dilations, group, "conv3d")


def _evaluate_conv(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    pads: Union[Tuple[int, ...], List[int]],  # pylint: disable=unused-argument
    strides: Union[Tuple[int, ...], List[int]],
    dilations: Union[Tuple[int, ...], List[int]],
    group: int,
    conv_func: str,
) -> np.ndarray:
    """
    Evaluate 2D convolution.

    Args:
        x (np.ndarray): input of shape (N, C, D1, ..., DN)
        weight (np.ndarray): kernel of shape (F, C / group, K1, ..., KN)
        bias (np.ndarray): bias of shape (F,)
        pads (Union[Tuple[int, ...], List[int]]):
            padding for the beginning and ending along each spatial axis
            (D1_begin, D2_begin, ..., D1_end, D2_end, ...).
        strides (Union[Tuple[int, ...], List[int]]): stride along each spatial axis.
        dilations (Union[Tuple[int, ...], List[int]]): dilation along each spatial axis.
        group (int, optional):
            number of groups input channels and output channels are divided into.
        conv_func (str): convolution to apply, should be one of {conv1d,conv2d,conv3d}

    Returns:
        np.ndarray: result of the convolution
    """

    # pylint: disable=no-member
    conv_funcs = {
        "conv1d": torch.conv1d,
        "conv2d": torch.conv2d,
        "conv3d": torch.conv3d,
    }

    torch_conv_func = conv_funcs.get(conv_func, None)
    assert_that(
        torch_conv_func is not None,
        f"expected conv_func to be one of {list(conv_funcs.keys())}, but got {conv_func}",
    )
    torch_conv_func = cast(Callable, torch_conv_func)

    n_dim = x.ndim - 2  # remove batch_size and channel dims
    torch_padding = []
    for dim in range(n_dim):
        if pads[dim] != pads[n_dim + dim]:
            message = (
                f"padding should be the same for the beginning of the dimension and its end, but "
                f"got {pads[dim]} in the beginning, and {pads[n_dim + dim]} at the end for "
                f"dimension {dim}"
            )
            raise ValueError(message)
        torch_padding.append(pads[dim])

    dtype = (
        torch.float64
        if np.issubdtype(x.dtype, np.floating)
        or np.issubdtype(weight.dtype, np.floating)
        or np.issubdtype(bias.dtype, np.floating)
        else torch.long
    )
    return torch_conv_func(
        torch.tensor(x, dtype=dtype),
        torch.tensor(weight, dtype=dtype),
        torch.tensor(bias, dtype=dtype),
        stride=strides,
        padding=torch_padding,
        dilation=dilations,
        groups=group,
    ).numpy()

    # pylint: enable=no-member
