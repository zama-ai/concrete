"""Test file for convolution"""

import numpy as np
import pytest
import torch

from concrete.common.extensions import convolution
from concrete.common.representation.intermediate import Conv2D
from concrete.common.tracing.base_tracer import BaseTracer
from concrete.common.values.tensors import TensorValue
from concrete.numpy.tracing import NPConstant, NPTracer


@pytest.mark.parametrize(
    "kwargs, error_msg",
    [
        pytest.param(
            {"x": None, "weight": np.zeros(1)},
            "input x must be an ndarray, or a BaseTracer, not a",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": None},
            "weight must be an ndarray, or a BaseTracer, not a",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "bias": 0},
            "bias must be an ndarray, a BaseTracer, or None, not a",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "strides": None},
            "strides must be a tuple, or list, not a",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "dilations": None},
            "dilations must be a tuple, or list, not a",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "pads": None},
            "padding must be a tuple, or list, not a",
        ),
    ],
)
def test_invalid_arg_types(kwargs, error_msg):
    """Test function to make sure convolution doesn't accept invalid types"""

    with pytest.raises(TypeError) as err:
        convolution.conv2d(**kwargs)

    assert error_msg in str(err)


@pytest.mark.parametrize(
    "kwargs, error_msg",
    [
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1)},
            "input x should have size (N x C x H x W), not",
        ),
        pytest.param(
            {"x": np.zeros((1, 2, 3, 4)), "weight": np.zeros(1)},
            "weight should have size (F x C x H x W), not",
        ),
        pytest.param(
            {
                "x": np.zeros((1, 2, 3, 4)),
                "weight": np.zeros((1, 2, 3, 4)),
                "bias": np.zeros((1, 2)),
            },
            "bias should have size (F), not",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "strides": (1,)},
            "strides should be of the form",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "dilations": (1,)},
            "dilations should be of the form",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "pads": (1,)},
            "padding should be of the form",
        ),
        pytest.param(
            {"x": np.zeros(1), "weight": np.zeros(1), "auto_pad": None},
            "invalid auto_pad is specified",
        ),
    ],
)
def test_invalid_input_shape(kwargs, error_msg):
    """Test function to make sure convolution doesn't accept invalid shapes"""

    with pytest.raises((ValueError, AssertionError)) as err:
        convolution.conv2d(**kwargs)

    assert error_msg in str(err)


@pytest.mark.parametrize(
    "input_shape, weight_shape",
    [
        pytest.param((1, 1, 4, 4), (1, 1, 2, 2)),
        pytest.param((3, 1, 4, 4), (1, 1, 2, 2)),
        pytest.param((1, 1, 4, 4), (3, 1, 2, 2)),
        pytest.param((1, 3, 4, 4), (1, 3, 2, 2)),
        pytest.param((4, 3, 4, 4), (3, 3, 2, 2)),
        pytest.param((4, 3, 16, 16), (3, 3, 2, 2)),
        pytest.param((4, 3, 16, 16), (3, 3, 3, 3)),
    ],
)
@pytest.mark.parametrize("strides", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("dilations", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("use_ndarray", [True, False])
def test_tracing(input_shape, weight_shape, strides, dilations, has_bias, use_ndarray):
    """Test function to make sure tracong of conv2d works properly"""
    if has_bias:
        bias = np.random.randint(0, 4, size=(weight_shape[0],))
        if not use_ndarray:
            bias = NPTracer([], NPConstant(bias), 0)
    else:
        bias = None

    x = NPTracer([], NPConstant(np.random.randint(0, 4, size=input_shape)), 0)
    weight = np.random.randint(0, 4, size=weight_shape)
    if not use_ndarray:
        weight = NPTracer([], NPConstant(weight), 0)

    output_tracer = convolution.conv2d(x, weight, bias, strides=strides, dilations=dilations)
    traced_computation = output_tracer.traced_computation
    assert isinstance(traced_computation, Conv2D)

    if has_bias:
        assert len(output_tracer.inputs) == 3
    else:
        assert len(output_tracer.inputs) == 2

    assert all(
        isinstance(input_, BaseTracer) for input_ in output_tracer.inputs
    ), f"{output_tracer.inputs}"

    assert len(traced_computation.outputs) == 1
    output_value = traced_computation.outputs[0]
    assert isinstance(output_value, TensorValue) and output_value.is_encrypted
    # pylint: disable=no-member
    expected_shape = torch.conv2d(
        torch.randn(input_shape),
        torch.randn(weight_shape),
        torch.randn((weight_shape[0])),
        stride=strides,
        dilation=dilations,
    ).shape
    # pylint: enable=no-member

    assert output_value.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape, weight_shape",
    [
        pytest.param((1, 1, 4, 4), (1, 1, 2, 2)),
        pytest.param((3, 1, 4, 4), (1, 1, 2, 2)),
        pytest.param((1, 1, 4, 4), (3, 1, 2, 2)),
        pytest.param((1, 3, 4, 4), (1, 3, 2, 2)),
        pytest.param((4, 3, 4, 4), (3, 3, 2, 2)),
        pytest.param((4, 3, 16, 16), (3, 3, 2, 2)),
        pytest.param((4, 3, 16, 16), (3, 3, 3, 3)),
    ],
)
@pytest.mark.parametrize("strides", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("dilations", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("has_bias", [True, False])
def test_evaluation(input_shape, weight_shape, strides, dilations, has_bias):
    """Test function to make sure evaluation of conv2d on plain data works properly"""
    if has_bias:
        bias = np.random.randint(0, 4, size=(weight_shape[0],))
    else:
        bias = np.zeros((weight_shape[0],))
    x = np.random.randint(0, 4, size=input_shape)
    weight = np.random.randint(0, 4, size=weight_shape)
    # pylint: disable=no-member
    expected = torch.conv2d(
        torch.tensor(x, dtype=torch.long),
        torch.tensor(weight, dtype=torch.long),
        torch.tensor(bias, dtype=torch.long),
        stride=strides,
        dilation=dilations,
    ).numpy()
    # pylint: enable=no-member
    # conv2d should handle None biases
    if not has_bias:
        bias = None
    result = convolution.conv2d(x, weight, bias, strides=strides, dilations=dilations)
    assert (result == expected).all()
