"""
Tests of execution of convolution operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp
import concrete.onnx as connx


@pytest.mark.parametrize(
    "input_shape,weight_shape",
    [
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
        ),
        pytest.param(
            (4, 3, 4, 4),
            (2, 3, 2, 2),
        ),
    ],
)
@pytest.mark.parametrize(
    "strides",
    [
        (2, 2),
    ],
)
@pytest.mark.parametrize(
    "dilations",
    [
        (1, 1),
    ],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        True,
        False,
    ],
)
def test_conv2d(input_shape, weight_shape, strides, dilations, has_bias, helpers):
    """
    Test conv2d.
    """

    configuration = helpers.configuration()

    weight = np.random.randint(0, 4, size=weight_shape)

    if has_bias:
        bias = np.random.randint(0, 4, size=(weight_shape[0],))
    else:
        bias = None

    @cnp.compiler({"x": "encrypted"}, configuration=configuration)
    def function(x):
        return connx.conv2d(x, weight, bias, strides=strides, dilations=dilations)

    inputset = [np.random.randint(0, 4, size=input_shape) for i in range(100)]
    circuit = function.compile(inputset)

    sample = np.random.randint(0, 4, size=input_shape, dtype=np.uint8)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "input_shape,weight_shape,bias_shape,pads,strides,dilations,auto_pad,"
    "expected_error,expected_message",
    [
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "VALID",
            ValueError,
            "Auto pad should be in {'NOTSET'} but it's 'VALID'",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (),
            (1, 1),
            (1, 1),
            "NOTSET",
            ValueError,
            "Pads should be of form "
            "(height_begin_pad, width_begin_pad, height_end_pad, width_end_pad) "
            "but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (),
            (1, 1),
            "NOTSET",
            ValueError,
            "Strides should be of form (height_stride, width_stride) but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (),
            "NOTSET",
            ValueError,
            "Dilations should be of form (height_dilation, width_dilation) but it's ()",
        ),
        pytest.param(
            (),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NOTSET",
            ValueError,
            "Input should be of shape (N, C, H, W) but it's of shape ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NOTSET",
            ValueError,
            "Weight should be of shape (F, C, H, W) but it's of shape ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NOTSET",
            ValueError,
            "Bias should be of shape (F,) but it's of shape ()",
        ),
    ],
)
def test_bad_conv2d_tracing(
    input_shape,
    weight_shape,
    bias_shape,
    pads,
    strides,
    dilations,
    auto_pad,
    expected_error,
    expected_message,
    helpers,
):
    """
    Test conv2d with bad parameters.
    """

    configuration = helpers.configuration()

    weight = np.random.randint(0, 4, size=weight_shape)

    if bias_shape is not None:
        bias = np.random.randint(0, 4, size=bias_shape)
    else:
        bias = None

    @cnp.compiler({"x": "encrypted"}, configuration=configuration)
    def function(x):
        return connx.conv2d(x, weight, bias, pads, strides, dilations, auto_pad)

    inputset = [np.random.randint(0, 4, size=input_shape) for i in range(100)]
    with pytest.raises(expected_error) as excinfo:
        function.compile(inputset)

    assert str(excinfo.value) == expected_message


def test_bad_conv2d_evaluation():
    """
    Test conv2d evaluation with bad parameters.
    """

    x = np.random.randint(0, 4, size=(1, 1, 4, 4))

    with pytest.raises(ValueError) as excinfo:
        connx.conv2d(x, "abc")

    assert str(excinfo.value) == "Weight should be of type np.ndarray for evaluation"

    weight = np.random.randint(0, 4, size=(1, 1, 2, 2))

    with pytest.raises(ValueError) as excinfo:
        connx.conv2d(x, weight, "abc")

    assert str(excinfo.value) == "Bias should be of type np.ndarray for evaluation"
