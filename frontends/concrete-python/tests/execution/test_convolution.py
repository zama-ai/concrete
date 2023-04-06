"""
Tests of execution of convolution operation.
"""

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe.representation.node import Node
from concrete.fhe.tracing.tracer import Tracer


@pytest.mark.parametrize(
    "input_shape,weight_shape, group",
    [
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            1,
        ),
        pytest.param(
            (4, 3, 4, 4),
            (2, 3, 2, 2),
            1,
        ),
        pytest.param(
            (1, 6, 4, 4),
            (6, 1, 2, 2),
            6,
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
def test_conv2d(input_shape, weight_shape, group, strides, dilations, has_bias, helpers):
    """
    Test conv2d.
    """

    configuration = helpers.configuration()

    weight = np.random.randint(0, 4, size=weight_shape)
    bias = np.random.randint(0, 4, size=(weight_shape[0],)) if has_bias else None

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.conv(x, weight, bias, strides=strides, dilations=dilations, group=group)

    inputset = [np.random.randint(0, 4, size=input_shape) for i in range(100)]
    circuit = function.compile(inputset, configuration)

    sample = np.random.randint(0, 4, size=input_shape)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "input_shape,weight_shape,bias_shape,pads,strides,dilations,kernel_shape,group,auto_pad,"
    "expected_error,expected_message",
    [
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "VALID",
            ValueError,
            "auto_pad should be in {'NOTSET'}, but got 'VALID'",
        ),
        pytest.param(
            (1, 1, 1, 4),
            (1, 1, 2, 2),
            (1,),
            (1, 0, 2, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            RuntimeError,
            "padding should be the same for the beginning of the dimension and its end, but got "
            "1 in the beginning, and 2 at the end for dimension 0",
        ),
        pytest.param(
            (1, 1, 4),
            (1, 1, 2),
            (1,),
            (),
            (1,),
            (1,),
            None,
            1,
            "NOTSET",
            ValueError,
            "pads should be of form (D_begin_pad, D_end_pad) when performing 1D convolution, "
            "but it's ()",
        ),
        pytest.param(
            (1, 1, 4),
            (1, 1, 2),
            (1,),
            (0, 0),
            (),
            (1,),
            None,
            1,
            "NOTSET",
            ValueError,
            "strides should be of form (D_stride,) when performing 1D convolution, but it's ()",
        ),
        pytest.param(
            (1, 1, 4),
            (1, 1, 2),
            (1,),
            (0, 0),
            (1,),
            (),
            None,
            1,
            "NOTSET",
            ValueError,
            "dilations should be of form (D_dilation,) when performing 1D "
            "convolution, but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "pads should be of form (height_begin_pad, width_begin_pad, "
            "height_end_pad, width_end_pad) when performing 2D convolution, "
            "but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "strides should be of form (height_stride, width_stride) when performing 2D "
            "convolution, but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (),
            None,
            1,
            "NOTSET",
            ValueError,
            "dilations should be of form (height_dilation, width_dilation) when performing 2D "
            "convolution, but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4, 4),
            (1, 1, 2, 2, 4),
            (1,),
            (),
            (1, 1, 1),
            (1, 1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "pads should be of form (D_begin_pad, height_begin_pad, width_begin_pad, "
            "D_end_pad, height_end_pad, width_end_pad) when performing 3D convolution, "
            "but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4, 4),
            (1, 1, 2, 2, 2),
            (1,),
            (0, 0, 0, 0, 0, 0),
            (),
            (1, 1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "strides should be of form (D_stride, height_stride, width_stride) when performing 3D "
            "convolution, but it's ()",
        ),
        pytest.param(
            (1, 1, 4, 4, 4),
            (1, 1, 2, 2, 2),
            (1,),
            (0, 0, 0, 0, 0, 0),
            (1, 1, 1),
            (),
            None,
            1,
            "NOTSET",
            ValueError,
            "dilations should be of form (D_dilation, height_dilation, width_dilation) when "
            "performing 3D convolution, but it's ()",
        ),
        pytest.param(
            (),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "expected input x to have at least 3 dimensions (N, C, D1, ...), but got 0",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "expected weight to have at least 3 dimensions (F, C / group, K1, ...), but got 0",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "expected bias to have a single dimension (F,), but got 0",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            1,
            "NOTSET",
            ValueError,
            "expected kernel_shape to be (2, 2), but got (1, 2)",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            None,
            "NOTSET",
            ValueError,
            "expected group to be an integer > 0, but got None",
        ),
        pytest.param(
            (1, 1, 4, 4),
            (1, 2, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            1,
            "NOTSET",
            ValueError,
            "expected number of channel in weight to be 1.0 (C / group), but got 2",
        ),
        pytest.param(
            (1, 1, 4, 4, 4, 4),
            (1, 1, 2, 2, 2, 2),
            (1,),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            None,
            1,
            "NOTSET",
            NotImplementedError,
            "only 1D, 2D, and 3D convolutions are supported",
        ),
        pytest.param(
            (1, 2, 4, 4),
            (1, 1, 2, 2),
            (1,),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            None,
            2,
            "NOTSET",
            ValueError,
            "expected number of feature maps (1) to be a multiple of group (2)",
        ),
    ],
)
# pylint: disable=too-many-arguments
def test_bad_conv_compilation(
    input_shape,
    weight_shape,
    bias_shape,
    pads,
    strides,
    dilations,
    kernel_shape,
    group,
    auto_pad,
    expected_error,
    expected_message,
    helpers,
):
    # pylint: enable=too-many-arguments
    """
    Test conv with bad parameters.
    """

    configuration = helpers.configuration()

    weight = np.random.randint(0, 4, size=weight_shape)
    bias = np.random.randint(0, 4, size=bias_shape) if bias_shape is not None else None

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return fhe.conv(
            x,
            weight,
            bias=bias,
            pads=pads,
            strides=strides,
            dilations=dilations,
            kernel_shape=kernel_shape,
            group=group,
            auto_pad=auto_pad,
        )

    inputset = [np.random.randint(0, 4, size=input_shape) for i in range(100)]
    with pytest.raises(expected_error) as excinfo:
        function.compile(inputset, configuration)

    # Get the root cause error
    current_error = excinfo.value
    cause = current_error.__cause__
    while cause:
        current_error = cause
        cause = current_error.__cause__

    assert str(current_error) == expected_message


@pytest.mark.parametrize(
    "conv_func_name",
    [
        "conv",
        223,
        None,
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        # pylint: disable=protected-access
        fhe.extensions.convolution._evaluate_conv,
        fhe.extensions.convolution._trace_conv,
        # pylint: enable=protected-access
    ],
)
def test_bad_conv_func_name(conv_func_name, func):
    """
    Test invalid conv function name.
    """
    with pytest.raises(AssertionError) as excinfo:
        func(None, None, None, None, None, None, None, conv_func_name)
    assert (
        str(excinfo.value) == f"expected conv_func to be one of ['conv1d', 'conv2d', 'conv3d'], "
        f"but got {conv_func_name}"
    )


@pytest.mark.parametrize(
    "x,weight,bias,expected_error,expected_message",
    [
        pytest.param(
            np.array([1, 2, 3]),
            "not same type as x",
            None,
            TypeError,
            "expected weight to be of same type as x",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            "not same type as x",
            TypeError,
            "expected bias to be of same type as x",
        ),
        pytest.param(
            Tracer(Node.constant(np.array([1, 2, 3])), []),
            "not same type as x",
            None,
            TypeError,
            "expected weight to be of type Tracer or ndarray",
        ),
        pytest.param(
            Tracer(Node.constant(np.array([1, 2, 3])), []),
            np.array([1, 2, 3]),
            "not same type as x",
            TypeError,
            "expected bias to be of type Tracer or ndarray",
        ),
    ],
)
def test_inconsistent_input_types(
    x,
    weight,
    bias,
    expected_error,
    expected_message,
):
    """
    Test conv with inconsistent input types.
    """
    with pytest.raises(expected_error) as excinfo:
        fhe.conv(
            x,
            weight,
            bias=bias,
        )

    assert str(excinfo.value) == expected_message
