"""Test module for convolution compilation and execution."""

import numpy as np
import pytest

import concrete.numpy as hnp
from concrete.common.data_types.integers import Integer
from concrete.common.values.tensors import EncryptedTensor
from concrete.numpy.compile import compile_numpy_function


@pytest.mark.parametrize(
    "input_shape, weight_shape",
    [
        pytest.param((1, 1, 4, 4), (1, 1, 2, 2)),
        pytest.param((4, 3, 4, 4), (2, 3, 2, 2)),
    ],
)
@pytest.mark.parametrize("strides", [(2, 2)])
@pytest.mark.parametrize("dilations", [(1, 1)])
@pytest.mark.parametrize("has_bias", [True, False])
def test_compile_and_run(
    input_shape, weight_shape, strides, dilations, has_bias, default_compilation_configuration
):
    """Test function to make sure compilation and execution of conv2d works properly"""
    if has_bias:
        bias = np.random.randint(0, 4, size=(weight_shape[0],))
    else:
        bias = None
    weight = np.random.randint(0, 4, size=weight_shape)

    def conv(x):
        return hnp.conv2d(x, weight, bias, strides=strides, dilations=dilations)

    compiler_engine = compile_numpy_function(
        conv,
        {"x": EncryptedTensor(Integer(64, False), input_shape)},
        [np.random.randint(0, 4, size=input_shape) for i in range(20)],
        default_compilation_configuration,
    )
    x = np.random.randint(0, 4, size=input_shape, dtype=np.uint8)
    expected = conv(x)
    result = compiler_engine.encrypt_run_decrypt(x)
    assert (expected == result).all()
