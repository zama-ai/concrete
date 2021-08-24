"""Test converter functions"""
import pytest

from hdk.common.data_types.floats import Float
from hdk.common.data_types.integers import Integer
from hdk.common.mlir.converters import add, constant, mul, sub
from hdk.common.values import ClearValue


class MockNode:
    """Mocking an intermediate node"""

    def __init__(self, inputs_n=5, outputs_n=5, inputs=None, outputs=None):
        if inputs is None:
            self.inputs = [None for i in range(inputs_n)]
        else:
            self.inputs = inputs
        if outputs is None:
            self.outputs = [None for i in range(outputs_n)]
        else:
            self.outputs = outputs


@pytest.mark.parametrize("converter", [add, sub, mul])
def test_failing_converter(converter):
    """Test failing converter"""
    with pytest.raises(TypeError, match=r"Don't support .* between .* and .*"):
        converter(MockNode(2, 1), None, None, None)


def test_fail_non_integer_const():
    """Test failing constant converter with non-integer"""
    with pytest.raises(TypeError, match=r"Don't support non-integer constants"):
        constant(MockNode(outputs=[ClearValue(Float(32))]), None, None, None)


def test_fail_signed_integer_const():
    """Test failing constant converter with non-integer"""
    with pytest.raises(TypeError, match=r"Don't support signed constant integer"):
        constant(MockNode(outputs=[ClearValue(Integer(8, True))]), None, None, None)
