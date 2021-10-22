"""Test converter functions"""
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.mlir.converters import add, apply_lut, constant, dot, mul, sub
from concrete.common.values import ClearScalar, ClearTensor, EncryptedScalar


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


@pytest.mark.parametrize("converter", [add, sub, mul, dot])
def test_failing_converter(converter):
    """Test failing converter"""
    with pytest.raises(TypeError, match=r"Don't support .* between .* and .*"):
        converter(MockNode(2, 1), None, None, None)


def test_fail_non_integer_const():
    """Test failing constant converter with non-integer"""
    with pytest.raises(TypeError, match=r"Don't support .* constants"):
        constant(MockNode(outputs=[ClearScalar(Float(32))]), None, None, None)

    with pytest.raises(TypeError, match=r"Don't support .* constants"):
        constant(MockNode(outputs=[ClearTensor(Float(32), shape=(2,))]), None, None, None)


@pytest.mark.parametrize(
    "input_node",
    [
        ClearScalar(Integer(8, True)),
        ClearScalar(Integer(8, False)),
        EncryptedScalar(Integer(8, True)),
    ],
)
def test_fail_tlu_input(input_node):
    """Test failing LUT converter with invalid input"""
    with pytest.raises(
        TypeError, match=r"Only support LUT with encrypted unsigned integers inputs"
    ):
        apply_lut(
            MockNode(inputs=[input_node], outputs=[EncryptedScalar(Integer(8, False))]),
            [None],
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    "input_node",
    [
        ClearScalar(Integer(8, True)),
        ClearScalar(Integer(8, False)),
        EncryptedScalar(Integer(8, True)),
    ],
)
def test_fail_tlu_output(input_node):
    """Test failing LUT converter with invalid output"""
    with pytest.raises(
        TypeError, match=r"Only support LUT with encrypted unsigned integers outputs"
    ):
        apply_lut(
            MockNode(inputs=[EncryptedScalar(Integer(8, False))], outputs=[input_node]),
            [None],
            None,
            None,
            None,
        )
