"""Test converter functions"""
import pytest

from hdk.common.mlir.converters import add, mul, sub


class MockNode:
    """Mocking an intermediate node"""

    def __init__(self, inputs=5, outputs=5):
        self.inputs = [None for i in range(inputs)]
        self.outputs = [None for i in range(outputs)]


@pytest.mark.parametrize("converter", [add, sub, mul])
def test_failing_converter(converter):
    """Test failing converter"""
    with pytest.raises(TypeError, match=r"Don't support .* between .* and .*"):
        converter(MockNode(2, 1), None, None, None)
