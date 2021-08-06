"""Test file for intermediate representation"""

import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.representation import intermediate as ir


@pytest.mark.parametrize(
    "node,input_data,expected_result",
    [
        pytest.param(
            ir.Add([EncryptedValue(Integer(64, False)), EncryptedValue(Integer(64, False))]),
            [10, 4589],
            4599,
            id="Add",
        ),
        pytest.param(
            ir.Sub([EncryptedValue(Integer(64, False)), EncryptedValue(Integer(64, False))]),
            [10, 4589],
            -4579,
            id="Sub",
        ),
        pytest.param(
            ir.Mul([EncryptedValue(Integer(64, False)), EncryptedValue(Integer(64, False))]),
            [10, 4589],
            45890,
            id="Mul",
        ),
        pytest.param(ir.Input(ClearValue(Integer(32, True)), "in", 0), [42], 42, id="Input"),
        pytest.param(ir.ConstantInput(42), None, 42, id="ConstantInput"),
        pytest.param(ir.ConstantInput(-42), None, -42, id="ConstantInput"),
        pytest.param(
            ir.ArbitraryFunction(
                EncryptedValue(Integer(7, False)), lambda x: x + 3, Integer(7, False)
            ),
            [10],
            13,
            id="ArbitraryFunction, x + 3",
        ),
        pytest.param(
            ir.ArbitraryFunction(
                EncryptedValue(Integer(7, False)),
                lambda x, y: x + y,
                Integer(7, False),
                op_kwargs={"y": 3},
            ),
            [10],
            13,
            id="ArbitraryFunction, (x, y) -> x + y, where y is constant == 3",
        ),
        pytest.param(
            ir.ArbitraryFunction(
                EncryptedValue(Integer(7, False)),
                lambda x, y: y[x],
                Integer(7, False),
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            3,
            id="ArbitraryFunction, (x, y) -> y[x], where y is constant == (1, 2, 3, 4)",
        ),
        pytest.param(
            ir.ArbitraryFunction(
                EncryptedValue(Integer(7, False)),
                lambda x, y: y[3],
                Integer(7, False),
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            4,
            id="ArbitraryFunction, x, y -> y[3], where y is constant == (1, 2, 3, 4)",
        ),
    ],
)
def test_evaluate(
    node: ir.IntermediateNode,
    input_data,
    expected_result: int,
):
    """Test evaluate methods on IntermediateNodes"""
    assert node.evaluate(input_data) == expected_result
