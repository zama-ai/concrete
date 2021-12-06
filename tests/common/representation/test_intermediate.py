"""Test file for intermediate representation"""
from copy import deepcopy

import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.representation import intermediate as ir
from concrete.common.values import ClearScalar, ClearTensor, EncryptedScalar, EncryptedTensor


@pytest.mark.parametrize(
    "node,input_data,expected_result",
    [
        pytest.param(
            ir.Add([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            4599,
            id="Add",
        ),
        pytest.param(
            ir.Sub([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            -4579,
            id="Sub",
        ),
        pytest.param(
            ir.Mul([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            45890,
            id="Mul",
        ),
        pytest.param(ir.Input(ClearScalar(Integer(32, True)), "in", 0), [42], 42, id="Input"),
        pytest.param(ir.Constant(42), None, 42, id="Constant"),
        pytest.param(ir.Constant(-42), None, -42, id="Constant"),
        pytest.param(
            ir.GenericFunction(
                [EncryptedScalar(Integer(7, False))],
                lambda x: x + 3,
                EncryptedScalar(Integer(7, False)),
                op_kind="TLU",
            ),
            [10],
            13,
            id="GenericFunction, x + 3",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedScalar(Integer(7, False))],
                lambda x, y: x + y,
                EncryptedScalar(Integer(7, False)),
                op_kind="TLU",
                op_kwargs={"y": 3},
            ),
            [10],
            13,
            id="GenericFunction, (x, y) -> x + y, where y is constant == 3",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedScalar(Integer(7, False))],
                lambda x, y: y[x],
                EncryptedScalar(Integer(7, False)),
                op_kind="TLU",
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            3,
            id="GenericFunction, (x, y) -> y[x], where y is constant == (1, 2, 3, 4)",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedScalar(Integer(7, False))],
                lambda x, y: y[3],
                EncryptedScalar(Integer(7, False)),
                op_kind="TLU",
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            4,
            id="GenericFunction, x, y -> y[3], where y is constant == (1, 2, 3, 4)",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
            ),
            [[1, 2, 3, 4], [4, 3, 2, 1]],
            20,
            id="Dot, [1, 2, 3, 4], [4, 3, 2, 1]",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Float(32), shape=(4,)),
                    ClearTensor(Float(32), shape=(4,)),
                ],
                Float(32),
            ),
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            20,
            id="Dot, [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            [
                numpy.array([1, 2, 3, 4], dtype=numpy.int32),
                numpy.array([4, 3, 2, 1], dtype=numpy.int32),
            ],
            20,
            id="Dot, np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])",
        ),
        pytest.param(
            ir.IndexConstant(EncryptedTensor(Integer(4, True), shape=(4,)), (0,)),
            [
                numpy.array([1, 2, 3, 4], dtype=numpy.int32),
            ],
            1,
            id="IndexConstant, np.array([1, 2, 3, 4])[0]",
        ),
        pytest.param(
            ir.IndexConstant(EncryptedTensor(Integer(4, True), shape=(4,)), (slice(1, 3, None),)),
            [
                numpy.array([1, 2, 3, 4], dtype=numpy.int32),
            ],
            numpy.array([2, 3]),
            id="IndexConstant, np.array([1, 2, 3, 4])[1:3]",
        ),
        pytest.param(
            ir.IndexConstant(EncryptedTensor(Integer(4, True), shape=(4,)), (slice(3, 1, -1),)),
            [
                numpy.array([1, 2, 3, 4], dtype=numpy.int32),
            ],
            numpy.array([4, 3], dtype=numpy.int32),
            id="IndexConstant, np.array([1, 2, 3, 4])[3:1:-1]",
        ),
        pytest.param(
            ir.IndexConstant(
                EncryptedTensor(Integer(5, True), shape=(4, 4)), (slice(1, 3, 1), slice(2, 0, -1))
            ),
            [
                numpy.array(
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ],
                    dtype=numpy.int32,
                ),
            ],
            numpy.array(
                [
                    [7, 6],
                    [11, 10],
                ],
                dtype=numpy.int32,
            ),
            id="IndexConstant, np.array([[1, 2, 3, 4]...[13, 14, 15, 16]])[1:3, 2:0:-1]",
        ),
        pytest.param(
            ir.MatMul(
                [
                    EncryptedTensor(Integer(32, True), shape=(3, 2)),
                    ClearTensor(Integer(32, True), shape=(2, 3)),
                ],
                Integer(32, True),
            ),
            [numpy.arange(1, 7).reshape(3, 2), numpy.arange(1, 7).reshape(2, 3)],
            numpy.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]]),
            id="MatMul, numpy.arange(1, 7).reshape(3, 2), numpy.arange(1, 7).reshape(2, 3)",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedTensor(Integer(32, False), shape=(3, 5))],
                lambda x: numpy.transpose(x),
                EncryptedTensor(Integer(32, False), shape=(5, 3)),
                op_kind="Memory",
            ),
            [numpy.arange(15).reshape(3, 5)],
            numpy.array([[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]]),
            id="GenericFunction, x transpose",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedTensor(Integer(32, False), shape=(3, 5))],
                lambda x: numpy.ravel(x),
                EncryptedTensor(Integer(32, False), shape=(5, 3)),
                op_kind="Memory",
            ),
            [numpy.arange(15).reshape(3, 5)],
            numpy.arange(15),
            id="GenericFunction, x ravel",
        ),
        pytest.param(
            ir.GenericFunction(
                [EncryptedTensor(Integer(32, False), shape=(3, 5))],
                lambda x: numpy.reshape(x, (5, 3)),
                output_value=EncryptedTensor(Integer(32, False), shape=(5, 3)),
                op_kind="Memory",
            ),
            [numpy.arange(15).reshape(3, 5)],
            numpy.arange(15).reshape(5, 3),
            id="GenericFunction, x reshape",
        ),
    ],
)
def test_evaluate(
    node: ir.IntermediateNode,
    input_data,
    expected_result: int,
    check_array_equality,
):
    """Test evaluate methods on IntermediateNodes"""
    if isinstance(expected_result, numpy.ndarray):
        check_array_equality(node.evaluate(input_data), expected_result)
    else:
        assert node.evaluate(input_data) == expected_result


@pytest.mark.parametrize(
    "node1,node2,expected_result",
    [
        (
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Add([EncryptedScalar(Integer(16, False)), EncryptedScalar(Integer(32, False))]),
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            True,
        ),
        (
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            True,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            ir.Sub([EncryptedScalar(Integer(16, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            True,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "y", 0),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 1),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(8, False)), "x", 0),
            False,
        ),
        (
            ir.Constant(10),
            ir.Constant(10),
            True,
        ),
        (
            ir.Constant(10),
            ir.Input(EncryptedScalar(Integer(8, False)), "x", 0),
            False,
        ),
        (
            ir.Constant(10),
            ir.Constant(10.0),
            False,
        ),
        (
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
            ),
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
            ),
            True,
        ),
        (
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
                op_args=(1, 2, 3),
            ),
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
            ),
            False,
        ),
        (
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
                op_kwargs={"tuple": (1, 2, 3)},
            ),
            ir.GenericFunction(
                [EncryptedScalar(Integer(8, False))],
                lambda x: x,
                EncryptedScalar(Integer(8, False)),
                op_kind="TLU",
            ),
            False,
        ),
        (
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            True,
        ),
        (
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
            ),
            False,
        ),
    ],
)
def test_is_equivalent_to(
    node1: ir.IntermediateNode,
    node2: ir.IntermediateNode,
    expected_result: bool,
    test_helpers,
):
    """Test is_equivalent_to methods on IntermediateNodes"""
    assert (
        test_helpers.nodes_are_equivalent(node1, node2)
        == test_helpers.nodes_are_equivalent(node2, node1)
        == expected_result
    )


@pytest.mark.parametrize(
    "list_to_fill,expected_list",
    [
        pytest.param([None, 1, 2, 3, None, None], [1, 1, 2, 3, 3, 3]),
        pytest.param([None], None, marks=pytest.mark.xfail(strict=True)),
        pytest.param([None, None, None, None, 7, None, None, None], [7, 7, 7, 7, 7, 7, 7, 7]),
        pytest.param([None, None, 3, None, None, None, 2, None], [3, 3, 3, 3, 3, 2, 2, 2]),
    ],
)
def test_flood_replace_none_values(list_to_fill: list, expected_list: list):
    """Unit test for flood_replace_none_values"""

    # avoid modifying the test input
    list_to_fill_copy = deepcopy(list_to_fill)
    ir.flood_replace_none_values(list_to_fill_copy)

    assert all(value is not None for value in list_to_fill_copy)
    assert list_to_fill_copy == expected_list
