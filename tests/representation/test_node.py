"""
Tests of `Node` class.
"""

import numpy as np
import pytest

from concrete.numpy.dtypes import UnsignedInteger
from concrete.numpy.representation import Node
from concrete.numpy.values import ClearScalar, EncryptedScalar, EncryptedTensor, Value


@pytest.mark.parametrize(
    "constant,expected_error,expected_message",
    [
        pytest.param(
            "abc",
            ValueError,
            "Constant 'abc' is not supported",
        ),
    ],
)
def test_node_bad_constant(constant, expected_error, expected_message):
    """
    Test `constant` function of `Node` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        Node.constant(constant)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "node,args,expected_error,expected_message",
    [
        pytest.param(
            Node.constant(1),
            ["abc"],
            ValueError,
            "Evaluation of constant '1' node using 'abc' failed "
            "because of invalid number of arguments",
        ),
        pytest.param(
            Node.generic(
                name="add",
                inputs=[Value.of(4), Value.of(10, is_encrypted=True)],
                output=Value.of(14),
                operation=lambda x, y: x + y,
            ),
            ["abc"],
            ValueError,
            "Evaluation of generic 'add' node using 'abc' failed "
            "because of invalid number of arguments",
        ),
        pytest.param(
            Node.generic(
                name="add",
                inputs=[Value.of(4), Value.of(10, is_encrypted=True)],
                output=Value.of(14),
                operation=lambda x, y: x + y,
            ),
            ["abc", "def"],
            ValueError,
            "Evaluation of generic 'add' node using 'abc', 'def' failed "
            "because argument 'abc' is not valid",
        ),
        pytest.param(
            Node.generic(
                name="add",
                inputs=[Value.of([3, 4]), Value.of(10, is_encrypted=True)],
                output=Value.of([13, 14]),
                operation=lambda x, y: x + y,
            ),
            [[1, 2, 3, 4], 10],
            ValueError,
            "Evaluation of generic 'add' node using [1, 2, 3, 4], 10 failed "
            "because argument [1, 2, 3, 4] does not have the expected shape of (2,)",
        ),
        pytest.param(
            Node.generic(
                name="unknown",
                inputs=[],
                output=Value.of(10),
                operation=lambda: "abc",
            ),
            [],
            ValueError,
            "Evaluation of generic 'unknown' node resulted in 'abc' of type str "
            "which is not acceptable either because of the type or because of overflow",
        ),
        pytest.param(
            Node.generic(
                name="unknown",
                inputs=[],
                output=Value.of(10),
                operation=lambda: np.array(["abc", "def"]),
            ),
            [],
            ValueError,
            "Evaluation of generic 'unknown' node resulted in array(['abc', 'def'], dtype='<U3') "
            "of type np.ndarray and of underlying type 'dtype[str_]' "
            "which is not acceptable because of the underlying type",
        ),
        pytest.param(
            Node.generic(
                name="unknown",
                inputs=[],
                output=Value.of(10),
                operation=lambda: [1, (), 3],
            ),
            [],
            ValueError,
            "Evaluation of generic 'unknown' node resulted in [1, (), 3] of type list "
            "which is not acceptable either because of the type or because of overflow",
        ),
        pytest.param(
            Node.generic(
                name="unknown",
                inputs=[],
                output=Value.of(10),
                operation=lambda: [1, 2, 3],
            ),
            [],
            ValueError,
            "Evaluation of generic 'unknown' node resulted in array([1, 2, 3]) "
            "which does not have the expected shape of ()",
        ),
    ],
)
def test_node_bad_call(node, args, expected_error, expected_message):
    """
    Test `__call__` method of `Node` class.
    """

    with pytest.raises(expected_error) as excinfo:
        node(*args)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "node,predecessors,expected_result",
    [
        pytest.param(
            Node.constant(1),
            [],
            "1",
        ),
        pytest.param(
            Node.input("x", EncryptedScalar(UnsignedInteger(3))),
            [],
            "x",
        ),
        pytest.param(
            Node.generic(
                name="tlu",
                inputs=[
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                ],
                output=EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                operation=lambda x, table: table[x],
                kwargs={"table": np.array([4, 1, 3, 2])},
            ),
            ["%0"],
            "tlu(%0, table=[4 1 3 2])",
        ),
        pytest.param(
            Node.generic(
                name="index.static",
                inputs=[EncryptedTensor(UnsignedInteger(3), shape=(3,))],
                output=EncryptedTensor(UnsignedInteger(3), shape=(3,)),
                operation=lambda x: x[slice(None, None, -1)],
                kwargs={"index": (slice(None, None, -1),)},
            ),
            ["%0"],
            "%0[::-1]",
        ),
        pytest.param(
            Node.generic(
                name="concatenate",
                inputs=[
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                ],
                output=EncryptedTensor(UnsignedInteger(3), shape=(3, 6)),
                operation=lambda *args, **kwargs: np.concatenate(tuple(args), **kwargs),
                kwargs={"axis": 1},
            ),
            ["%0", "%1", "%2"],
            "concatenate((%0, %1, %2), axis=1)",
        ),
        pytest.param(
            Node.generic(
                name="array",
                inputs=[
                    EncryptedScalar(UnsignedInteger(3)),
                    ClearScalar(UnsignedInteger(3)),
                    ClearScalar(UnsignedInteger(3)),
                    EncryptedScalar(UnsignedInteger(3)),
                ],
                output=EncryptedTensor(UnsignedInteger(3), shape=(2, 2)),
                operation=lambda *args: np.array(args).reshape((2, 2)),
            ),
            ["%0", "%1", "%2", "%3"],
            "array([[%0, %1], [%2, %3]])",
        ),
    ],
)
def test_node_format(node, predecessors, expected_result):
    """
    Test `format` method of `Node` class.
    """

    assert node.format(predecessors) == expected_result


@pytest.mark.parametrize(
    "node,expected_result",
    [
        pytest.param(
            Node.constant(1),
            "1",
        ),
        pytest.param(
            Node.input("x", EncryptedScalar(UnsignedInteger(3))),
            "x",
        ),
        pytest.param(
            Node.generic(
                name="tlu",
                inputs=[
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                ],
                output=EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                operation=lambda x, table: table[x],
                kwargs={"table": np.array([4, 1, 3, 2])},
            ),
            "tlu",
        ),
        pytest.param(
            Node.generic(
                name="concatenate",
                inputs=[
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                    EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                ],
                output=EncryptedTensor(UnsignedInteger(3), shape=(3, 6)),
                operation=lambda *args, **kwargs: np.concatenate(tuple(args), **kwargs),
                kwargs={"axis": -1},
            ),
            "concatenate",
        ),
    ],
)
def test_node_label(node, expected_result):
    """
    Test `label` method of `Node` class.
    """

    assert node.label() == expected_result
