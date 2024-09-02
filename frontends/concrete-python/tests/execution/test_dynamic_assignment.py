"""
Tests of execution of dynamic assignment operation.
"""

import random

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "dtype,index,value_status,value",
    [
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5,) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5,) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5,) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5,) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 50],
            (lambda _: np.random.randint(0, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (50,) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 50],
            (lambda _: np.random.randint(-5, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (50,) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 50],
            (lambda _: np.random.randint(0, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (50,) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 50],
            (lambda _: np.random.randint(-5, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (50,) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), 0),
            "clear",
            42,
            id="x[i, 0] = 42 where x.shape = (5, 3) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), 0),
            "clear",
            42,
            id="x[i, 0] = 42 where x.shape = (5, 3) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), 0),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, 0] = y where x.shape = (5, 3) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), 0),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, 0] = y where x.shape = (5, 3) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (1, lambda _: np.random.randint(0, 5)),
            "clear",
            42,
            id="x[1, i] = 42 where x.shape = (5, 3) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (1, lambda _: np.random.randint(-5, 5)),
            "clear",
            42,
            id="x[1, i] = 42 where x.shape = (5, 3) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (1, lambda _: np.random.randint(0, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[1, i] = y where x.shape = (5, 3) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (1, lambda _: np.random.randint(-5, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[1, i] = y where x.shape = (5, 3) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), lambda _: np.random.randint(0, 3)),
            "clear",
            42,
            id="x[i, j] = 42 where x.shape = (5, 3) | 0 < i < 5 | 0 < j < 3",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), lambda _: np.random.randint(0, 3)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, j] = y where x.shape = (5, 3) | 0 < i < 5 | 0 < j < 3 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), lambda _: np.random.randint(-3, 3)),
            "clear",
            42,
            id="x[i, j] = 42 where x.shape = (5, 3) | 0 < i < 5 | -3 < j < 3",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5), lambda _: np.random.randint(-3, 3)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, j] = y where x.shape = (5, 3) | 0 < i < 5 | -3 < j < 3 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), lambda _: np.random.randint(0, 3)),
            "clear",
            42,
            id="x[i, j] = 42 where x.shape = (5, 3) | -5 < i < 5 | 0 < j < 3",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), lambda _: np.random.randint(0, 3)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, j] = y where x.shape = (5, 3) | -5 < i < 5 | 0 < j < 3 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), lambda _: np.random.randint(-3, 3)),
            "clear",
            42,
            id="x[i, j] = 42 where x.shape = (5, 3) | -5 < i < 5 | -3 < j < 3",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5), lambda _: np.random.randint(-3, 3)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i, j] = y where x.shape = (5, 3) | -5 < i < 5 | -3 < j < 3 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5, 3) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5, 3) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5, 3) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5, 3) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5),),
            "clear",
            [10, 20, 30],
            id="x[i] = [10, 20, 30] where x.shape = (5, 3) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(0, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id="x[i] = y where x.shape = (5, 3) | 0 < i < 5 | -10 < y < 10 | y.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5),),
            "clear",
            [10, 20, 30],
            id="x[i] = [10, 20, 30] where x.shape = (5, 3) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5, 3],
            (lambda _: np.random.randint(-5, 5),),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id="x[i] = y where x.shape = (5, 3) | -5 < i < 5 | -10 < y < 10 | y.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(0, 5)),
            "clear",
            42,
            id="x[:, i] = 42 where x.shape = (3, 5) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(0, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[:, i] = y where x.shape = (3, 5) | 0 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(-5, 5)),
            "clear",
            42,
            id="x[:, i] = 42 where x.shape = (3, 5) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(-5, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[:, i] = y where x.shape = (3, 5) | -5 < i < 5 | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(0, 5)),
            "clear",
            [10, 20, 30],
            id="x[:, i] = [10, 20, 30] where x.shape = (3, 5) | 0 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(0, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id="x[:, i] = y where x.shape = (3, 5) | 0 < i < 5 | -10 < y < 10 | y.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(-5, 5)),
            "clear",
            [10, 20, 30],
            id="x[:, i] = [10, 20, 30] where x.shape = (3, 5) | -5 < i < 5",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 3, 5],
            (slice(None, None, None), lambda _: np.random.randint(-5, 5)),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id="x[:, i] = y where x.shape = (3, 5) | -5 < i < 5 | -10 < y < 10 | y.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 10, 9, 8],
            (slice(1, 3, None), lambda _: np.random.randint(0, 9), slice(4, 6, None)),
            "clear",
            42,
            id="x[1:3, i, 4:6] = 42 where x.shape = (10, 9, 8) | 0 < i < 9",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 10, 9, 8],
            (slice(1, 3, None), lambda _: np.random.randint(-9, 9), slice(4, 6, None)),
            "clear",
            42,
            id="x[1:3, i, 4:6] = 42 where x.shape = (10, 9, 8) | -9 < i < 9",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 10, 9, 8],
            (
                lambda _: np.random.randint(0, 10),
                slice(2, 5, None),
                lambda _: np.random.randint(0, 8),
            ),
            "clear",
            42,
            id="x[i, 2:5, j] = 42 where x.shape = (10, 9, 8) | 0 < i < 10 | 0 < j < 8",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5, size=(3,)),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5,) | 0 < i < 5 | i.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5, size=(3,)),),
            "clear",
            42,
            id="x[i] = 42 where x.shape = (5,) | -5 < i < 5 | i.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5, size=(3,)),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5,) | 0 < i < 5 | i.shape = (3,) | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5, size=(3,)),),
            "encrypted",
            lambda _: np.random.randint(-10, 10),
            id="x[i] = y where x.shape = (5,) | -5 < i < 5 | i.shape = (3,) | -10 < y < 10",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5, size=(3,)),),
            "clear",
            [10, 20, 30],
            id="x[i] = [10, 20, 30] where x.shape = (5,) | 0 < i < 5 | i.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5, size=(3,)),),
            "clear",
            [10, 20, 30],
            id="x[i] = [10, 20, 30] where x.shape = (5,) | -5 < i < 5 | i.shape = (3,)",
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(0, 5, size=(3,)),),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id=(
                "x[i] = y where x.shape = (5,) "
                "| 0 < i < 5 | i.shape = (3,) "
                "| -10 < y < 10 | y.shape = (3,)"
            ),
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 5],
            (lambda _: np.random.randint(-5, 5, size=(3,)),),
            "encrypted",
            lambda _: np.random.randint(-10, 10, size=(3,)),
            id=(
                "x[i] = y where x.shape = (5,) "
                "| -5 < i < 5 | i.shape = (3,) "
                "| -10 < y < 10 | y.shape = (3,)"
            ),
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 20],
            (lambda _: np.random.randint(0, 20, size=(3, 2)),),
            "clear",
            [[10, 11], [20, 21], [30, 31]],
            id=(
                "x[i] = [[10, 11], [20, 21], [30, 31]] where x.shape = (20,) "
                "| 0 < i < 20 | i.shape = (3, 2)"
            ),
        ),
        pytest.param(
            fhe.tensor[fhe.int6, 20],
            (lambda _: np.random.randint(0, 20, size=(3, 2)),),
            "clear",
            [42, 24],
            id="x[i] = [42, 24] where x.shape = (20,) | 0 < i < 20 | i.shape = (3, 2)",
        ),
    ],
)
def test_dynamic_assignment(dtype, index, value_status, value, helpers):
    """
    Test dynamic assignment.
    """

    dynamic_index_positions = []
    dynamic_indices = []

    for position, indexing_element in enumerate(index):
        if callable(indexing_element):
            dynamic_indices.append(indexing_element)
            dynamic_index_positions.append(position)

    processed_index = list(index)

    def f(tensor, *args, value):
        for cursor, position in enumerate(dynamic_index_positions):
            processed_index[position] = args[cursor]

        tensor[tuple(processed_index)] = value
        return tensor

    # pylint: disable=invalid-name

    if len(dynamic_index_positions) == 1:
        if callable(value):

            def function(tensor, i0, value=value):
                return f(tensor, i0, value=value)

        else:

            def function(tensor, i0):
                return f(tensor, i0, value=value)

    elif len(dynamic_index_positions) == 2:
        if callable(value):

            def function(tensor, i0, i1, value=value):
                return f(tensor, i0, i1, value=value)

        else:

            def function(tensor, i0, i1):
                return f(tensor, i0, i1, value=value)

    elif len(dynamic_index_positions) == 3:
        if callable(value):

            def function(tensor, i0, i1, i2, value=value):
                return f(tensor, i0, i1, i2, value=value)

        else:

            def function(tensor, i0, i1, i2):
                return f(tensor, i0, i1, i2, value=value)

    elif len(dynamic_index_positions) == 4:
        if callable(value):

            def function(tensor, i0, i1, i2, i3, value=value):
                return f(tensor, i0, i1, i2, i3, value=value)

        else:

            def function(tensor, i0, i1, i2, i3):
                return f(tensor, i0, i1, i2, i3, value=value)

    else:
        message = (
            f"expected at least 1 at most 4 dynamic indexing elements "
            f"but got {len(dynamic_index_positions)}"
        )
        raise RuntimeError(message)

    # pylint: enable=invalid-name

    encryption_status = {"tensor": "encrypted"}
    inputset_types = [dtype]

    cursor = 0
    for indexing_element in index:
        if callable(indexing_element):
            encryption_status[f"i{cursor}"] = "clear"
            inputset_types.append(indexing_element)
            cursor += 1
    if callable(value):
        encryption_status["value"] = value_status
        inputset_types.append(value)

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, encryption_status)

    inputset = fhe.inputset(*inputset_types)
    circuit = compiler.compile(inputset, configuration, show_mlir=True)

    for sample in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, list(sample))
