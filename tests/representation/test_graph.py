"""
Tests of `Graph` class.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function,inputset,expected_result",
    [
        pytest.param(
            lambda x: x + 1,
            range(5),
            3,
        ),
        pytest.param(
            lambda x: x + 42,
            range(10),
            6,
        ),
        pytest.param(
            lambda x: x + 42,
            range(50),
            7,
        ),
        pytest.param(
            lambda x: x + 1.2,
            [1.5, 4.2],
            -1,
        ),
    ],
)
def test_graph_maximum_integer_bit_width(function, inputset, expected_result, helpers):
    """
    Test `maximum_integer_bit_width` method of `Graph` class.
    """

    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, {"x": "encrypted"})
    graph = compiler.trace(inputset, configuration)

    print(graph.format())

    assert graph.maximum_integer_bit_width() == expected_result
