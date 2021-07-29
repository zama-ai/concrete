"""Test file for HDK's hnumpy debugging functions"""

import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.debugging import draw_graph, get_printable_graph
from hdk.hnumpy import tracing


@pytest.mark.parametrize(
    "lambda_f,ref_graph_str",
    [
        (lambda x, y: x + y, "\n%0 = x\n%1 = y\n%2 = Add(0, 1)"),
        (lambda x, y: x - y, "\n%0 = x\n%1 = y\n%2 = Sub(0, 1)"),
        (
            lambda x, y: x + x - y * y * y + x,
            "\n%0 = x\n%1 = y\n%2 = Add(0, 0)\n%3 = Mul(1, 1)"
            "\n%4 = Mul(3, 1)\n%5 = Sub(2, 4)\n%6 = Add(5, 0)",
        ),
    ],
)
@pytest.mark.parametrize(
    "x_y",
    [
        pytest.param(
            (
                EncryptedValue(Integer(64, is_signed=False)),
                EncryptedValue(Integer(64, is_signed=False)),
            ),
            id="Encrypted uint",
        ),
        pytest.param(
            (
                EncryptedValue(Integer(64, is_signed=False)),
                ClearValue(Integer(64, is_signed=False)),
            ),
            id="Clear uint",
        ),
    ],
)
def test_hnumpy_print_and_draw_graph(lambda_f, ref_graph_str, x_y):
    "Test hnumpy get_printable_graph and draw_graph"
    x, y = x_y
    graph = tracing.trace_numpy_function(lambda_f, {"x": x, "y": y})

    draw_graph(graph, block_until_user_closes_graph=False)

    str_of_the_graph = get_printable_graph(graph)

    print(f"\n{str_of_the_graph}\n")

    assert str_of_the_graph == ref_graph_str
