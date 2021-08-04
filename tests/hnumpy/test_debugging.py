"""Test file for hnumpy debugging functions"""

import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.debugging import draw_graph, get_printable_graph
from hdk.hnumpy import tracing


@pytest.mark.parametrize(
    "lambda_f,ref_graph_str",
    [
        (lambda x, y: x + y, "\n%0 = x\n%1 = y\n%2 = Add(0, 1)\nreturn(%2)"),
        (lambda x, y: x - y, "\n%0 = x\n%1 = y\n%2 = Sub(0, 1)\nreturn(%2)"),
        (lambda x, y: x + x, "\n%0 = x\n%1 = Add(0, 0)\nreturn(%1)"),
        (
            lambda x, y: x + x - y * y * y + x,
            "\n%0 = x\n%1 = y\n%2 = Add(0, 0)\n%3 = Mul(1, 1)"
            "\n%4 = Mul(3, 1)\n%5 = Sub(2, 4)\n%6 = Add(5, 0)\nreturn(%6)",
        ),
        (lambda x, y: x + 1, "\n%0 = x\n%1 = ConstantInput(1)\n%2 = Add(0, 1)\nreturn(%2)"),
        (lambda x, y: 1 + x, "\n%0 = x\n%1 = ConstantInput(1)\n%2 = Add(0, 1)\nreturn(%2)"),
        (lambda x, y: (-1) + x, "\n%0 = x\n%1 = ConstantInput(-1)\n%2 = Add(0, 1)\nreturn(%2)"),
        (lambda x, y: 3 * x, "\n%0 = x\n%1 = ConstantInput(3)\n%2 = Mul(0, 1)\nreturn(%2)"),
        (lambda x, y: x * 3, "\n%0 = x\n%1 = ConstantInput(3)\n%2 = Mul(0, 1)\nreturn(%2)"),
        (lambda x, y: x * (-3), "\n%0 = x\n%1 = ConstantInput(-3)\n%2 = Mul(0, 1)\nreturn(%2)"),
        (lambda x, y: x - 11, "\n%0 = x\n%1 = ConstantInput(11)\n%2 = Sub(0, 1)\nreturn(%2)"),
        (lambda x, y: 11 - x, "\n%0 = ConstantInput(11)\n%1 = x\n%2 = Sub(0, 1)\nreturn(%2)"),
        (lambda x, y: (-11) - x, "\n%0 = ConstantInput(-11)\n%1 = x\n%2 = Sub(0, 1)\nreturn(%2)"),
        (
            lambda x, y: x + 13 - y * (-21) * y + 44,
            "\n%0 = ConstantInput(44)"
            "\n%1 = x"
            "\n%2 = ConstantInput(13)"
            "\n%3 = y"
            "\n%4 = ConstantInput(-21)"
            "\n%5 = Add(1, 2)"
            "\n%6 = Mul(3, 4)"
            "\n%7 = Mul(6, 3)"
            "\n%8 = Sub(5, 7)"
            "\n%9 = Add(8, 0)"
            "\nreturn(%9)",
        ),
        # Multiple outputs
        (
            lambda x, y: (x + 1, x + y + 2),
            "\n%0 = x"
            "\n%1 = ConstantInput(1)"
            "\n%2 = ConstantInput(2)"
            "\n%3 = y"
            "\n%4 = Add(0, 1)"
            "\n%5 = Add(0, 3)"
            "\n%6 = Add(5, 2)"
            "\nreturn(%4, %6)",
        ),
        (
            lambda x, y: (y, x),
            "\n%0 = y\n%1 = x\nreturn(%0, %1)",
        ),
        (
            lambda x, y: (x, x + 1),
            "\n%0 = x\n%1 = ConstantInput(1)\n%2 = Add(0, 1)\nreturn(%0, %2)",
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

    print(f"\nGot {str_of_the_graph}\n")
    print(f"\nExp {ref_graph_str}\n")

    assert str_of_the_graph == ref_graph_str
