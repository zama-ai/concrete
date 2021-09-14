"""Test file for debugging functions"""

import numpy
import pytest

from concrete.common.data_types.integers import Integer
from concrete.common.debugging import draw_graph, get_printable_graph
from concrete.common.extensions.table import LookupTable
from concrete.common.values import ClearScalar, EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing

LOOKUP_TABLE_FROM_2B_TO_4B = LookupTable([9, 2, 4, 11])
LOOKUP_TABLE_FROM_3B_TO_2B = LookupTable([0, 1, 3, 2, 2, 3, 1, 0])


def issue_130_a(x, y):
    """Test case derived from issue #130"""
    # pylint: disable=unused-argument
    intermediate = x + 1
    return (intermediate, intermediate)
    # pylint: enable=unused-argument


def issue_130_b(x, y):
    """Test case derived from issue #130"""
    # pylint: disable=unused-argument
    intermediate = x - 1
    return (intermediate, intermediate)
    # pylint: enable=unused-argument


def issue_130_c(x, y):
    """Test case derived from issue #130"""
    # pylint: disable=unused-argument
    intermediate = 1 - x
    return (intermediate, intermediate)
    # pylint: enable=unused-argument


@pytest.mark.parametrize(
    "lambda_f,ref_graph_str",
    [
        (lambda x, y: x + y, "%0 = x\n%1 = y\n%2 = Add(0, 1)\nreturn(%2)\n"),
        (lambda x, y: x - y, "%0 = x\n%1 = y\n%2 = Sub(0, 1)\nreturn(%2)\n"),
        (lambda x, y: x + x, "%0 = x\n%1 = Add(0, 0)\nreturn(%1)\n"),
        (
            lambda x, y: x + x - y * y * y + x,
            "%0 = x\n%1 = y\n%2 = Add(0, 0)\n%3 = Mul(1, 1)"
            "\n%4 = Mul(3, 1)\n%5 = Sub(2, 4)\n%6 = Add(5, 0)\nreturn(%6)\n",
        ),
        (lambda x, y: x + 1, "%0 = x\n%1 = Constant(1)\n%2 = Add(0, 1)\nreturn(%2)\n"),
        (lambda x, y: 1 + x, "%0 = x\n%1 = Constant(1)\n%2 = Add(0, 1)\nreturn(%2)\n"),
        (lambda x, y: (-1) + x, "%0 = x\n%1 = Constant(-1)\n%2 = Add(0, 1)\nreturn(%2)\n"),
        (lambda x, y: 3 * x, "%0 = x\n%1 = Constant(3)\n%2 = Mul(0, 1)\nreturn(%2)\n"),
        (lambda x, y: x * 3, "%0 = x\n%1 = Constant(3)\n%2 = Mul(0, 1)\nreturn(%2)\n"),
        (lambda x, y: x * (-3), "%0 = x\n%1 = Constant(-3)\n%2 = Mul(0, 1)\nreturn(%2)\n"),
        (lambda x, y: x - 11, "%0 = x\n%1 = Constant(11)\n%2 = Sub(0, 1)\nreturn(%2)\n"),
        (lambda x, y: 11 - x, "%0 = Constant(11)\n%1 = x\n%2 = Sub(0, 1)\nreturn(%2)\n"),
        (lambda x, y: (-11) - x, "%0 = Constant(-11)\n%1 = x\n%2 = Sub(0, 1)\nreturn(%2)\n"),
        (
            lambda x, y: x + 13 - y * (-21) * y + 44,
            "%0 = Constant(44)"
            "\n%1 = x"
            "\n%2 = Constant(13)"
            "\n%3 = y"
            "\n%4 = Constant(-21)"
            "\n%5 = Add(1, 2)"
            "\n%6 = Mul(3, 4)"
            "\n%7 = Mul(6, 3)"
            "\n%8 = Sub(5, 7)"
            "\n%9 = Add(8, 0)"
            "\nreturn(%9)\n",
        ),
        # Multiple outputs
        (
            lambda x, y: (x + 1, x + y + 2),
            "%0 = x"
            "\n%1 = Constant(1)"
            "\n%2 = Constant(2)"
            "\n%3 = y"
            "\n%4 = Add(0, 1)"
            "\n%5 = Add(0, 3)"
            "\n%6 = Add(5, 2)"
            "\nreturn(%4, %6)\n",
        ),
        (
            lambda x, y: (y, x),
            "%0 = y\n%1 = x\nreturn(%0, %1)\n",
        ),
        (
            lambda x, y: (x, x + 1),
            "%0 = x\n%1 = Constant(1)\n%2 = Add(0, 1)\nreturn(%0, %2)\n",
        ),
        (
            lambda x, y: (x + 1, x + 1),
            "%0 = x"
            "\n%1 = Constant(1)"
            "\n%2 = Constant(1)"
            "\n%3 = Add(0, 1)"
            "\n%4 = Add(0, 2)"
            "\nreturn(%3, %4)\n",
        ),
        (
            issue_130_a,
            "%0 = x\n%1 = Constant(1)\n%2 = Add(0, 1)\nreturn(%2, %2)\n",
        ),
        (
            issue_130_b,
            "%0 = x\n%1 = Constant(1)\n%2 = Sub(0, 1)\nreturn(%2, %2)\n",
        ),
        (
            issue_130_c,
            "%0 = Constant(1)\n%1 = x\n%2 = Sub(0, 1)\nreturn(%2, %2)\n",
        ),
    ],
)
@pytest.mark.parametrize(
    "x_y",
    [
        pytest.param(
            (
                EncryptedScalar(Integer(64, is_signed=False)),
                EncryptedScalar(Integer(64, is_signed=False)),
            ),
            id="Encrypted uint",
        ),
        pytest.param(
            (
                EncryptedScalar(Integer(64, is_signed=False)),
                ClearScalar(Integer(64, is_signed=False)),
            ),
            id="Clear uint",
        ),
    ],
)
def test_print_and_draw_graph(lambda_f, ref_graph_str, x_y):
    "Test get_printable_graph and draw_graph"
    x, y = x_y
    graph = tracing.trace_numpy_function(lambda_f, {"x": x, "y": y})

    draw_graph(graph, show=False)

    str_of_the_graph = get_printable_graph(graph)

    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )


@pytest.mark.parametrize(
    "lambda_f,params,ref_graph_str",
    [
        (
            lambda x: LOOKUP_TABLE_FROM_2B_TO_4B[x],
            {"x": EncryptedScalar(Integer(2, is_signed=False))},
            "%0 = x\n%1 = TLU(0)\nreturn(%1)\n",
        ),
        (
            lambda x: LOOKUP_TABLE_FROM_3B_TO_2B[x + 4],
            {"x": EncryptedScalar(Integer(2, is_signed=False))},
            "%0 = x\n%1 = Constant(4)\n%2 = Add(0, 1)\n%3 = TLU(2)\nreturn(%3)\n",
        ),
    ],
)
def test_print_and_draw_graph_with_direct_tlu(lambda_f, params, ref_graph_str):
    "Test get_printable_graph and draw_graph on graphs with direct table lookup"
    graph = tracing.trace_numpy_function(lambda_f, params)

    draw_graph(graph, show=False)

    str_of_the_graph = get_printable_graph(graph)

    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )


@pytest.mark.parametrize(
    "lambda_f,params,ref_graph_str",
    [
        # pylint: disable=unnecessary-lambda
        (
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(3,)),
                "y": EncryptedTensor(Integer(2, is_signed=False), shape=(3,)),
            },
            "%0 = x\n%1 = y\n%2 = Dot(0, 1)\nreturn(%2)\n",
        ),
        # pylint: enable=unnecessary-lambda
    ],
)
def test_print_and_draw_graph_with_dot(lambda_f, params, ref_graph_str):
    "Test get_printable_graph and draw_graph on graphs with dot"
    graph = tracing.trace_numpy_function(lambda_f, params)

    draw_graph(graph, show=False)

    str_of_the_graph = get_printable_graph(graph)

    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )


# Remark that the bitwidths are not particularly correct (eg, a MUL of a 17b times 23b
# returning 23b), since they are replaced later by the real bitwidths computed on the
# inputset
@pytest.mark.parametrize(
    "lambda_f,x_y,ref_graph_str",
    [
        (
            lambda x, y: x + y,
            (
                EncryptedScalar(Integer(64, is_signed=False)),
                EncryptedScalar(Integer(32, is_signed=True)),
            ),
            "%0 = x                                   # EncryptedScalar<Integer<unsigned, 64 bits>>"
            "\n%1 = y                                   # EncryptedScalar<Integer<signed, 32 bits>>"
            "\n%2 = Add(0, 1)                           # EncryptedScalar<Integer<signed, 65 bits>>"
            "\nreturn(%2)\n",
        ),
        (
            lambda x, y: x * y,
            (
                EncryptedScalar(Integer(17, is_signed=False)),
                EncryptedScalar(Integer(23, is_signed=False)),
            ),
            "%0 = x                                   "
            "# EncryptedScalar<Integer<unsigned, 17 bits>>"
            "\n%1 = y                                   "
            "# EncryptedScalar<Integer<unsigned, 23 bits>>"
            "\n%2 = Mul(0, 1)                           "
            "# EncryptedScalar<Integer<unsigned, 23 bits>>"
            "\nreturn(%2)\n",
        ),
    ],
)
def test_print_with_show_data_types(lambda_f, x_y, ref_graph_str):
    """Test get_printable_graph with show_data_types"""
    x, y = x_y
    graph = tracing.trace_numpy_function(lambda_f, {"x": x, "y": y})

    str_of_the_graph = get_printable_graph(graph, show_data_types=True)

    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )


@pytest.mark.parametrize(
    "lambda_f,params,ref_graph_str",
    [
        (
            lambda x: LOOKUP_TABLE_FROM_2B_TO_4B[x],
            {"x": EncryptedScalar(Integer(2, is_signed=False))},
            "%0 = x                                   "
            "# EncryptedScalar<Integer<unsigned, 2 bits>>"
            "\n%1 = TLU(0)                              "
            "# EncryptedScalar<Integer<unsigned, 4 bits>>"
            "\nreturn(%1)\n",
        ),
        (
            lambda x: LOOKUP_TABLE_FROM_3B_TO_2B[x + 4],
            {"x": EncryptedScalar(Integer(2, is_signed=False))},
            "%0 = x                                   "
            "# EncryptedScalar<Integer<unsigned, 2 bits>>"
            "\n%1 = Constant(4)                         "
            "# ClearScalar<Integer<unsigned, 3 bits>>"
            "\n%2 = Add(0, 1)                           "
            "# EncryptedScalar<Integer<unsigned, 3 bits>>"
            "\n%3 = TLU(2)                              "
            "# EncryptedScalar<Integer<unsigned, 2 bits>>"
            "\nreturn(%3)\n",
        ),
        (
            lambda x: LOOKUP_TABLE_FROM_2B_TO_4B[LOOKUP_TABLE_FROM_3B_TO_2B[x + 4]],
            {"x": EncryptedScalar(Integer(2, is_signed=False))},
            "%0 = x                                   "
            "# EncryptedScalar<Integer<unsigned, 2 bits>>"
            "\n%1 = Constant(4)                         "
            "# ClearScalar<Integer<unsigned, 3 bits>>"
            "\n%2 = Add(0, 1)                           "
            "# EncryptedScalar<Integer<unsigned, 3 bits>>"
            "\n%3 = TLU(2)                              "
            "# EncryptedScalar<Integer<unsigned, 2 bits>>"
            "\n%4 = TLU(3)                              "
            "# EncryptedScalar<Integer<unsigned, 4 bits>>"
            "\nreturn(%4)\n",
        ),
    ],
)
def test_print_with_show_data_types_with_direct_tlu(lambda_f, params, ref_graph_str):
    """Test get_printable_graph with show_data_types on graphs with direct table lookup"""
    graph = tracing.trace_numpy_function(lambda_f, params)

    draw_graph(graph, show=False)

    str_of_the_graph = get_printable_graph(graph, show_data_types=True)

    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )
