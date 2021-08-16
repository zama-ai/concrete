"""Test file for hnumpy compilation functions"""
import itertools

import numpy
import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import EncryptedValue
from hdk.common.debugging import draw_graph, get_printable_graph
from hdk.common.extensions.table import LookupTable
from hdk.hnumpy.compile import compile_numpy_function


def no_fuse_unhandled(x, y):
    """No fuse unhandled"""
    x_intermediate = x + 2.8
    y_intermediate = y + 9.3
    intermediate = x_intermediate + y_intermediate
    return intermediate.astype(numpy.int32)


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x + 42, ((-2, 2),), ["x"]),
        pytest.param(lambda x, y: x + y + 8, ((2, 10), (4, 8)), ["x", "y"]),
        pytest.param(lambda x, y: (x + 1, y + 10), ((-1, 1), (3, 4)), ["x", "y"]),
        pytest.param(
            lambda x, y, z: (x + y + 1 - z, x * y + 42, z, z + 99),
            ((4, 8), (3, 4), (0, 4)),
            ["x", "y", "z"],
        ),
        pytest.param(
            no_fuse_unhandled,
            ((-2, 2), (-2, 2)),
            ["x", "y"],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_compile_function_multiple_outputs(function, input_ranges, list_of_arg_names):
    """Test function compile_numpy_function for a program with multiple outputs"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedValue(Integer(64, True)) for arg_name in list_of_arg_names
    }

    op_graph = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )

    # TODO: For the moment, we don't have really checks, but some printfs. Later,
    # when we have the converter, we can check the MLIR
    draw_graph(op_graph, block_until_user_closes_graph=False)

    str_of_the_graph = get_printable_graph(op_graph, show_data_types=True)
    print(f"\n{str_of_the_graph}\n")


def test_compile_function_with_direct_tlu():
    """Test compile_numpy_function for a program with direct table lookup"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return x + table[x]

    op_graph = compile_numpy_function(
        function,
        {"x": EncryptedValue(Integer(2, is_signed=False))},
        iter([(0,), (1,), (2,), (3,)]),
    )

    str_of_the_graph = get_printable_graph(op_graph, show_data_types=True)
    print(f"\n{str_of_the_graph}\n")


def test_compile_function_with_direct_tlu_overflow():
    """Test compile_numpy_function for a program with direct table lookup overflow"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return table[x]

    with pytest.raises(ValueError):
        compile_numpy_function(
            function,
            {"x": EncryptedValue(Integer(3, is_signed=False))},
            iter([(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]),
        )


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x - 10, ((-2, 2),), ["x"]),
    ],
)
def test_fail_compile(function, input_ranges, list_of_arg_names):
    """Test function compile_numpy_function for a program with signed values"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedValue(Integer(64, True)) for arg_name in list_of_arg_names
    }

    with pytest.raises(TypeError, match=r"signed integers aren't supported for MLIR lowering"):
        compile_numpy_function(
            function,
            function_parameters,
            data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        )
