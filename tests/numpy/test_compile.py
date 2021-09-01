"""Test file for numpy compilation functions"""
import itertools
import random

import numpy
import pytest

from hdk.common.compilation import CompilationConfiguration
from hdk.common.data_types.integers import Integer
from hdk.common.debugging import draw_graph, get_printable_graph
from hdk.common.extensions.table import LookupTable
from hdk.common.values import EncryptedScalar, EncryptedTensor
from hdk.numpy.compile import (
    compile_numpy_function,
    compile_numpy_function_into_op_graph,
)


def no_fuse_unhandled(x, y):
    """No fuse unhandled"""
    x_intermediate = x + 2.8
    y_intermediate = y + 9.3
    intermediate = x_intermediate + y_intermediate
    return intermediate.astype(numpy.int32)


def lut(x):
    """Test lookup table"""
    table = LookupTable(list(range(128)))
    return table[x]


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
    """Test function compile_numpy_function_into_op_graph for a program with multiple outputs"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, True)) for arg_name in list_of_arg_names
    }

    op_graph = compile_numpy_function_into_op_graph(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
    )

    # TODO: For the moment, we don't have really checks, but some printfs. Later,
    # when we have the converter, we can check the MLIR
    draw_graph(op_graph, show=False)

    str_of_the_graph = get_printable_graph(op_graph, show_data_types=True)
    print(f"\n{str_of_the_graph}\n")


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x + 42, ((0, 2),), ["x"]),
        pytest.param(lambda x: x + numpy.int32(42), ((0, 2),), ["x"]),
        pytest.param(lambda x: x * 2, ((0, 2),), ["x"]),
        pytest.param(lambda x: 8 - x, ((0, 2),), ["x"]),
        pytest.param(lambda x, y: x + y + 8, ((2, 10), (4, 8)), ["x", "y"]),
        pytest.param(lut, ((0, 127),), ["x"]),
    ],
)
def test_compile_and_run_function_multiple_outputs(function, input_ranges, list_of_arg_names):
    """Test function compile_numpy_function for a program with multiple outputs"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    compiler_engine.run(*args)


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x + 64, ((0, 10),), ["x"]),
        pytest.param(lambda x: x * 3, ((0, 40),), ["x"]),
        pytest.param(lambda x: 120 - x, ((40, 80),), ["x"]),
        pytest.param(lambda x, y: x + y + 64, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 100 - y + x, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 50 - y * 2 + x, ((0, 20), (0, 20)), ["x", "y"]),
    ],
)
def test_compile_and_run_correctness(function, input_ranges, list_of_arg_names):
    """Test correctness of results when running a compiled function"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    assert compiler_engine.run(*args) == function(*args)


def test_compile_function_with_direct_tlu():
    """Test compile_numpy_function_into_op_graph for a program with direct table lookup"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return x + table[x]

    op_graph = compile_numpy_function_into_op_graph(
        function,
        {"x": EncryptedScalar(Integer(2, is_signed=False))},
        iter([(0,), (1,), (2,), (3,)]),
    )

    str_of_the_graph = get_printable_graph(op_graph, show_data_types=True)
    print(f"\n{str_of_the_graph}\n")


def test_compile_function_with_direct_tlu_overflow():
    """Test compile_numpy_function_into_op_graph for a program with direct table lookup overflow"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return table[x]

    with pytest.raises(ValueError):
        compile_numpy_function_into_op_graph(
            function,
            {"x": EncryptedScalar(Integer(3, is_signed=False))},
            iter([(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]),
            CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
        )


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x - 10, ((-2, 2),), ["x"]),
    ],
)
def test_fail_compile(function, input_ranges, list_of_arg_names):
    """Test function compile_numpy_function_into_op_graph for a program with signed values"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, True)) for arg_name in list_of_arg_names
    }

    with pytest.raises(TypeError, match=r"signed integers aren't supported for MLIR lowering"):
        compile_numpy_function_into_op_graph(
            function,
            function_parameters,
            data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
            CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
        )


@pytest.mark.parametrize(
    "function,params,shape,ref_graph_str",
    [
        # pylint: disable=unnecessary-lambda
        (
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(4,)),
                "y": EncryptedTensor(Integer(2, is_signed=False), shape=(4,)),
            },
            (4,),
            # Remark that, when you do the dot of tensors of 4 values between 0 and 3,
            # you can get a maximal value of 4*3*3 = 36, ie something on 6 bits
            "%0 = x                                   # Integer<unsigned, 2 bits>"
            "\n%1 = y                                   # Integer<unsigned, 2 bits>"
            "\n%2 = Dot(0, 1)                           # Integer<unsigned, 6 bits>"
            "\nreturn(%2)\n",
        ),
        # pylint: enable=unnecessary-lambda
    ],
)
def test_compile_function_with_dot(function, params, shape, ref_graph_str):
    """Test compile_numpy_function_into_op_graph for a program with np.dot"""

    # This is the exhaust, but if ever we have too long inputs (ie, large 'repeat'),
    # we'll have to take random values, not all values one by one
    def data_gen(max_for_ij, repeat):
        iter_i = itertools.product(range(0, max_for_ij + 1), repeat=repeat)
        iter_j = itertools.product(range(0, max_for_ij + 1), repeat=repeat)
        for prod_i, prod_j in itertools.product(iter_i, iter_j):
            yield (prod_i, prod_j)

    max_for_ij = 3
    assert len(shape) == 1
    repeat = shape[0]

    op_graph = compile_numpy_function_into_op_graph(
        function,
        params,
        data_gen(max_for_ij, repeat),
    )
    str_of_the_graph = get_printable_graph(op_graph, show_data_types=True)
    assert str_of_the_graph == ref_graph_str, (
        f"\n==================\nGot \n{str_of_the_graph}"
        f"==================\nExpected \n{ref_graph_str}"
        f"==================\n"
    )


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x + 64, ((0, 10),), ["x"]),
        pytest.param(lambda x: x * 3, ((0, 40),), ["x"]),
        pytest.param(lambda x: 120 - x, ((40, 80),), ["x"]),
        pytest.param(lambda x, y: x + y + 64, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 100 - y + x, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 50 - y * 2 + x, ((0, 20), (0, 20)), ["x", "y"]),
    ],
)
def test_compile_with_show_mlir(function, input_ranges, list_of_arg_names):
    """Test show_mlir option"""

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        show_mlir=True,
    )
