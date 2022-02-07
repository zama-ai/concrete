"""Test file for numpy compilation functions"""
import itertools
import random
from copy import deepcopy

import numpy
import pytest

from concrete.common.compilation import CompilationConfiguration
from concrete.common.data_types.integers import Integer, SignedInteger, UnsignedInteger
from concrete.common.debugging import draw_graph, format_operation_graph
from concrete.common.extensions.multi_table import MultiLookupTable
from concrete.common.extensions.table import LookupTable
from concrete.common.values import ClearTensor, EncryptedScalar, EncryptedTensor
from concrete.numpy import compile as compile_
from concrete.numpy import tracing
from concrete.numpy.compile import (
    FHECircuit,
    compile_numpy_function,
    compile_numpy_function_into_op_graph_and_measure_bounds,
)

# pylint: disable=too-many-lines


def data_gen(args):
    """Helper to create an inputset"""
    for prod in itertools.product(*args):
        yield prod if len(prod) > 1 else prod[0]


def numpy_array_data_gen(args, tensor_shapes):
    """Helper to create an inputset containing numpy arrays filled with the same value and of a
    particular shape"""
    for prod in itertools.product(*args):
        yield tuple(
            numpy.full(tensor_shape, val, numpy.int64)
            for val, tensor_shape in zip(prod, tensor_shapes)
        )


def no_fuse_unhandled(x, y):
    """No fuse unhandled"""
    x_intermediate = x + 2.8
    y_intermediate = y + 9.3
    intermediate = x_intermediate - y_intermediate
    return (intermediate * 1.5).astype(numpy.int32)


def identity_lut_generator(n):
    """Test lookup table"""
    return lambda x: LookupTable(list(range(2 ** n)))[x]


def negative_identity_smaller_lut_generator(n):
    """Test negative lookup table"""

    table = LookupTable(range(2 ** (n - 1)))
    offset = 2 ** (n - 1)

    return (lambda x: table[x + (-offset)]), table


def negative_identity_lut_generator(n):
    """Test negative lookup table (bigger than bit-width)"""

    table = LookupTable(range(2 ** n))
    offset = 2 ** (n - 1)

    return (lambda x: table[x + (-offset)]), table


def negative_identity_bigger_lut_generator(n):
    """Test negative lookup table (bigger than bit-width)"""

    table = LookupTable(range(2 ** (n + 1)))
    offset = 2 ** (n - 1)

    return (lambda x: table[x + (-offset)]), table


def weird_lut(n):
    """A weird lookup table to test an edge case related to negative indexing"""

    table = LookupTable([0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7])
    offset = 2 ** (n - 1)

    return (lambda x: table[x + (-offset)]), table


def random_lut_1b(x):
    """1-bit random table lookup"""

    # fmt: off
    table = LookupTable([10, 12])
    # fmt: on

    return table[x]


def random_lut_2b(x):
    """2-bit random table lookup"""

    # fmt: off
    table = LookupTable([3, 8, 22, 127])
    # fmt: on

    return table[x]


def random_lut_3b(x):
    """3-bit random table lookup"""

    # fmt: off
    table = LookupTable([30, 52, 125, 23, 17, 12, 90, 4])
    # fmt: on

    return table[x]


def random_lut_4b(x):
    """4-bit random table lookup"""

    # fmt: off
    table = LookupTable([30, 52, 125, 23, 17, 12, 90, 4, 21, 51, 22, 15, 53, 100, 75, 90])
    # fmt: on

    return table[x]


def random_lut_5b(x):
    """5-bit random table lookup"""

    # fmt: off
    table = LookupTable(
        [
            1, 5, 2, 3, 10, 2, 4, 8, 1, 12, 15, 12, 10, 1, 0, 2,
            4, 3, 8, 7, 10, 11, 6, 13, 9, 0, 2, 1, 15, 11, 12, 5
        ]
    )
    # fmt: on

    return table[x]


def random_lut_6b(x):
    """6-bit random table lookup"""

    # fmt: off
    table = LookupTable(
        [
            95, 74, 11, 83, 24, 116, 28, 75, 26, 85, 114, 121, 91, 123, 78, 69,
            72, 115, 67, 5, 39, 11, 120, 88, 56, 43, 74, 16, 72, 85, 103, 92,
            44, 115, 50, 56, 107, 77, 25, 71, 52, 45, 80, 35, 69, 8, 40, 87,
            26, 85, 84, 53, 73, 95, 86, 22, 16, 45, 59, 112, 53, 113, 98, 116
        ]
    )
    # fmt: on

    return table[x]


def random_lut_7b(x):
    """7-bit random table lookup"""

    # fmt: off
    table = LookupTable(
        [
            13, 58, 38, 58, 15, 15, 77, 86, 80, 94, 108, 27, 126, 60, 65, 95,
            50, 79, 22, 97, 38, 60, 25, 48, 73, 112, 27, 45, 88, 20, 67, 17,
            16, 6, 71, 60, 77, 43, 93, 40, 41, 31, 99, 122, 120, 40, 94, 13,
            111, 44, 96, 62, 108, 91, 34, 90, 103, 58, 3, 103, 19, 69, 55, 108,
            0, 111, 113, 0, 0, 73, 22, 52, 81, 2, 88, 76, 36, 121, 97, 121,
            123, 79, 82, 120, 12, 65, 54, 101, 90, 52, 84, 106, 23, 15, 110, 79,
            85, 101, 30, 61, 104, 35, 81, 30, 98, 44, 111, 32, 68, 18, 45, 123,
            84, 80, 68, 27, 31, 38, 126, 61, 51, 7, 49, 37, 63, 114, 22, 18,
        ]
    )
    # fmt: on

    return table[x]


def multi_lut(x):
    """2-bit multi table lookup"""

    table = MultiLookupTable(
        [
            [LookupTable([1, 2, 1, 0]), LookupTable([2, 2, 1, 3])],
            [LookupTable([1, 0, 1, 0]), LookupTable([0, 2, 3, 3])],
            [LookupTable([0, 2, 3, 0]), LookupTable([2, 1, 2, 0])],
        ]
    )
    return table[x]


def small_fused_table(x):
    """Test with a small fused table"""
    return (10 * (numpy.cos(x + 1) + 1)).astype(numpy.uint32)


def complicated_topology(x):
    """Mix x in an intricated way."""
    intermediate = x
    x_p_1 = intermediate + 1
    x_p_2 = intermediate + 2
    x_p_3 = x_p_1 + x_p_2
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
    )


def mix_x_and_y_and_call_f(func, x, y):
    """Create an upper function to test `func`"""
    z = numpy.abs(10 * func(x))
    z = z / 2
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_with_float_inputs(func, x, y):
    """Create an upper function to test `func`, with inputs which are forced to be floats"""
    z = numpy.abs(10 * func(x + 0.1))
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_with_integer_inputs(func, x, y):
    """Create an upper function to test `func`, with inputs which are forced to be integers but
    in a way which is fusable into a TLU"""
    x = x // 2
    a = x + 0.1
    a = numpy.rint(a).astype(numpy.int32)
    z = numpy.abs(10 * func(a))
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_which_expects_small_inputs(func, x, y):
    """Create an upper function to test `func`, which expects small values to not use too much
    precision"""
    # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/993
    # Understand why it's failing with 0.77 for numpy.arctanh
    a = numpy.abs(0.5 * numpy.sin(x))
    z = numpy.abs(3 * func(a))
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_which_has_large_outputs(func, x, y):
    """Create an upper function to test `func`, which outputs large values"""
    a = numpy.abs(2 * numpy.sin(x))
    z = numpy.abs(func(a) * 0.131)
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_avoid_0_input(func, x, y):
    """Create an upper function to test `func`, which makes that inputs are not 0"""
    a = numpy.abs(7 * numpy.sin(x)) + 1
    c = 100 // a
    b = 100 / a
    a = a + b + c
    z = numpy.abs(5 * func(a))
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_binary_f_one(func, c, x, y):
    """Create an upper function to test `func`"""
    z = numpy.abs(func(x, c) + 1)
    z = z.astype(numpy.uint32) + y
    return z


def mix_x_and_y_and_call_binary_f_two(func, c, x, y):
    """Create an upper function to test `func`"""
    z = numpy.abs(func(c, x) + 1)
    z = z.astype(numpy.uint32) + y
    return z


def negative_binary_f_one(func, c, x, y):
    """Test negative values as input to func as first argument."""
    x = x + (-4)
    z = func(x, c)
    z = numpy.clip(z, 0, 63).astype(numpy.int32) + y
    return z


def negative_binary_f_two(func, c, x, y):
    """Test negative values as input to func as second argument."""
    x = x + (-4)
    z = func(c, x)
    z = numpy.clip(z, 0, 63).astype(numpy.int32) + y
    return z


def negative_unary_f(func, x, y):
    """Test negative values as input to func."""
    x = x + (-4)
    z = func(x)
    z = numpy.clip(z, 0, 63).astype(numpy.int32) + y
    return z


def subtest_compile_and_run_unary_ufunc_correctness(
    ufunc,
    upper_function,
    input_ranges,
    tensor_shape,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function"""

    def get_function(ufunc, upper_function):
        return lambda x, y: upper_function(ufunc, x, y)

    function = get_function(ufunc, upper_function)

    function_parameters = {
        arg_name: EncryptedTensor(Integer(64, True), shape=tensor_shape) for arg_name in ["x", "y"]
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        numpy_array_data_gen(
            tuple(range(x[0], x[1] + 1) for x in input_ranges),
            [tensor_shape] * len(function_parameters),
        ),
        default_compilation_configuration,
    )

    # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/910
    args = [
        numpy.random.randint(low, high, size=tensor_shape, dtype=numpy.uint8)
        if tensor_shape != ()
        else random.randint(low, high)
        for (low, high) in input_ranges
    ]

    check_is_good_execution(compiler_engine, function, args)


def subtest_compile_and_run_binary_ufunc_correctness(
    ufunc,
    upper_function,
    c,
    input_ranges,
    tensor_shape,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function"""

    def get_function(ufunc, upper_function):
        return lambda x, y: upper_function(ufunc, c, x, y)

    function = get_function(ufunc, upper_function)

    function_parameters = {
        arg_name: EncryptedTensor(Integer(64, True), shape=tensor_shape) for arg_name in ["x", "y"]
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        numpy_array_data_gen(
            tuple(range(x[0], x[1] + 1) for x in input_ranges),
            [tensor_shape] * len(function_parameters),
        ),
        default_compilation_configuration,
    )

    # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/910
    args = [
        numpy.random.randint(low, high, size=tensor_shape, dtype=numpy.uint8)
        if tensor_shape != ()
        else random.randint(low, high)
        for (low, high) in input_ranges
    ]

    check_is_good_execution(compiler_engine, function, args)


@pytest.mark.parametrize(
    "ufunc",
    [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 2],
)
@pytest.mark.parametrize(
    "tensor_shape", [pytest.param((), id="scalar"), pytest.param((3, 1, 2), id="tensor")]
)
def test_binary_ufunc_operations(
    ufunc,
    tensor_shape,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test biary functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""

    run_multi_tlu_test = False
    if tensor_shape != ():
        run_multi_tlu_test = True
        tensor_for_multi_tlu = numpy.arange(numpy.prod(tensor_shape)).reshape(tensor_shape)
        tensor_for_multi_tlu_small_values = tensor_for_multi_tlu // 2

    if ufunc in [numpy.power, numpy.float_power]:
        # Need small constants to keep results really small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_one,
            3,
            ((0, 4), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_two,
            2,
            ((0, 4), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        if run_multi_tlu_test:
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_one,
                tensor_for_multi_tlu_small_values,
                ((0, 4), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_two,
                tensor_for_multi_tlu_small_values,
                ((0, 4), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
    elif ufunc in [numpy.floor_divide, numpy.fmod, numpy.remainder, numpy.true_divide]:
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_two,
            31,
            ((1, 5), (1, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        if run_multi_tlu_test:
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_two,
                tensor_for_multi_tlu,
                ((1, 5), (1, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
    elif ufunc in [numpy.lcm, numpy.left_shift]:
        # Need small constants to keep results sufficiently small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_one,
            3,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_two,
            2,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        if run_multi_tlu_test:
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_one,
                tensor_for_multi_tlu
                if ufunc != numpy.left_shift
                else tensor_for_multi_tlu_small_values,
                ((0, 5), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_two,
                tensor_for_multi_tlu
                if ufunc != numpy.left_shift
                else tensor_for_multi_tlu_small_values,
                ((0, 5), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
    elif ufunc in [numpy.ldexp]:
        # Need small constants to keep results sufficiently small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_two,
            2,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        if run_multi_tlu_test:
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_two,
                tensor_for_multi_tlu // 2,
                ((0, 5), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
    else:
        # General case
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_one,
            41,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_binary_f_two,
            42,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        if run_multi_tlu_test:
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_one,
                tensor_for_multi_tlu,
                ((0, 5), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )
            subtest_compile_and_run_binary_ufunc_correctness(
                ufunc,
                mix_x_and_y_and_call_binary_f_two,
                tensor_for_multi_tlu,
                ((0, 5), (0, 5)),
                tensor_shape,
                default_compilation_configuration,
                check_is_good_execution,
            )

    # Negative inputs tests on compatible functions
    if ufunc not in [
        numpy.floor_divide,
        numpy.fmod,
        numpy.remainder,
        numpy.true_divide,
        numpy.power,
        numpy.float_power,
    ]:
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            negative_binary_f_one,
            2,
            ((0, 7), (0, 3)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc,
            negative_binary_f_two,
            2,
            ((0, 7), (0, 3)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )


@pytest.mark.parametrize(
    "ufunc", [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 1]
)
@pytest.mark.parametrize(
    "tensor_shape", [pytest.param((), id="scalar"), pytest.param((3, 1, 2), id="tensor")]
)
def test_unary_ufunc_operations(
    ufunc, tensor_shape, default_compilation_configuration, check_is_good_execution
):
    """Test unary functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""

    if ufunc in [
        numpy.degrees,
        numpy.rad2deg,
    ]:
        # Need to reduce the output value, to avoid to need too much precision
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_f_which_has_large_outputs,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
    elif ufunc in [
        numpy.negative,
    ]:
        # Need to turn the input into a float
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_f_with_float_inputs,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
    elif ufunc in [
        numpy.arccosh,
        numpy.log,
        numpy.log2,
        numpy.log10,
        numpy.reciprocal,
    ]:
        # No 0 in the domain of definition
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_f_avoid_0_input,
            ((1, 5), (1, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
    elif ufunc in [
        numpy.cosh,
        numpy.exp,
        numpy.exp2,
        numpy.expm1,
        numpy.square,
        numpy.arccos,
        numpy.arcsin,
        numpy.arctanh,
        numpy.sinh,
    ]:
        # Need a small range of inputs, to avoid to need too much precision
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_f_which_expects_small_inputs,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )
    else:
        # Regular case for univariate functions
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            mix_x_and_y_and_call_f,
            ((0, 5), (0, 5)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )

    # Negative inputs tests on compatible functions
    if ufunc not in [
        numpy.arccosh,
        numpy.arccos,
        numpy.arcsin,
        numpy.arctanh,
        numpy.sqrt,
        numpy.log,
        numpy.log1p,
        numpy.log2,
        numpy.log10,
        numpy.reciprocal,
    ]:
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc,
            negative_unary_f,
            ((0, 7), (0, 3)),
            tensor_shape,
            default_compilation_configuration,
            check_is_good_execution,
        )


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x + 42, ((-5, 5),), ["x"]),
        pytest.param(lambda x, y: x + y + 8, ((2, 10), (4, 8)), ["x", "y"]),
        pytest.param(lambda x, y: (x + 1, y + 10), ((-1, 1), (3, 8)), ["x", "y"]),
        pytest.param(
            lambda x, y, z: (x + y + 1 - z, x * y + 42, z, z + 99),
            ((4, 8), (3, 4), (0, 4)),
            ["x", "y", "z"],
        ),
        pytest.param(complicated_topology, ((0, 10),), ["x"]),
    ],
)
def test_compile_function_multiple_outputs(
    function, input_ranges, list_of_arg_names, default_compilation_configuration
):
    """Test function compile_numpy_function_into_op_graph for a program with multiple outputs"""

    def data_gen_local(args):
        for prod in itertools.product(*args):
            yield tuple(numpy.array(val) for val in prod) if len(prod) > 1 else numpy.array(prod[0])

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, True)) for arg_name in list_of_arg_names
    }

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        function_parameters,
        data_gen_local(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    # TODO: For the moment, we don't have really checks, but some printfs. Later,
    # when we have the converter, we can check the MLIR
    draw_graph(op_graph, show=False)

    str_of_the_graph = format_operation_graph(op_graph)
    print(f"\n{str_of_the_graph}\n")


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: (-27) + 4 * (x + 8), ((0, 10),), ["x"]),
        pytest.param(lambda x: x + (-33), ((40, 60),), ["x"]),
        pytest.param(lambda x: 17 - (0 - x), ((0, 10),), ["x"]),
        pytest.param(lambda x: 42 + x * (-3), ((0, 10),), ["x"]),
        pytest.param(lambda x: 43 + (-4) * x, ((0, 10),), ["x"]),
        pytest.param(lambda x: 3 - (-5) * x, ((0, 10),), ["x"]),
        pytest.param(lambda x: (-2) * (-5) * x, ((0, 10),), ["x"]),
        pytest.param(lambda x: (-2) * x * (-5), ((0, 10),), ["x"]),
        pytest.param(lambda x, y: 40 - (-3 * x) + (-2 * y), ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x: x + numpy.int32(42), ((0, 10),), ["x"]),
        pytest.param(lambda x: x + 64, ((0, 10),), ["x"]),
        pytest.param(lambda x: x * 3, ((0, 40),), ["x"]),
        pytest.param(lambda x: 120 - x, ((40, 80),), ["x"]),
        pytest.param(lambda x, y: x + y + 64, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 100 - y + x, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x, y: 50 - y * 2 + x, ((0, 20), (0, 20)), ["x", "y"]),
        pytest.param(lambda x: -x + 50, ((0, 20),), ["x"]),
        pytest.param(lambda x: numpy.dot(x, 2), ((0, 20),), ["x"]),
        pytest.param(lambda x: numpy.dot(2, x), ((0, 20),), ["x"]),
    ],
)
def test_compile_and_run_correctness(
    function, input_ranges, list_of_arg_names, default_compilation_configuration
):
    """Test correctness of results when running a compiled function"""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    assert compiler_engine.run(*args) == function(*args)


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: x ** 2, ((0, 10),), ["x"]),
        pytest.param(lambda x: 2 ** (x % 5), ((0, 20),), ["x"]),
        pytest.param(lambda x: x << 1, ((0, 13),), ["x"]),
        pytest.param(lambda x: 2 << (x % 6), ((0, 13),), ["x"]),
        pytest.param(lambda x: x >> 2, ((30, 100),), ["x"]),
        pytest.param(lambda x: 115 >> (x % 3), ((0, 17),), ["x"]),
        pytest.param(lambda x: x % 7, ((0, 100),), ["x"]),
        pytest.param(lambda x: x > 7, ((0, 20),), ["x"]),
        pytest.param(lambda x: x < 11, ((0, 20),), ["x"]),
        pytest.param(lambda x: x >= 8, ((0, 20),), ["x"]),
        pytest.param(lambda x: x <= 10, ((0, 20),), ["x"]),
        pytest.param(lambda x: x == 15, ((0, 20),), ["x"]),
        pytest.param(lambda x: x & 14, ((0, 20),), ["x"]),
        pytest.param(lambda x: x | 18, ((0, 20),), ["x"]),
        pytest.param(lambda x: x ^ 23, ((0, 20),), ["x"]),
        pytest.param(lambda x: x % 3, ((0, 20),), ["x"]),
        pytest.param(lambda x: 17 & x, ((0, 20),), ["x"]),
        pytest.param(lambda x: 19 | x, ((0, 20),), ["x"]),
        pytest.param(lambda x: 45 ^ x, ((0, 20),), ["x"]),
        pytest.param(lambda x: 19 % (x + 1), ((0, 20),), ["x"]),
    ],
)
def test_compile_and_run_correctness__for_prog_with_tlu(
    function,
    input_ranges,
    list_of_arg_names,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function which uses a TLU"""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    for _ in range(16):
        args = [random.randint(low, high) for (low, high) in input_ranges]
        check_is_good_execution(compiler_engine, function, args, verbose=False)


@pytest.mark.parametrize(
    "function,parameters,inputset,test_input,use_check_good_exec",
    [
        pytest.param(
            lambda x: x + 1,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: x + numpy.array([[1, 0], [2, 0], [3, 1]], dtype=numpy.uint32),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                "y": EncryptedScalar(UnsignedInteger(3)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(3, 2)),
                    random.randint(0, (2 ** 3) - 1),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
                2,
            ),
            False,
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(3, 2)),
                    numpy.random.randint(0, 2 ** 3, size=(3, 2)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
                [
                    [1, 6],
                    [2, 5],
                    [3, 4],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: 100 - x,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.array([[10, 15], [20, 15], [10, 30]], dtype=numpy.uint32) - x,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: x * 2,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: x * numpy.array([[1, 2], [2, 1], [3, 1]], dtype=numpy.uint32),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [4, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: LookupTable([2, 1, 3, 0])[x],
            {
                "x": EncryptedTensor(UnsignedInteger(2), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 2, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 1],
                    [2, 1],
                    [3, 0],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: numpy.dot(x, 2),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            ([2, 7, 1],),
            False,
        ),
        pytest.param(
            lambda x: numpy.dot(2, x),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            ([2, 7, 1],),
            False,
        ),
        pytest.param(
            lambda x: x + x.shape[0],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            ([2, 1, 3],),
            False,
        ),
        pytest.param(
            lambda x: numpy.clip(x, 1, 5),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: numpy.clip(x + (-4), -3, 5) + 3,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: x.clip(1, 5),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: (x + (-4)).clip(-3, 5) + 3,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for _ in range(10)],
            (
                [
                    [0, 7],
                    [6, 1],
                    [2, 5],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: numpy.array([120, 60, 30], dtype=numpy.uint8) // x,
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(1, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            True,
        ),
        pytest.param(
            lambda x: numpy.sum(x),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=0),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=1),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=(0, 1)),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=0, keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=1, keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=(0, 1), keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=-1, keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=-2, keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis=(-2, -1), keepdims=True),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [numpy.random.randint(0, 2 ** 3, size=(2, 3)) for _ in range(10)],
            (
                [
                    [1, 7, 6],
                    [3, 2, 5],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: numpy.concatenate((x, y)),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4, 2)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(4, 2)),
                    numpy.random.randint(0, 2 ** 3, size=(3, 2)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                ],
                [
                    [4, 5],
                    [2, 3],
                    [0, 1],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: numpy.concatenate((x, y), axis=1),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 4)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(2, 4)),
                    numpy.random.randint(0, 2 ** 3, size=(2, 3)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
                [
                    [5, 4, 3],
                    [2, 1, 0],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: numpy.concatenate((x, y), axis=-1),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(2, 4)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(2, 4)),
                    numpy.random.randint(0, 2 ** 3, size=(2, 3)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
                [
                    [5, 4, 3],
                    [2, 1, 0],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: numpy.concatenate((x, y), axis=-2),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4, 2)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(3, 2)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(4, 2)),
                    numpy.random.randint(0, 2 ** 3, size=(3, 2)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                ],
                [
                    [4, 5],
                    [2, 3],
                    [0, 1],
                ],
            ),
            False,
        ),
        pytest.param(
            lambda x, y: numpy.concatenate((x, y), axis=None),
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3, 4)),
                "y": EncryptedTensor(UnsignedInteger(3), shape=(2, 3)),
            },
            [
                (
                    numpy.random.randint(0, 2 ** 3, size=(3, 4)),
                    numpy.random.randint(0, 2 ** 3, size=(2, 3)),
                )
                for _ in range(10)
            ],
            (
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [7, 6, 5, 4],
                ],
                [
                    [5, 4, 3],
                    [2, 1, 0],
                ],
            ),
            False,
        ),
    ],
)
def test_compile_and_run_tensor_correctness(
    function,
    parameters,
    inputset,
    test_input,
    use_check_good_exec,
    default_compilation_configuration,
    check_is_good_execution,
    check_array_equality,
):
    """Test correctness of results when running a compiled function with tensor operators"""
    circuit = compile_numpy_function(
        function,
        parameters,
        inputset,
        default_compilation_configuration,
    )

    numpy_test_input = tuple(
        item if isinstance(item, int) else numpy.array(item, dtype=numpy.uint8)
        for item in test_input
    )

    if use_check_good_exec:
        check_is_good_execution(circuit, function, numpy_test_input)
    else:
        check_array_equality(
            circuit.run(*numpy_test_input),
            numpy.array(function(*numpy_test_input), dtype=numpy.uint8),
        )


@pytest.mark.parametrize(
    "size, input_range",
    [
        pytest.param(
            1,
            (0, 8),
        ),
        pytest.param(
            4,
            (0, 5),
        ),
        pytest.param(
            6,
            (0, 4),
        ),
        pytest.param(
            10,
            (0, 3),
        ),
    ],
)
def test_compile_and_run_dot_correctness(size, input_range, default_compilation_configuration):
    """Test correctness of results when running a compiled function"""

    low, high = input_range
    shape = (size,)

    inputset = [
        (numpy.zeros(shape, dtype=numpy.uint32), numpy.zeros(shape, dtype=numpy.uint32)),
        (
            numpy.ones(shape, dtype=numpy.uint32) * high,
            numpy.ones(shape, dtype=numpy.uint32) * high,
        ),
    ]
    for _ in range(8):
        inputset.append(
            (
                numpy.random.randint(low, high + 1, size=shape),
                numpy.random.randint(low, high + 1, size=shape),
            )
        )

    function_parameters = {
        "x": EncryptedTensor(Integer(64, False), shape),
        "y": ClearTensor(Integer(64, False), shape),
    }

    def function(x, y):
        return numpy.dot(x, y)

    def function_indirect_args(x, y):
        return numpy.dot(x.flatten(), y.flatten())

    for func_to_compile in [function, function_indirect_args]:
        compiler_engine = compile_numpy_function(
            func_to_compile,
            function_parameters,
            inputset,
            default_compilation_configuration,
        )

        args = [
            numpy.random.randint(low, high + 1, size=shape, dtype=numpy.uint8) for __ in range(2)
        ]
        assert compiler_engine.run(*args) == func_to_compile(*args)


@pytest.mark.parametrize(
    "size, input_range_x, input_range_y",
    [
        pytest.param(6, (0, 3), (-3, 3)),
        pytest.param(3, (0, 3), (-7, 7)),
    ],
)
def test_compile_and_run_dot_correctness_with_signed_cst(
    size, input_range_x, input_range_y, default_compilation_configuration
):
    """Test correctness of dot with signed constant tensor."""

    low_x, high_x = input_range_x
    low_y, high_y = input_range_y
    shape = (size,)

    # Check that never, the dot goes too high
    # For this, we simplify our check knowing that low_x >= 0. Under this condition, the maximal
    # value is for the dot is size * max(abs(high_x * low_y), abs(high_x * high_y)). And we want
    # is to be less than 64, to have a signed value on strictly less than 8b
    assert low_x >= 0
    assert size * max(abs(high_x * low_y), abs(high_x * high_y)) < 64

    function_parameters = {
        "x": EncryptedTensor(Integer(64, False), shape),
    }

    constant1 = numpy.random.randint(low_y, high_y + 1, size=(size,))
    constant2 = numpy.random.randint(low_y, high_y + 1, size=(size,))

    worst_x_1_1 = numpy.where(constant1 < 0, 0, high_x)
    worst_x_1_2 = numpy.where(constant1 > 0, 0, high_x)

    worst_x_2_1 = numpy.where(constant2 < 0, 0, high_x)
    worst_x_2_2 = numpy.where(constant2 > 0, 0, high_x)

    for i in range(2):

        inputset = [
            numpy.zeros(shape, dtype=numpy.uint32),
            numpy.ones(shape, dtype=numpy.uint32) * low_x,
            numpy.ones(shape, dtype=numpy.uint32) * high_x,
        ]

        for _ in range(128):
            inputset.append(numpy.random.randint(low_x, high_x + 1, size=shape))

        if i == 0:

            def function(x):
                return numpy.dot(x, constant1)

            inputset.extend([worst_x_1_1, worst_x_1_2])

        else:

            def function(x):
                return numpy.dot(constant2, x)

            inputset.extend([worst_x_2_1, worst_x_2_2])

        compiler_engine = compile_numpy_function(
            function, function_parameters, inputset, default_compilation_configuration
        )

        # compute modulus used for the output
        output_bit_width = compiler_engine.op_graph.output_nodes[0].outputs[0].dtype.bit_width
        # bit width + 1 padding bit
        modulus = 2 ** (output_bit_width + 1)

        for _ in range(5):
            args = [
                numpy.random.randint(low_x, high_x + 1, size=(size,), dtype=numpy.uint8),
            ]
            assert check_equality_modulo(compiler_engine.run(*args), function(*args), modulus)


@pytest.mark.parametrize(
    "size,input_range",
    [
        pytest.param(
            1,
            (0, 8),
        ),
        pytest.param(
            4,
            (0, 5),
        ),
        pytest.param(
            6,
            (0, 4),
        ),
        pytest.param(
            10,
            (0, 3),
        ),
    ],
)
def test_compile_and_run_constant_dot_correctness(
    size, input_range, default_compilation_configuration
):
    """Test correctness of results when running a compiled function"""

    low, high = input_range
    shape = (size,)

    inputset = [
        numpy.zeros(shape, dtype=numpy.uint32),
        numpy.ones(shape, dtype=numpy.uint32) * high,
    ]
    for _ in range(8):
        inputset.append(numpy.random.randint(low, high + 1))

    constant = numpy.random.randint(low, high + 1, size=shape)

    def left(x):
        return numpy.dot(x, constant)

    def right(x):
        return numpy.dot(constant, x)

    left_circuit = compile_numpy_function(
        left,
        {"x": EncryptedTensor(Integer(64, False), shape)},
        inputset,
        default_compilation_configuration,
    )
    right_circuit = compile_numpy_function(
        right,
        {"x": EncryptedTensor(Integer(64, False), shape)},
        inputset,
        default_compilation_configuration,
    )

    args = (numpy.random.randint(low, high + 1, size=shape, dtype=numpy.uint8),)
    assert left_circuit.run(*args) == left(*args)
    assert right_circuit.run(*args) == right(*args)


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,input_range_inclusive_bound",
    [
        pytest.param(
            (3, 2),
            (2, 3),
            (0, 3),
        ),
        pytest.param(
            (1, 2),
            (2, 1),
            (0, 3),
        ),
        pytest.param(
            (3, 3),
            (3, 3),
            (0, 3),
        ),
        pytest.param(
            (2, 1),
            (1, 2),
            (0, 7),
        ),
        pytest.param(
            (2,),
            (2,),
            (0, 7),
        ),
        pytest.param(
            (5, 5),
            (5,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (5, 5),
            (0, 3),
        ),
        pytest.param(
            (3, 2),
            (2, 3),
            (-4, 3),
        ),
        pytest.param(
            (5,),
            (5, 3),
            (0, 3),
        ),
        pytest.param(
            (5, 3),
            (3,),
            (0, 3),
        ),
        pytest.param(
            (5,),
            (4, 5, 3),
            (0, 5),
        ),
        pytest.param(
            (4, 5, 3),
            (3,),
            (0, 5),
        ),
        pytest.param(
            (5,),
            (2, 4, 5, 3),
            (0, 5),
        ),
        pytest.param(
            (2, 4, 5, 3),
            (3,),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 5, 4, 3),
            (3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (5, 4, 3),
            (2, 1, 3, 2),
            (0, 5),
        ),
        pytest.param(
            (2, 1, 4, 3),
            (5, 3, 2),
            (0, 5),
        ),
    ],
)
def test_compile_and_run_matmul_correctness(
    lhs_shape,
    rhs_shape,
    input_range_inclusive_bound,
    default_compilation_configuration,
    check_array_equality,
):
    """Test correctness of results when running a compiled function"""

    low, high = input_range_inclusive_bound

    check_mod = low < 0 or high < 0

    max_abs = max(abs(low), abs(high))

    # Inputset for x as lhs of matmul
    lhs_inputset = [
        numpy.zeros(lhs_shape, dtype=numpy.uint32),
        numpy.ones(lhs_shape, dtype=numpy.uint32) * high,
    ]
    # Inputset for x as rhs of matmul
    rhs_inputset = [
        numpy.zeros(rhs_shape, dtype=numpy.uint32),
        numpy.ones(rhs_shape, dtype=numpy.uint32) * high,
    ]
    for _ in range(8):
        lhs_inputset.append(numpy.random.randint(low, high + 1, size=lhs_shape))
        rhs_inputset.append(numpy.random.randint(low, high + 1, size=rhs_shape))

    left_constant = numpy.random.randint(low, high + 1, size=lhs_shape)
    right_constant = numpy.random.randint(low, high + 1, size=rhs_shape)

    # Generate worst case inputsets for bit widths, replacing negative values by 0 and putting
    # the max value elsewhere, and then doing the same for positive values
    rhs_inputset.extend(
        [
            numpy.where(right_constant < 0, 0, max_abs),
            numpy.where(right_constant > 0, 0, max_abs),
        ]
    )
    lhs_inputset.extend(
        [
            numpy.where(left_constant < 0, 0, max_abs),
            numpy.where(left_constant > 0, 0, max_abs),
        ]
    )

    # Keep inputset positive
    rhs_inputset = [numpy.clip(val, 0, high) for val in rhs_inputset]
    lhs_inputset = [numpy.clip(val, 0, high) for val in lhs_inputset]

    def get_output_mod(circuit: FHECircuit):
        assert len(circuit.op_graph.output_nodes) == 1
        assert isinstance(
            output_dtype := circuit.op_graph.get_ordered_outputs()[0].outputs[0].dtype, Integer
        )
        return 2 ** output_dtype.bit_width

    def using_operator_left(x):
        return x @ right_constant

    def using_function_left(x):
        return numpy.matmul(x, right_constant)

    def using_operator_right(x):
        return left_constant @ x

    def using_function_right(x):
        return numpy.matmul(left_constant, x)

    operator_left_circuit = compile_numpy_function(
        using_operator_left,
        {"x": EncryptedTensor(UnsignedInteger(3), lhs_shape)},
        lhs_inputset,
        default_compilation_configuration,
    )
    function_left_circuit = compile_numpy_function(
        using_function_left,
        {"x": EncryptedTensor(UnsignedInteger(3), lhs_shape)},
        lhs_inputset,
        default_compilation_configuration,
    )
    operator_right_circuit = compile_numpy_function(
        using_operator_right,
        {"x": EncryptedTensor(UnsignedInteger(3), rhs_shape)},
        rhs_inputset,
        default_compilation_configuration,
    )
    function_right_circuit = compile_numpy_function(
        using_function_right,
        {"x": EncryptedTensor(UnsignedInteger(3), rhs_shape)},
        rhs_inputset,
        default_compilation_configuration,
    )

    def check_result(circuit: FHECircuit, func, arg):
        # Stay positive for input to FHE circuit
        arg = numpy.clip(arg, 0, high).astype(numpy.uint8)

        circuit_output = circuit.run(arg)
        func_output = func(arg)

        if check_mod:
            output_mod = get_output_mod(circuit)

            circuit_output %= output_mod
            func_output %= output_mod

        check_array_equality(circuit_output, func_output)

    arg = numpy.random.randint(low, high + 1, size=lhs_shape)
    check_result(operator_left_circuit, using_operator_left, arg)
    check_result(function_left_circuit, using_function_left, arg)

    arg = numpy.random.randint(low, high + 1, size=rhs_shape)
    check_result(operator_right_circuit, using_operator_right, arg)
    check_result(function_right_circuit, using_function_right, arg)


@pytest.mark.parametrize(
    "function,input_bits,list_of_arg_names",
    [
        pytest.param(identity_lut_generator(1), (1,), ["x"], id="identity function (1-bit)"),
        pytest.param(identity_lut_generator(2), (2,), ["x"], id="identity function (2-bit)"),
        pytest.param(identity_lut_generator(3), (3,), ["x"], id="identity function (3-bit)"),
        pytest.param(identity_lut_generator(4), (4,), ["x"], id="identity function (4-bit)"),
        pytest.param(identity_lut_generator(5), (5,), ["x"], id="identity function (5-bit)"),
        pytest.param(identity_lut_generator(6), (6,), ["x"], id="identity function (6-bit)"),
        pytest.param(identity_lut_generator(7), (7,), ["x"], id="identity function (7-bit)"),
        pytest.param(random_lut_1b, (1,), ["x"], id="random function (1-bit)"),
        pytest.param(random_lut_2b, (2,), ["x"], id="random function (2-bit)"),
        pytest.param(random_lut_3b, (3,), ["x"], id="random function (3-bit)"),
        pytest.param(random_lut_4b, (4,), ["x"], id="random function (4-bit)"),
        pytest.param(random_lut_5b, (5,), ["x"], id="random function (5-bit)"),
        pytest.param(random_lut_6b, (6,), ["x"], id="random function (6-bit)"),
        pytest.param(random_lut_7b, (7,), ["x"], id="random function (7-bit)"),
        pytest.param(small_fused_table, (5,), ["x"], id="small fused table (5-bits)"),
    ],
)
def test_compile_and_run_lut_correctness(
    function,
    input_bits,
    list_of_arg_names,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function with LUT"""

    input_ranges = tuple((0, 2 ** input_bit - 1) for input_bit in input_bits)

    function_parameters = {
        arg_name: EncryptedScalar(Integer(input_bit, False))
        for input_bit, arg_name in zip(input_bits, list_of_arg_names)
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    # testing random values
    for _ in range(10):
        args = [random.randint(low, high) for (low, high) in input_ranges]
        check_is_good_execution(compiler_engine, function, args)

    # testing low values
    args = [low for (low, _) in input_ranges]
    check_is_good_execution(compiler_engine, function, args)

    # testing high values
    args = [high for (_, high) in input_ranges]
    check_is_good_execution(compiler_engine, function, args)


@pytest.mark.parametrize(
    "function,table,bit_width",
    [
        pytest.param(*negative_identity_smaller_lut_generator(n), n, id=f"smaller ({n}-bit)")
        for n in range(1, 8)
    ]
    + [
        pytest.param(*negative_identity_lut_generator(n), n, id=f"normal ({n}-bit)")
        for n in range(1, 8)
    ]
    + [
        pytest.param(*negative_identity_bigger_lut_generator(n), n, id=f"bigger ({n}-bit)")
        for n in range(1, 7)
    ]
    + [
        pytest.param(*weird_lut(3), 3, id="weird"),
    ],
)
def test_compile_and_run_negative_lut_correctness(
    function,
    table,
    bit_width,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness when running a compiled function with LUT using negative values"""

    circuit = compile_numpy_function(
        function,
        {"x": EncryptedScalar(UnsignedInteger(bit_width))},
        range(2 ** bit_width),
        default_compilation_configuration,
    )

    offset = 2 ** (bit_width - 1)
    for value in range(-offset, offset):
        assert table[value] == function(value + offset)
        check_is_good_execution(circuit, function, [value + offset])


def test_compile_and_run_multi_lut_correctness(
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function with Multi LUT"""

    def function_to_compile(x):
        table = MultiLookupTable(
            [
                [LookupTable([1, 2, 1, 0]), LookupTable([2, 2, 1, 3])],
                [LookupTable([1, 0, 1, 0]), LookupTable([0, 2, 3, 3])],
                [LookupTable([0, 2, 3, 0]), LookupTable([2, 1, 2, 0])],
            ]
        )
        return table[x]

    compiler_engine = compile_numpy_function(
        function_to_compile,
        {
            "x": EncryptedTensor(UnsignedInteger(2), shape=(3, 2)),
        },
        [numpy.random.randint(0, 2 ** 2, size=(3, 2)) for _ in range(10)],
        default_compilation_configuration,
    )

    # testing random values
    for _ in range(10):
        args = [numpy.random.randint(0, 2 ** 2, size=(3, 2), dtype=numpy.uint8)]
        check_is_good_execution(compiler_engine, function_to_compile, args)


def test_compile_function_with_direct_tlu(default_compilation_configuration):
    """Test compile_numpy_function_into_op_graph for a program with direct table lookup"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return x + table[x]

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        {"x": EncryptedScalar(Integer(2, is_signed=False))},
        range(4),
        default_compilation_configuration,
    )

    str_of_the_graph = format_operation_graph(op_graph)
    print(f"\n{str_of_the_graph}\n")


def test_compile_function_with_direct_tlu_overflow(default_compilation_configuration):
    """Test compile_numpy_function_into_op_graph for a program with direct table lookup overflow"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return table[x]

    with pytest.raises(ValueError):
        compile_numpy_function_into_op_graph_and_measure_bounds(
            function,
            {"x": EncryptedScalar(Integer(3, is_signed=False))},
            range(8),
            default_compilation_configuration,
        )


# pylint: disable=line-too-long
@pytest.mark.parametrize(
    "function,parameters,inputset,error,match",
    [
        pytest.param(
            lambda x: numpy.dot(x, numpy.array([-1.5])),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(1,)),
            },
            [numpy.array([i]) for i in [1, 1, 0, 0, 1, 1, 0, 0, 2, 2]],
            RuntimeError,
            (
                """

 function you are trying to compile isn't supported for MLIR lowering

%0 = x                  # EncryptedTensor<uint2, shape=(1,)>
%1 = [-1.5]             # ClearTensor<float64, shape=(1,)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%2 = dot(%0, %1)        # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer dot product is supported
return %2
                 """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            no_fuse_unhandled,
            {"x": EncryptedScalar(Integer(2, False)), "y": EncryptedScalar(Integer(2, False))},
            [(numpy.array(i), numpy.array(i)) for i in range(10)],
            RuntimeError,
            (
                """

function you are trying to compile isn't supported for MLIR lowering

%0 = 1.5                            # ClearScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%1 = x                              # EncryptedScalar<uint4>
%2 = 2.8                            # ClearScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%3 = y                              # EncryptedScalar<uint4>
%4 = 9.3                            # ClearScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%5 = add(%1, %2)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer addition is supported
%6 = add(%3, %4)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer addition is supported
%7 = sub(%5, %6)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer subtraction is supported
%8 = mul(%7, %0)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer multiplication is supported
%9 = astype(%8, dtype=int32)        # EncryptedScalar<int5>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ astype with floating-point inputs is required to be fused to be supported
return %9

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.ravel(x),
            {"x": EncryptedTensor(Integer(3, is_signed=False), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)],
            RuntimeError,
            (
                """

function you are trying to compile isn't supported for MLIR lowering

%0 = x                # EncryptedTensor<uint3, shape=(3, 2)>
%1 = ravel(%0)        # EncryptedTensor<uint3, shape=(6,)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ravel is not supported for the time being
return %1

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.sum(x),
            {"x": EncryptedScalar(UnsignedInteger(3))},
            [numpy.random.randint(0, 2 ** 3) for i in range(10)],
            ValueError,
            (
                """

only encrypted tensor sum is supported but you tried to sum EncryptedScalar<uint3>

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.sum(x, axis="abc"),  # type: ignore
            {"x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)],
            ValueError,
            (
                """

invalid sum on EncryptedTensor<uint3, shape=(3, 2)> with axis=abc

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.sum(x),  # type: ignore
            {"x": ClearTensor(UnsignedInteger(3), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)],
            ValueError,
            (
                """

only encrypted tensor sum is supported but you tried to sum ClearTensor<uint3, shape=(3, 2)>

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.concatenate((x, x)),
            {"x": EncryptedScalar(UnsignedInteger(3))},
            [numpy.random.randint(0, 2 ** 3) for i in range(10)],
            ValueError,
            (
                """

only encrypted tensor concatenation is supported but you tried to concatenate EncryptedScalar<uint3>, EncryptedScalar<uint3>

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.concatenate((x, x), axis="abc"),  # type: ignore
            {"x": EncryptedTensor(UnsignedInteger(3), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)],
            ValueError,
            (
                """

invalid concatenation of EncryptedTensor<uint3, shape=(3, 2)>, EncryptedTensor<uint3, shape=(3, 2)> with axis=abc

                """.strip()  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: numpy.concatenate((x, x)),  # type: ignore
            {"x": ClearTensor(UnsignedInteger(3), shape=(3, 2))},
            [numpy.random.randint(0, 2 ** 3, size=(3, 2)) for i in range(10)],
            ValueError,
            (
                """

only encrypted tensor concatenation is supported but you tried to concatenate ClearTensor<uint3, shape=(3, 2)>, ClearTensor<uint3, shape=(3, 2)>

                """.strip()  # noqa: E501
            ),
        ),
    ],
)
# pylint: enable=line-too-long
def test_fail_compile(
    function,
    parameters,
    inputset,
    error,
    match,
    default_compilation_configuration,
):
    """Test function compile_numpy_function_into_op_graph for a program with signed values"""

    with pytest.raises(error) as excinfo:
        compile_numpy_function(
            function,
            parameters,
            inputset,
            default_compilation_configuration,
        )

    assert str(excinfo.value) == match, str(excinfo.value)


@pytest.mark.parametrize(
    "function,parameters,inputset,match",
    [
        pytest.param(
            lambda x: (x * 1.5)[0, 1],
            {"x": EncryptedTensor(SignedInteger(3), shape=(2, 2))},
            [numpy.random.randint(-4, 3, size=(2, 2)) for i in range(10)],
            (
                """

function you are trying to compile isn't supported for MLIR lowering

%0 = x                  # EncryptedTensor<int3, shape=(2, 2)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only unsigned integer inputs are supported
%1 = 1.5                # ClearScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%2 = mul(%0, %1)        # EncryptedTensor<float64, shape=(2, 2)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer multiplication is supported
%3 = %2[0, 1]           # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer outputs are supported
return %3

                """.strip()  # noqa: E501
            ),
        ),
    ],
)
def test_fail_compile_while_fusing_is_disabled(
    function, parameters, inputset, match, default_compilation_configuration
):
    """Test compile_numpy_function without fusing and with failing inputs"""

    configuration_to_use = deepcopy(default_compilation_configuration)
    configuration_to_use.enable_topological_optimizations = False

    with pytest.raises(RuntimeError) as excinfo:
        compile_numpy_function(
            function,
            parameters,
            inputset,
            configuration_to_use,
        )

    assert str(excinfo.value) == match, str(excinfo.value)


def test_small_inputset_no_fail():
    """Test function compile_numpy_function_into_op_graph with an unacceptably small inputset"""
    compile_numpy_function_into_op_graph_and_measure_bounds(
        lambda x: x + 42,
        {"x": EncryptedScalar(Integer(5, is_signed=False))},
        [0, 3],
        CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
    )


def test_small_inputset_treat_warnings_as_errors():
    """Test function compile_numpy_function_into_op_graph with an unacceptably small inputset"""
    with pytest.raises(ValueError, match=".* inputset contains too few inputs .*"):
        compile_numpy_function_into_op_graph_and_measure_bounds(
            lambda x: x + 42,
            {"x": EncryptedScalar(Integer(5, is_signed=False))},
            [0, 3],
            CompilationConfiguration(
                dump_artifacts_on_unexpected_failures=False,
                treat_warnings_as_errors=True,
            ),
        )


@pytest.mark.parametrize(
    "function,params,shape,ref_graph_str",
    [
        (
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(4,)),
                "y": EncryptedTensor(Integer(2, is_signed=False), shape=(4,)),
            },
            (4,),
            # Remark that, when you do the dot of tensors of 4 values between 0 and 3,
            # you can get a maximal value of 4*3*3 = 36, ie something on 6 bits
            """

%0 = x                  # EncryptedTensor<uint2, shape=(4,)>
%1 = y                  # EncryptedTensor<uint2, shape=(4,)>
%2 = dot(%0, %1)        # EncryptedScalar<uint6>
return %2

            """.strip(),
        ),
    ],
)
def test_compile_function_with_dot(
    function, params, shape, ref_graph_str, default_compilation_configuration
):
    """Test compile_numpy_function_into_op_graph for a program with np.dot"""

    # This is the exhaust, but if ever we have too long inputs (ie, large 'repeat'),
    # we'll have to take random values, not all values one by one
    def data_gen_local(max_for_ij, repeat):
        iter_i = itertools.product(range(0, max_for_ij + 1), repeat=repeat)
        iter_j = itertools.product(range(0, max_for_ij + 1), repeat=repeat)
        for prod_i, prod_j in itertools.product(iter_i, iter_j):
            yield numpy.array(prod_i), numpy.array(prod_j)

    max_for_ij = 3
    assert len(shape) == 1
    repeat = shape[0]

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        params,
        data_gen_local(max_for_ij, repeat),
        default_compilation_configuration,
    )
    str_of_the_graph = format_operation_graph(op_graph)
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
def test_compile_with_show_mlir(
    function, input_ranges, list_of_arg_names, default_compilation_configuration
):
    """Test show_mlir option"""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
        show_mlir=True,
    )


def test_compile_too_high_bitwidth(default_compilation_configuration):
    """Check that the check of maximal bitwidth of intermediate data works fine."""

    def function(x, y):
        return x + y

    function_parameters = {
        "x": EncryptedScalar(Integer(64, False)),
        "y": EncryptedScalar(Integer(64, False)),
    }

    # A bit too much
    input_ranges = [(0, 100), (0, 28)]

    with pytest.raises(RuntimeError) as excinfo:
        compile_numpy_function(
            function,
            function_parameters,
            data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
            default_compilation_configuration,
        )

    assert (
        str(excinfo.value)
        == """

max_bit_width of some nodes is too high for the current version of the compiler (maximum must be 7) which is not compatible with:

%0 = x                  # EncryptedScalar<uint7>
%1 = y                  # EncryptedScalar<uint5>
%2 = add(%0, %1)        # EncryptedScalar<uint8>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 8 bits is not supported for the time being
return %2

        """.strip()  # noqa: E501 # pylint: disable=line-too-long
    )

    # Just ok
    input_ranges = [(0, 99), (0, 28)]

    compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )


def test_compile_with_random_inputset(default_compilation_configuration):
    """Test function for compile with random input set"""

    configuration_to_use = deepcopy(default_compilation_configuration)
    configuration_to_use.enable_unsafe_features = True

    compile_numpy_function_into_op_graph_and_measure_bounds(
        lambda x: x + 1,
        {"x": EncryptedScalar(UnsignedInteger(6))},
        inputset="random",
        compilation_configuration=configuration_to_use,
    )
    compile_numpy_function(
        lambda x: x + 32,
        {"x": EncryptedScalar(UnsignedInteger(6))},
        inputset="random",
        compilation_configuration=configuration_to_use,
    )


def test_fail_compile_with_random_inputset():
    """Test function for failed compile with random input set"""

    compilation_configuration = CompilationConfiguration(
        dump_artifacts_on_unexpected_failures=False,
        treat_warnings_as_errors=True,
    )

    with pytest.raises(ValueError):
        try:
            compile_numpy_function_into_op_graph_and_measure_bounds(
                lambda x: x + 1,
                {"x": EncryptedScalar(UnsignedInteger(3))},
                inputset="unsupported",
                compilation_configuration=compilation_configuration,
            )
        except Exception as error:
            expected = (
                "inputset can only be an iterable of tuples or the string 'random' "
                "but you specified 'unsupported' for it"
            )
            assert str(error) == expected
            raise

    with pytest.raises(RuntimeError):
        try:
            compile_numpy_function(
                lambda x: x + 1,
                {"x": EncryptedScalar(UnsignedInteger(3))},
                inputset="random",
                compilation_configuration=compilation_configuration,
            )
        except Exception as error:
            expected = (
                "Random inputset generation is an unsafe feature "
                "and should not be used if you don't know what you are doing"
            )
            assert str(error) == expected
            raise


def test_wrong_inputs(default_compilation_configuration):
    """Test compilation with faulty inputs"""

    # x should have been something like EncryptedScalar(UnsignedInteger(3))
    x = [1, 2, 3]
    input_ranges = ((0, 10),)
    inputset = data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges))
    dict_for_inputs = {"x": x}

    with pytest.raises(AssertionError) as excinfo:
        compile_numpy_function(
            lambda x: 2 * x, dict_for_inputs, inputset, default_compilation_configuration
        )

    list_of_possible_basevalue = [
        "ClearTensor",
        "EncryptedTensor",
        "ClearScalar",
        "EncryptedScalar",
    ]
    assert (
        str(excinfo.value) == f"wrong type for inputs {dict_for_inputs}, "
        f"needs to be one of {list_of_possible_basevalue}"
    )


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(lambda x: (x + (-27)) + 32, ((0, 10),), ["x"]),
        pytest.param(lambda x: ((-3) * x) + (100 - (x + 1)), ((0, 10),), ["x"]),
        pytest.param(
            lambda x, y: (-1) * x + (-2) * y + 40,
            (
                (0, 10),
                (0, 10),
            ),
            ["x", "y"],
        ),
    ],
)
def test_compile_and_run_correctness_with_negative_values(
    function, input_ranges, list_of_arg_names, default_compilation_configuration
):
    """Test correctness of results when running a compiled function, which has some negative
    intermediate values."""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    assert compiler_engine.run(*args) == function(*args)


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names",
    [
        pytest.param(
            lambda x: (20 + 10 * numpy.tanh(50 * (numpy.cos(x + 33.0)))).astype(numpy.uint32),
            ((0, 31),),
            ["x"],
        ),
        pytest.param(
            lambda x: (20 * (numpy.cos(x + 33.0)) + 30).astype(numpy.uint32),
            ((0, 31),),
            ["x"],
        ),
    ],
)
def test_compile_and_run_correctness_with_negative_values_and_pbs(
    function,
    input_ranges,
    list_of_arg_names,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test correctness of results when running a compiled function, which has some negative
    intermediate values."""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    check_is_good_execution(compiler_engine, function, args, verbose=False)


def check_equality_modulo(a, b, modulus):
    """Check that (a mod modulus) == (b mod modulus)"""
    return (a % modulus) == (b % modulus)


@pytest.mark.parametrize(
    "function,input_ranges,list_of_arg_names,modulus",
    [
        pytest.param(lambda x: x + (-20), ((0, 10),), ["x"], 128),
        pytest.param(lambda x: 10 + x * (-3), ((0, 20),), ["x"], 128),
    ],
)
def test_compile_and_run_correctness_with_negative_results(
    function, input_ranges, list_of_arg_names, modulus, default_compilation_configuration
):
    """Test correctness of computations when the result is possibly negative: until #845 is fixed,
    results are currently only correct modulo a power of 2 (given by `modulus` parameter). Eg,
    instead of returning -3, the execution may return -3 mod 128 = 125."""

    function_parameters = {
        arg_name: EncryptedScalar(Integer(64, False)) for arg_name in list_of_arg_names
    }

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        default_compilation_configuration,
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]
    assert check_equality_modulo(compiler_engine.run(*args), function(*args), modulus)


@pytest.mark.parametrize(
    "compilation_configuration",
    [
        CompilationConfiguration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=False,
            use_insecure_key_cache=False,
        ),
        CompilationConfiguration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=True,
            use_insecure_key_cache=False,
        ),
        CompilationConfiguration(
            dump_artifacts_on_unexpected_failures=False,
            enable_unsafe_features=False,
            use_insecure_key_cache=True,
        ),
    ],
)
def test_compile_improper_use_of_insecure_key_cache(
    default_keyring_path, compilation_configuration
):
    """Test the case where the key cache is used with wrong compilation configuration.

    DO NOT USE INSECURE KEY CACHE FOR NORMAL PRODUCTION WORK

    This is a test to check we properly fail for users trying to incorrectly use the insecure key
    cache (to reuse keys across compilations). This allows to speed up tests A LOT but should not be
    used in normal prod environment /!\\ DANGER /!\\."""

    def f(x):
        return x + 42

    if compile_._COMPILE_FHE_INSECURE_KEY_CACHE_DIR is None:  # pylint: disable=protected-access
        compile_._COMPILE_FHE_INSECURE_KEY_CACHE_DIR = str(  # pylint: disable=protected-access
            default_keyring_path
        )

    with pytest.raises(
        RuntimeError,
        match="Unable to use insecure key cache .* "
        "as use_insecure_key_cache or enable_unsafe_features are not set to True in"
        "compilation_configuration",
    ):
        _ = compile_numpy_function(
            f,
            {"x": EncryptedScalar(Integer(64, False))},
            range(10),
            compilation_configuration,
        )
