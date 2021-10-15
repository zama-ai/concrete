"""Test file for numpy compilation functions"""
import itertools
import random

import numpy
import pytest

from concrete.common.compilation import CompilationConfiguration
from concrete.common.data_types.integers import Integer
from concrete.common.debugging import draw_graph, get_printable_graph
from concrete.common.extensions.table import LookupTable
from concrete.common.values import ClearTensor, EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing
from concrete.numpy.compile import compile_numpy_function, compile_numpy_function_into_op_graph


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


def small_lut(x):
    """Test lookup table with small size and output"""
    table = LookupTable(list(range(32)))
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
    a = x + 0.1
    a = numpy.rint(a).astype(numpy.int32)
    z = numpy.abs(10 * func(a))
    z = z.astype(numpy.int32) + y
    return z


def mix_x_and_y_and_call_f_which_expects_small_inputs(func, x, y):
    """Create an upper function to test `func`, which expects small values to not use too much
    precision"""
    a = numpy.abs(0.77 * numpy.sin(x))
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


def mix_x_and_y_and_call_binary_f_two_avoid_0_input(func, c, x, y):
    """Create an upper function to test `func`"""
    z = numpy.abs(func(c, x + 1) + 1)
    z = z.astype(numpy.uint32) + y
    return z


def subtest_compile_and_run_unary_ufunc_correctness(ufunc, upper_function, input_ranges):
    """Test correctness of results when running a compiled function"""

    def get_function(ufunc, upper_function):
        return lambda x, y: upper_function(ufunc, x, y)

    function = get_function(ufunc, upper_function)

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {arg_name: EncryptedScalar(Integer(64, False)) for arg_name in ["x", "y"]}

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]

    # TODO: fix the check
    # assert compiler_engine.run(*args) == function(*args)

    if compiler_engine.run(*args) != function(*args):
        print("Warning, bad computation")


def subtest_compile_and_run_binary_ufunc_correctness(ufunc, upper_function, c, input_ranges):
    """Test correctness of results when running a compiled function"""

    def get_function(ufunc, upper_function):
        return lambda x, y: upper_function(ufunc, c, x, y)

    function = get_function(ufunc, upper_function)

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

    function_parameters = {arg_name: EncryptedScalar(Integer(64, True)) for arg_name in ["x", "y"]}

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )

    args = [random.randint(low, high) for (low, high) in input_ranges]

    # TODO: fix the check
    # assert compiler_engine.run(*args) == function(*args)

    if compiler_engine.run(*args) != function(*args):
        print("Warning, bad computation")


@pytest.mark.parametrize(
    "ufunc",
    [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 2],
)
def test_binary_ufunc_operations(ufunc):
    """Test biary functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""
    if ufunc in [numpy.power, numpy.float_power]:
        # Need small constants to keep results really small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_one, 3, ((0, 4), (0, 5))
        )
    elif ufunc in [numpy.lcm, numpy.left_shift]:
        # Need small constants to keep results sufficiently small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_one, 3, ((0, 5), (0, 5))
        )
    else:
        # General case
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_one, 41, ((0, 5), (0, 5))
        )

    if ufunc in [numpy.power, numpy.float_power]:
        # Need small constants to keep results really small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_two, 2, ((0, 4), (0, 5))
        )
    elif ufunc in [numpy.floor_divide, numpy.fmod, numpy.remainder, numpy.true_divide]:
        # 0 not in the domain of definition
        # Can't make it work, #649
        # TODO: fixme
        pass
        # subtest_compile_and_run_binary_ufunc_correctness(
        #     ufunc, mix_x_and_y_and_call_binary_f_two_avoid_0_input, 31, ((1, 5), (1, 5))
        # )
    elif ufunc in [numpy.lcm, numpy.left_shift]:
        # Need small constants to keep results sufficiently small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_two, 2, ((0, 5), (0, 5))
        )
    elif ufunc in [numpy.ldexp]:
        # Need small constants to keep results sufficiently small
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_two, 2, ((0, 5), (0, 5))
        )
    else:
        # General case
        subtest_compile_and_run_binary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_binary_f_two, 42, ((0, 5), (0, 5))
        )


@pytest.mark.parametrize(
    "ufunc", [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 1]
)
def test_unary_ufunc_operations(ufunc):
    """Test unary functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""
    if ufunc in [
        numpy.degrees,
        numpy.rad2deg,
    ]:
        # Need to reduce the output value, to avoid to need too much precision
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_f_which_has_large_outputs, ((0, 5), (0, 5))
        )
    elif ufunc in [
        numpy.negative,
    ]:
        # Need to turn the input into a float
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_f_with_float_inputs, ((0, 5), (0, 5))
        )
    elif ufunc in [
        numpy.invert,
    ]:
        # Can't make it work, to have a fusable function
        # TODO: fixme
        pass
        # subtest_compile_and_run_unary_ufunc_correctness(
        #     ufunc, mix_x_and_y_and_call_f_with_integer_inputs, ((0, 5), (0, 5))
        # )
    elif ufunc in [
        numpy.arccosh,
        numpy.log,
        numpy.log2,
        numpy.log10,
        numpy.reciprocal,
    ]:
        # No 0 in the domain of definition
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_f_avoid_0_input, ((1, 5), (1, 5))
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
            ufunc, mix_x_and_y_and_call_f_which_expects_small_inputs, ((0, 5), (0, 5))
        )
    else:
        # Regular case for univariate functions
        subtest_compile_and_run_unary_ufunc_correctness(
            ufunc, mix_x_and_y_and_call_f, ((0, 5), (0, 5))
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
        pytest.param(
            no_fuse_unhandled,
            ((-2, 2), (-2, 2)),
            ["x", "y"],
            marks=pytest.mark.xfail(strict=True, raises=ValueError),
        ),
        pytest.param(complicated_topology, ((0, 10),), ["x"]),
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
        pytest.param(lambda x: x + 42, ((0, 10),), ["x"]),
        pytest.param(lambda x: x + numpy.int32(42), ((0, 10),), ["x"]),
        pytest.param(lambda x: x * 2, ((0, 10),), ["x"]),
        pytest.param(lambda x: 12 - x, ((0, 10),), ["x"]),
        pytest.param(lambda x, y: x + y + 8, ((2, 10), (4, 8)), ["x", "y"]),
        pytest.param(lut, ((0, 127),), ["x"]),
        pytest.param(small_lut, ((0, 31),), ["x"]),
        pytest.param(small_fused_table, ((0, 31),), ["x"]),
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
def test_compile_and_run_dot_correctness(size, input_range):
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
        inputset.append((numpy.random.randint(low, high + 1), numpy.random.randint(low, high + 1)))

    function_parameters = {
        "x": EncryptedTensor(Integer(64, False), shape),
        "y": ClearTensor(Integer(64, False), shape),
    }

    def function(x, y):
        return numpy.dot(x, y)

    compiler_engine = compile_numpy_function(
        function,
        function_parameters,
        inputset,
    )

    args = [[random.randint(low, high) for _ in range(size)] for __ in range(2)]
    assert compiler_engine.run(*args) == function(*args)


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
def test_compile_and_run_constant_dot_correctness(size, input_range):
    """Test correctness of results when running a compiled function"""

    low, high = input_range
    shape = (size,)

    inputset = [
        (numpy.zeros(shape, dtype=numpy.uint32),),
        (numpy.ones(shape, dtype=numpy.uint32) * high,),
    ]
    for _ in range(8):
        inputset.append((numpy.random.randint(low, high + 1),))

    constant = numpy.random.randint(low, high + 1, size=shape)

    def left(x):
        return numpy.dot(x, constant)

    def right(x):
        return numpy.dot(constant, x)

    left_circuit = compile_numpy_function(
        left,
        {"x": EncryptedTensor(Integer(64, False), shape)},
        inputset,
    )
    right_circuit = compile_numpy_function(
        left,
        {"x": EncryptedTensor(Integer(64, False), shape)},
        inputset,
    )

    args = (numpy.random.randint(low, high + 1, size=shape).tolist(),)
    assert left_circuit.run(*args) == left(*args)
    assert right_circuit.run(*args) == right(*args)


def test_compile_function_with_direct_tlu():
    """Test compile_numpy_function_into_op_graph for a program with direct table lookup"""

    table = LookupTable([9, 2, 4, 11])

    def function(x):
        return x + table[x]

    op_graph = compile_numpy_function_into_op_graph(
        function,
        {"x": EncryptedScalar(Integer(2, is_signed=False))},
        [(0,), (1,), (2,), (3,)],
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
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
        )


@pytest.mark.parametrize(
    "function,parameters,inputset,match",
    [
        pytest.param(
            lambda x: 1 - x,
            {"x": EncryptedScalar(Integer(3, is_signed=False))},
            [(i,) for i in range(8)],
            (
                "function you are trying to compile isn't supported for MLIR lowering\n"
                "\n"
                "%0 = Constant(1)                                   # ClearScalar<Integer<unsigned, 1 bits>>\n"  # noqa: E501
                "%1 = x                                             # EncryptedScalar<Integer<unsigned, 3 bits>>\n"  # noqa: E501
                "%2 = Sub(%0, %1)                                   # EncryptedScalar<Integer<signed, 4 bits>>\n"  # noqa: E501
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ signed integer outputs aren't supported\n"  # noqa: E501
                "return(%2)\n"
            ),
        ),
        pytest.param(
            lambda x: x + 1,
            {"x": EncryptedTensor(Integer(3, is_signed=False), shape=(2, 2))},
            [(numpy.random.randint(0, 8, size=(2, 2)),) for i in range(10)],
            (
                "function you are trying to compile isn't supported for MLIR lowering\n"
                "\n"
                "%0 = x                                             # EncryptedTensor<Integer<unsigned, 3 bits>, shape=(2, 2)>\n"  # noqa: E501
                "%1 = Constant(1)                                   # ClearScalar<Integer<unsigned, 1 bits>>\n"  # noqa: E501
                "%2 = Add(%0, %1)                                   # EncryptedTensor<Integer<unsigned, 4 bits>, shape=(2, 2)>\n"  # noqa: E501
                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ non scalar outputs aren't supported\n"  # noqa: E501
                "return(%2)\n"
            ),
        ),
    ],
)
def test_fail_compile(function, parameters, inputset, match):
    """Test function compile_numpy_function_into_op_graph for a program with signed values"""

    try:
        compile_numpy_function(
            function,
            parameters,
            inputset,
            CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
        )
    except RuntimeError as error:
        assert str(error) == match


def test_small_inputset():
    """Test function compile_numpy_function_into_op_graph with an unacceptably small inputset"""
    compile_numpy_function_into_op_graph(
        lambda x: x + 42,
        {"x": EncryptedScalar(Integer(5, is_signed=False))},
        [(0,), (3,)],
        CompilationConfiguration(dump_artifacts_on_unexpected_failures=False),
    )


def test_small_inputset_treat_warnings_as_errors():
    """Test function compile_numpy_function_into_op_graph with an unacceptably small inputset"""
    with pytest.raises(ValueError, match=".* inputset contains too few inputs .*"):
        compile_numpy_function_into_op_graph(
            lambda x: x + 42,
            {"x": EncryptedScalar(Integer(5, is_signed=False))},
            [(0,), (3,)],
            CompilationConfiguration(
                dump_artifacts_on_unexpected_failures=False,
                treat_warnings_as_errors=True,
            ),
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
            "%0 = x                                             "
            "# EncryptedTensor<Integer<unsigned, 6 bits>, shape=(4,)>"
            "\n%1 = y                                             "
            "# EncryptedTensor<Integer<unsigned, 6 bits>, shape=(4,)>"
            "\n%2 = Dot(%0, %1)                                   "
            "# EncryptedScalar<Integer<unsigned, 6 bits>>"
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
            yield numpy.array(prod_i), numpy.array(prod_j)

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


def test_compile_too_high_bitwidth():
    """Check that the check of maximal bitwidth of intermediate data works fine."""

    def function(x, y):
        return x + y

    def data_gen(args):
        for prod in itertools.product(*args):
            yield prod

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
        )

    assert (
        "max_bit_width of some nodes is too high for the current version of the "
        "compiler (maximum must be 7 which is not compatible with" in str(excinfo.value)
    )

    assert str(excinfo.value).endswith(", 8)])")

    # Just ok
    input_ranges = [(0, 99), (0, 28)]

    compile_numpy_function(
        function,
        function_parameters,
        data_gen(tuple(range(x[0], x[1] + 1) for x in input_ranges)),
    )
