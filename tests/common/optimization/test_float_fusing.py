"""Test file for float subgraph fusing"""

import random
from inspect import signature

import numpy
import pytest

from concrete.common.data_types.integers import Integer
from concrete.common.debugging.custom_assert import assert_not_reached
from concrete.common.optimization.topological import fuse_float_operations
from concrete.common.values import EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing
from concrete.numpy.tracing import trace_numpy_function


def no_fuse(x):
    """No fuse"""
    return x + 2


def no_fuse_unhandled(x, y):
    """No fuse unhandled"""
    x_1 = x + 0.7
    y_1 = y + 1.3
    intermediate = x_1 + y_1
    return intermediate.astype(numpy.int32)


def no_fuse_dot(x):
    """No fuse dot"""
    return numpy.dot(x, numpy.full((10,), 1.33, dtype=numpy.float64)).astype(numpy.int32)


def simple_fuse_not_output(x):
    """Simple fuse not output"""
    intermediate = x.astype(numpy.float64)
    intermediate = intermediate.astype(numpy.int32)
    return intermediate + 2


def simple_fuse_output(x):
    """Simple fuse output"""
    return x.astype(numpy.float64).astype(numpy.int32)


def mix_x_and_y_intricately_and_call_f(function, x, y):
    """Mix x and y in an intricated way, that can't be simplified by
    an optimizer eg, and then call function
    """
    intermediate = x + y
    intermediate = intermediate + 2
    intermediate = intermediate.astype(numpy.float32)
    intermediate = intermediate.astype(numpy.int32)
    x_p_1 = intermediate + 1.5
    x_p_2 = intermediate + 2.7
    x_p_3 = function(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def mix_x_and_y_and_call_f(function, x, y):
    """Mix x and y and then call function"""
    x_p_1 = x + 0.1
    x_p_2 = x + 0.2
    x_p_3 = function(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def mix_x_and_y_into_range_0_to_1_and_call_f(function, x, y):
    """Mix x and y and then call function, in such a way that the input to function is between
    0 and 1"""
    x_p_1 = x + 0.1
    x_p_2 = x + 0.2
    x_p_4 = 1 - numpy.abs(numpy.sin(x_p_1 + x_p_2 + 0.3))
    x_p_3 = function(x_p_4)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def mix_x_and_y_into_integer_and_call_f(function, x, y):
    """Mix x and y but keep the entry to function as an integer"""
    x_p_1 = x + 1
    x_p_2 = x + 2
    x_p_3 = function(x_p_1 + x_p_2)
    return (
        x_p_3.astype(numpy.int32),
        x_p_2.astype(numpy.int32),
        (x_p_2 + 3).astype(numpy.int32),
        x_p_3.astype(numpy.int32) + 67,
        y,
        (y + 4.7).astype(numpy.int32) + 3,
    )


def get_func_params_scalar_int32(func):
    """Returns a dict with parameters as scalar int32"""

    return {
        param_name: EncryptedScalar(Integer(32, True))
        for param_name in signature(func).parameters.keys()
    }


@pytest.mark.parametrize(
    "function_to_trace,fused,params,warning_message",
    [
        pytest.param(no_fuse, False, get_func_params_scalar_int32(no_fuse), "", id="no_fuse"),
        pytest.param(
            no_fuse_unhandled,
            False,
            get_func_params_scalar_int32(no_fuse_unhandled),
            """The following subgraph is not fusable:
%0 = x                                             # EncryptedScalar<Integer<signed, 32 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ one of 2 variable inputs (can only have 1 for fusing)
%1 = Constant(0.7)                                 # ClearScalar<Float<64 bits>>
%2 = y                                             # EncryptedScalar<Integer<signed, 32 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ one of 2 variable inputs (can only have 1 for fusing)
%3 = Constant(1.3)                                 # ClearScalar<Float<64 bits>>
%4 = Add(%0, %1)                                   # EncryptedScalar<Float<64 bits>>
%5 = Add(%2, %3)                                   # EncryptedScalar<Float<64 bits>>
%6 = Add(%4, %5)                                   # EncryptedScalar<Float<64 bits>>
%7 = astype(int32)(%6)                             # EncryptedScalar<Integer<signed, 32 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ cannot fuse here as the subgraph has 2 variable inputs
return(%7)""",  # noqa: E501 # pylint: disable=line-too-long
            id="no_fuse_unhandled",
        ),
        pytest.param(
            no_fuse_dot,
            False,
            {"x": EncryptedTensor(Integer(32, True), (10,))},
            """The following subgraph is not fusable:
%0 = x                                             # EncryptedTensor<Integer<signed, 32 bits>, shape=(10,)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ input node with shape (10,)
%1 = Constant([1.33 1.33 ... 1.33 1.33])           # ClearTensor<Float<64 bits>, shape=(10,)>
%2 = Dot(%0, %1)                                   # EncryptedScalar<Float<64 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ output shapes: #0, () are not the same as the subgraph's input: (10,)
%3 = astype(int32)(%2)                             # EncryptedScalar<Integer<signed, 32 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ output shapes: #0, () are not the same as the subgraph's input: (10,)
return(%3)""",  # noqa: E501 # pylint: disable=line-too-long
            id="no_fuse_dot",
        ),
        pytest.param(
            simple_fuse_not_output,
            True,
            get_func_params_scalar_int32(simple_fuse_not_output),
            None,
            id="simple_fuse_not_output",
        ),
        pytest.param(
            simple_fuse_output,
            True,
            get_func_params_scalar_int32(simple_fuse_output),
            None,
            id="simple_fuse_output",
        ),
        pytest.param(
            lambda x, y: mix_x_and_y_intricately_and_call_f(numpy.rint, x, y),
            True,
            get_func_params_scalar_int32(lambda x, y: None),
            None,
            id="mix_x_and_y_intricately_and_call_f_with_rint",
        ),
        pytest.param(
            lambda x, y: mix_x_and_y_and_call_f(numpy.rint, x, y),
            True,
            get_func_params_scalar_int32(lambda x, y: None),
            None,
            id="mix_x_and_y_and_call_f_with_rint",
        ),
    ],
)
def test_fuse_float_operations(
    function_to_trace,
    fused,
    params,
    warning_message,
    capfd,
    remove_color_codes,
):
    """Test function for fuse_float_operations"""

    op_graph = trace_numpy_function(
        function_to_trace,
        params,
    )
    orig_num_nodes = len(op_graph.graph)
    fuse_float_operations(op_graph)
    fused_num_nodes = len(op_graph.graph)

    if fused:
        assert fused_num_nodes < orig_num_nodes
    else:
        assert fused_num_nodes == orig_num_nodes
        captured = capfd.readouterr()
        assert warning_message in remove_color_codes(captured.err)

    for input_ in [0, 2, 42, 44]:
        inputs = ()
        for param_input_value in params.values():
            if param_input_value.is_scalar:
                input_ = numpy.int32(input_)
            else:
                input_ = numpy.full(param_input_value.shape, input_, dtype=numpy.int32)
            inputs += (input_,)

        assert numpy.array_equal(function_to_trace(*inputs), op_graph(*inputs))


def subtest_tensor_no_fuse(fun, tensor_shape):
    """Test case to verify float fusing is only applied on functions on scalars."""

    if tensor_shape == ():
        # We want tensors
        return

    if fun in LIST_OF_UFUNC_WHICH_HAVE_INTEGER_ONLY_SOURCES:
        # We need at least one input of the bivariate function to be float
        return

    # Float fusing currently cannot work if the constant in a bivariate operator is bigger than the
    # variable input.
    # Make a broadcastable shape but with the constant being bigger
    variable_tensor_shape = (1,) + tensor_shape
    constant_bigger_shape = (random.randint(2, 10),) + tensor_shape

    def tensor_no_fuse(x):
        intermediate = x.astype(numpy.float64)
        intermediate = fun(intermediate, numpy.ones(constant_bigger_shape))
        return intermediate.astype(numpy.int32)

    function_to_trace = tensor_no_fuse
    params_names = signature(function_to_trace).parameters.keys()

    op_graph = trace_numpy_function(
        function_to_trace,
        {
            param_name: EncryptedTensor(Integer(32, True), shape=variable_tensor_shape)
            for param_name in params_names
        },
    )
    orig_num_nodes = len(op_graph.graph)
    fuse_float_operations(op_graph)
    fused_num_nodes = len(op_graph.graph)

    assert orig_num_nodes == fused_num_nodes


def check_results_are_equal(function_result, op_graph_result):
    """Check the output of function execution and OPGraph evaluation are equal."""

    if isinstance(function_result, tuple) and isinstance(op_graph_result, tuple):
        assert len(function_result) == len(op_graph_result)
        are_equal = (
            function_output == op_graph_output
            for function_output, op_graph_output in zip(function_result, op_graph_result)
        )
    elif not isinstance(function_result, tuple) and not isinstance(op_graph_result, tuple):
        are_equal = (function_result == op_graph_result,)
    else:
        assert_not_reached(f"Incompatible outputs: {function_result}, {op_graph_result}")

    return all(value.all() if isinstance(value, numpy.ndarray) else value for value in are_equal)


def subtest_fuse_float_unary_operations_correctness(fun, tensor_shape):
    """Test a unary function with fuse_float_operations."""

    # Some manipulation to avoid issues with domain of definitions of functions
    if fun == numpy.arccosh:
        # 0 is not in the domain of definition
        input_list = [1, 2, 42, 44]
        super_fun_list = [mix_x_and_y_and_call_f]
    elif fun in [numpy.arctanh, numpy.arccos, numpy.arcsin, numpy.arctan]:
        # Needs values between 0 and 1 in the call function
        input_list = [0, 2, 42, 44]
        super_fun_list = [mix_x_and_y_into_range_0_to_1_and_call_f]
    elif fun in [numpy.cosh, numpy.sinh, numpy.exp, numpy.exp2, numpy.expm1]:
        # Not too large values to avoid overflows
        input_list = [1, 2, 5, 11]
        super_fun_list = [mix_x_and_y_and_call_f, mix_x_and_y_intricately_and_call_f]
    else:
        # Regular case
        input_list = [0, 2, 42, 44]
        super_fun_list = [mix_x_and_y_and_call_f, mix_x_and_y_intricately_and_call_f]

    for super_fun in super_fun_list:

        for input_ in input_list:

            def get_function_to_trace():
                return lambda x, y: super_fun(fun, x, y)

            function_to_trace = get_function_to_trace()

            params_names = signature(function_to_trace).parameters.keys()

            op_graph = trace_numpy_function(
                function_to_trace,
                {
                    param_name: EncryptedTensor(Integer(32, True), tensor_shape)
                    for param_name in params_names
                },
            )
            orig_num_nodes = len(op_graph.graph)
            fuse_float_operations(op_graph)
            fused_num_nodes = len(op_graph.graph)

            assert fused_num_nodes < orig_num_nodes

            # Check that the call to the function or to the op_graph evaluation give the same
            # result
            tensor_diversifier = (
                # The following +1 in the range is to avoid to have 0's which is not in the
                # domain definition of some of our functions
                numpy.arange(1, numpy.product(tensor_shape) + 1, dtype=numpy.int32).reshape(
                    tensor_shape
                )
                if tensor_shape != ()
                else 1
            )

            if fun in [numpy.arctanh, numpy.arccos, numpy.arcsin, numpy.arctan]:
                # Domain of definition for these functions
                tensor_diversifier = (
                    numpy.ones(tensor_shape, dtype=numpy.int32) if tensor_shape != () else 1
                )

            input_ = numpy.int32(input_ * tensor_diversifier)

            num_params = len(params_names)
            assert num_params == 2

            # Create inputs which are either of the form [x, x] or [x, y]
            for j in range(4):

                if fun in [numpy.arctanh, numpy.arccos, numpy.arcsin, numpy.arctan] and j > 0:
                    # Domain of definition for these functions
                    break

                input_a = input_
                input_b = input_ + j

                if tensor_shape != ():
                    numpy.random.shuffle(input_a)
                    numpy.random.shuffle(input_b)

                inputs = (input_a, input_b) if random.randint(0, 1) == 0 else (input_b, input_a)

                function_result = function_to_trace(*inputs)
                op_graph_result = op_graph(*inputs)

                assert check_results_are_equal(function_result, op_graph_result)


LIST_OF_UFUNC_WHICH_HAVE_INTEGER_ONLY_SOURCES = {
    numpy.bitwise_and,
    numpy.bitwise_or,
    numpy.bitwise_xor,
    numpy.gcd,
    numpy.lcm,
    numpy.ldexp,
    numpy.left_shift,
    numpy.logical_and,
    numpy.logical_not,
    numpy.logical_or,
    numpy.logical_xor,
    numpy.remainder,
    numpy.right_shift,
}


def subtest_fuse_float_binary_operations_correctness(fun, tensor_shape):
    """Test a binary functions with fuse_float_operations, with a constant as a source."""

    for i in range(4):

        # Know if the function is defined for integer inputs
        if fun in LIST_OF_UFUNC_WHICH_HAVE_INTEGER_ONLY_SOURCES:
            if i not in [0, 2]:
                continue

        # The .astype(numpy.float64) that we have in cases 0 and 2 is here to force
        # a float output even for functions which return an integer (eg, XOR), such
        # that our frontend always try to fuse them

        # The .astype(numpy.float64) that we have in cases 1 and 3 is here to force
        # a float output even for functions which return a bool (eg, EQUAL), such
        # that our frontend always try to fuse them

        # For bivariate functions: fix one of the inputs
        if i == 0:
            # With an integer in first position
            ones_0 = numpy.ones(tensor_shape, dtype=numpy.int32) if tensor_shape != () else 1

            def get_function_to_trace():
                return lambda x, y: fun(3 * ones_0, x + y).astype(numpy.float64).astype(numpy.int32)

        elif i == 1:
            # With a float in first position
            ones_1 = numpy.ones(tensor_shape, dtype=numpy.float64) if tensor_shape != () else 1

            def get_function_to_trace():
                return (
                    lambda x, y: fun(2.3 * ones_1, x + y).astype(numpy.float64).astype(numpy.int32)
                )

        elif i == 2:
            # With an integer in second position
            ones_2 = numpy.ones(tensor_shape, dtype=numpy.int32) if tensor_shape != () else 1

            def get_function_to_trace():
                return lambda x, y: fun(x + y, 4 * ones_2).astype(numpy.float64).astype(numpy.int32)

        else:
            # With a float in second position
            ones_else = numpy.ones(tensor_shape, dtype=numpy.float64) if tensor_shape != () else 1

            def get_function_to_trace():
                return (
                    lambda x, y: fun(x + y, 5.7 * ones_else)
                    .astype(numpy.float64)
                    .astype(numpy.int32)
                )

        input_list = [0, 2, 42, 44]

        # Domain of definition
        if fun in [numpy.true_divide, numpy.remainder, numpy.floor_divide, numpy.fmod]:
            input_list = [2, 42, 44]

        for input_ in input_list:
            function_to_trace = get_function_to_trace()

            params_names = signature(function_to_trace).parameters.keys()

            op_graph = trace_numpy_function(
                function_to_trace,
                {
                    param_name: EncryptedTensor(Integer(32, True), tensor_shape)
                    for param_name in params_names
                },
            )
            orig_num_nodes = len(op_graph.graph)
            fuse_float_operations(op_graph)
            fused_num_nodes = len(op_graph.graph)

            assert fused_num_nodes < orig_num_nodes

            # Check that the call to the function or to the op_graph evaluation give the same
            # result
            tensor_diversifier = (
                # The following +1 in the range is to avoid to have 0's which is not in the
                # domain definition of some of our functions
                numpy.arange(1, numpy.product(tensor_shape) + 1, dtype=numpy.int32).reshape(
                    tensor_shape
                )
                if tensor_shape != ()
                else 1
            )
            input_ = input_ * tensor_diversifier

            num_params = len(params_names)
            assert num_params == 2

            # Create inputs which are either of the form [x, x] or [x, y]
            for j in range(4):
                inputs = (input_, input_ + j)

                function_result = function_to_trace(*inputs)
                op_graph_result = op_graph(*inputs)

                assert check_results_are_equal(function_result, op_graph_result)


def subtest_fuse_float_binary_operations_dont_support_two_variables(fun, tensor_shape):
    """Test a binary function with fuse_float_operations, with no constant as
    a source."""

    def get_function_to_trace():
        return lambda x, y: fun(x, y).astype(numpy.int32)

    function_to_trace = get_function_to_trace()

    params_names = signature(function_to_trace).parameters.keys()

    with pytest.raises(NotImplementedError, match=r"Can't manage binary operator"):
        trace_numpy_function(
            function_to_trace,
            {
                param_name: EncryptedTensor(Integer(32, True), tensor_shape)
                for param_name in params_names
            },
        )


@pytest.mark.parametrize("fun", tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC)
@pytest.mark.parametrize(
    "tensor_shape", [pytest.param((), id="scalar"), pytest.param((3, 1, 2), id="tensor")]
)
def test_ufunc_operations(fun, tensor_shape):
    """Test functions which are in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC."""

    if fun.nin == 1:
        subtest_fuse_float_unary_operations_correctness(fun, tensor_shape)
    elif fun.nin == 2:
        subtest_fuse_float_binary_operations_correctness(fun, tensor_shape)
        subtest_fuse_float_binary_operations_dont_support_two_variables(fun, tensor_shape)
        subtest_tensor_no_fuse(fun, tensor_shape)
    else:
        raise NotImplementedError("Only unary and binary functions are tested for now")
