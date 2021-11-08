"""Test file for numpy tracing"""

from copy import deepcopy

import networkx as nx
import numpy
import pytest

from concrete.common.data_types.dtypes_helpers import broadcast_shapes
from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.debugging import format_operation_graph
from concrete.common.representation import intermediate as ir
from concrete.common.values import ClearScalar, ClearTensor, EncryptedScalar, EncryptedTensor
from concrete.numpy import tracing

OPERATIONS_TO_TEST = [ir.Add, ir.Sub, ir.Mul]

# Functions from tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC, whose output
# is a float64, whatever the input type
LIST_OF_UFUNC_WHOSE_OUTPUT_IS_FLOAT64 = set(
    [
        numpy.arccos,
        numpy.arccosh,
        numpy.arcsin,
        numpy.arcsinh,
        numpy.arctan,
        numpy.arctanh,
        numpy.cbrt,
        numpy.ceil,
        numpy.cos,
        numpy.cosh,
        numpy.deg2rad,
        numpy.degrees,
        numpy.exp,
        numpy.exp2,
        numpy.expm1,
        numpy.fabs,
        numpy.floor,
        numpy.log,
        numpy.log10,
        numpy.log1p,
        numpy.log2,
        numpy.rad2deg,
        numpy.radians,
        numpy.rint,
        numpy.sin,
        numpy.sinh,
        numpy.spacing,
        numpy.sqrt,
        numpy.tan,
        numpy.tanh,
        numpy.trunc,
    ]
)

# Functions from tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC, whose output
# is a boolean, whatever the input type
LIST_OF_UFUNC_WHOSE_OUTPUT_IS_BOOL = set(
    [
        numpy.isfinite,
        numpy.isinf,
        numpy.isnan,
        numpy.signbit,
        numpy.logical_not,
    ]
)


@pytest.mark.parametrize(
    "operation",
    OPERATIONS_TO_TEST,
)
@pytest.mark.parametrize(
    "x",
    [
        pytest.param(EncryptedScalar(Integer(64, is_signed=False)), id="x: Encrypted uint"),
        pytest.param(
            EncryptedScalar(Integer(64, is_signed=True)),
            id="x: Encrypted int",
        ),
        pytest.param(
            ClearScalar(Integer(64, is_signed=False)),
            id="x: Clear uint",
        ),
        pytest.param(
            ClearScalar(Integer(64, is_signed=True)),
            id="x: Clear int",
        ),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        pytest.param(EncryptedScalar(Integer(64, is_signed=False)), id="y: Encrypted uint"),
        pytest.param(
            EncryptedScalar(Integer(64, is_signed=True)),
            id="y: Encrypted int",
        ),
        pytest.param(
            ClearScalar(Integer(64, is_signed=False)),
            id="y: Clear uint",
        ),
        pytest.param(
            ClearScalar(Integer(64, is_signed=True)),
            id="y: Clear int",
        ),
    ],
)
def test_numpy_tracing_binary_op(operation, x, y, test_helpers):
    "Test numpy tracing a binary operation (in the supported ops)"

    # Remark that the functions here have a common structure (which is
    # 2x op y), such that creating further the ref_graph is easy, by
    # hand
    def simple_add_function(x, y):
        z = x + x
        return z + y

    def simple_sub_function(x, y):
        z = x + x
        return z - y

    def simple_mul_function(x, y):
        z = x + x
        return z * y

    assert operation in OPERATIONS_TO_TEST, f"unknown operation {operation}"
    if operation == ir.Add:
        function_to_compile = simple_add_function
    elif operation == ir.Sub:
        function_to_compile = simple_sub_function
    elif operation == ir.Mul:
        function_to_compile = simple_mul_function

    op_graph = tracing.trace_numpy_function(function_to_compile, {"x": x, "y": y})

    ref_graph = nx.MultiDiGraph()

    input_x = ir.Input(x, input_name="x", program_input_idx=0)
    input_y = ir.Input(y, input_name="y", program_input_idx=1)

    add_node_z = ir.Add(
        (
            input_x.outputs[0],
            input_x.outputs[0],
        )
    )

    returned_final_node = operation(
        (
            add_node_z.outputs[0],
            input_y.outputs[0],
        )
    )

    ref_graph.add_node(input_x)
    ref_graph.add_node(input_y)
    ref_graph.add_node(add_node_z)
    ref_graph.add_node(returned_final_node)

    ref_graph.add_edge(input_x, add_node_z, input_idx=0, output_idx=0)
    ref_graph.add_edge(input_x, add_node_z, input_idx=1, output_idx=0)

    ref_graph.add_edge(add_node_z, returned_final_node, input_idx=0, output_idx=0)
    ref_graph.add_edge(input_y, returned_final_node, input_idx=1, output_idx=0)

    assert test_helpers.digraphs_are_equivalent(ref_graph, op_graph.graph)


def test_numpy_tracing_tensors():
    "Test numpy tracing tensors"

    def all_operations(x):
        intermediate = x + numpy.array([[1, 2], [3, 4]])
        intermediate = numpy.array([[5, 6], [7, 8]]) + intermediate

        intermediate = numpy.array([[100, 200], [300, 400]]) - intermediate
        intermediate = intermediate - numpy.array([[10, 20], [30, 40]])

        intermediate = intermediate * numpy.array([[1, 2], [2, 1]])
        intermediate = numpy.array([[2, 1], [1, 2]]) * intermediate

        return intermediate

    op_graph = tracing.trace_numpy_function(
        all_operations, {"x": EncryptedTensor(Integer(32, True), shape=(2, 2))}
    )

    expected = """
%0 = Constant([[2 1] [1 2]])                       # ClearTensor<Integer<unsigned, 2 bits>, shape=(2, 2)>
%1 = Constant([[1 2] [2 1]])                       # ClearTensor<Integer<unsigned, 2 bits>, shape=(2, 2)>
%2 = Constant([[10 20] [30 40]])                   # ClearTensor<Integer<unsigned, 6 bits>, shape=(2, 2)>
%3 = Constant([[100 200] [300 400]])               # ClearTensor<Integer<unsigned, 9 bits>, shape=(2, 2)>
%4 = Constant([[5 6] [7 8]])                       # ClearTensor<Integer<unsigned, 4 bits>, shape=(2, 2)>
%5 = x                                             # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%6 = Constant([[1 2] [3 4]])                       # ClearTensor<Integer<unsigned, 3 bits>, shape=(2, 2)>
%7 = Add(%5, %6)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%8 = Add(%4, %7)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%9 = Sub(%3, %8)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%10 = Sub(%9, %2)                                  # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%11 = Mul(%10, %1)                                 # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%12 = Mul(%0, %11)                                 # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
return(%12)
""".lstrip()  # noqa: E501

    assert format_operation_graph(op_graph, show_data_types=True) == expected


def test_numpy_explicit_tracing_tensors():
    "Test numpy tracing tensors using explicit operations"

    def all_explicit_operations(x):
        intermediate = numpy.add(x, numpy.array([[1, 2], [3, 4]]))
        intermediate = numpy.add(numpy.array([[5, 6], [7, 8]]), intermediate)

        intermediate = numpy.subtract(numpy.array([[100, 200], [300, 400]]), intermediate)
        intermediate = numpy.subtract(intermediate, numpy.array([[10, 20], [30, 40]]))

        intermediate = numpy.multiply(intermediate, numpy.array([[1, 2], [2, 1]]))
        intermediate = numpy.multiply(numpy.array([[2, 1], [1, 2]]), intermediate)

        return intermediate

    op_graph = tracing.trace_numpy_function(
        all_explicit_operations, {"x": EncryptedTensor(Integer(32, True), shape=(2, 2))}
    )

    expected = """
%0 = Constant([[2 1] [1 2]])                       # ClearTensor<Integer<unsigned, 2 bits>, shape=(2, 2)>
%1 = Constant([[1 2] [2 1]])                       # ClearTensor<Integer<unsigned, 2 bits>, shape=(2, 2)>
%2 = Constant([[10 20] [30 40]])                   # ClearTensor<Integer<unsigned, 6 bits>, shape=(2, 2)>
%3 = Constant([[100 200] [300 400]])               # ClearTensor<Integer<unsigned, 9 bits>, shape=(2, 2)>
%4 = Constant([[5 6] [7 8]])                       # ClearTensor<Integer<unsigned, 4 bits>, shape=(2, 2)>
%5 = x                                             # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%6 = Constant([[1 2] [3 4]])                       # ClearTensor<Integer<unsigned, 3 bits>, shape=(2, 2)>
%7 = Add(%5, %6)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%8 = Add(%4, %7)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%9 = Sub(%3, %8)                                   # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%10 = Sub(%9, %2)                                  # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%11 = Mul(%10, %1)                                 # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
%12 = Mul(%0, %11)                                 # EncryptedTensor<Integer<signed, 32 bits>, shape=(2, 2)>
return(%12)
""".lstrip()  # noqa: E501

    assert format_operation_graph(op_graph, show_data_types=True) == expected


@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        pytest.param((), ()),
        pytest.param((3,), ()),
        pytest.param((3,), (1,)),
        pytest.param((3,), (2,), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((3,), (3,)),
        pytest.param((2, 3), ()),
        pytest.param((2, 3), (1,)),
        pytest.param((2, 3), (2,), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 3), (3,)),
        pytest.param((2, 3), (1, 1)),
        pytest.param((2, 3), (2, 1)),
        pytest.param((2, 3), (3, 1), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 3), (1, 2), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 3), (2, 2), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 3), (3, 2), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 3), (1, 3)),
        pytest.param((2, 3), (2, 3)),
        pytest.param((2, 3), (3, 3), marks=pytest.mark.xfail(raises=AssertionError, strict=True)),
        pytest.param((2, 1, 3), (1, 1, 1)),
        pytest.param((2, 1, 3), (1, 4, 1)),
        pytest.param((2, 1, 3), (2, 4, 3)),
    ],
)
def test_numpy_tracing_broadcasted_tensors(x_shape, y_shape):
    """Test numpy tracing broadcasted tensors"""

    def f(x, y):
        return x + y

    op_graph = tracing.trace_numpy_function(
        f,
        {
            "x": EncryptedTensor(Integer(3, True), shape=x_shape),
            "y": EncryptedTensor(Integer(3, True), shape=y_shape),
        },
    )

    assert op_graph.input_nodes[0].outputs[0].shape == x_shape
    assert op_graph.input_nodes[1].outputs[0].shape == y_shape
    assert op_graph.output_nodes[0].outputs[0].shape == broadcast_shapes(x_shape, y_shape)


@pytest.mark.parametrize(
    "function_to_trace,op_graph_expected_output_type,input_and_expected_output_tuples",
    [
        (
            lambda x: x.astype(numpy.int32),
            Integer(32, is_signed=True),
            [
                (14, numpy.int32(14)),
                (1.5, numpy.int32(1)),
                (2.0, numpy.int32(2)),
                (-1.5, numpy.int32(-1)),
                (2 ** 31 - 1, numpy.int32(2 ** 31 - 1)),
                (-(2 ** 31), numpy.int32(-(2 ** 31))),
            ],
        ),
        (
            lambda x: x.astype(numpy.uint32),
            Integer(32, is_signed=False),
            [
                (14, numpy.uint32(14)),
                (1.5, numpy.uint32(1)),
                (2.0, numpy.uint32(2)),
                (2 ** 32 - 1, numpy.uint32(2 ** 32 - 1)),
            ],
        ),
        (
            lambda x: x.astype(numpy.int64),
            Integer(64, is_signed=True),
            [
                (14, numpy.int64(14)),
                (1.5, numpy.int64(1)),
                (2.0, numpy.int64(2)),
                (-1.5, numpy.int64(-1)),
                (2 ** 63 - 1, numpy.int64(2 ** 63 - 1)),
                (-(2 ** 63), numpy.int64(-(2 ** 63))),
            ],
        ),
        (
            lambda x: x.astype(numpy.uint64),
            Integer(64, is_signed=False),
            [
                (14, numpy.uint64(14)),
                (1.5, numpy.uint64(1)),
                (2.0, numpy.uint64(2)),
                (2 ** 64 - 1, numpy.uint64(2 ** 64 - 1)),
            ],
        ),
        (
            lambda x: x.astype(numpy.float64),
            Float(64),
            [
                (14, numpy.float64(14.0)),
                (1.5, numpy.float64(1.5)),
                (2.0, numpy.float64(2.0)),
                (-1.5, numpy.float64(-1.5)),
            ],
        ),
        (
            lambda x: x.astype(numpy.float32),
            Float(32),
            [
                (14, numpy.float32(14.0)),
                (1.5, numpy.float32(1.5)),
                (2.0, numpy.float32(2.0)),
                (-1.5, numpy.float32(-1.5)),
            ],
        ),
    ],
)
def test_tracing_astype(
    function_to_trace, op_graph_expected_output_type, input_and_expected_output_tuples
):
    """Test function for NPTracer.astype"""
    for input_, expected_output in input_and_expected_output_tuples:
        input_value = (
            EncryptedScalar(Integer(64, is_signed=True))
            if isinstance(input_, int)
            else EncryptedScalar(Float(64))
        )

        op_graph = tracing.trace_numpy_function(function_to_trace, {"x": input_value})
        output_node = op_graph.output_nodes[0]
        assert op_graph_expected_output_type == output_node.outputs[0].dtype

        node_results = op_graph.evaluate({0: numpy.array(input_)})
        evaluated_output = node_results[output_node]
        assert evaluated_output.dtype == expected_output.dtype
        assert expected_output == evaluated_output


def test_tracing_astype_single_element_array_corner_case():
    """Test corner case where an array could be transformed to its scalar element"""
    a = numpy.array([1], dtype=numpy.float64)

    op_graph = tracing.trace_numpy_function(
        lambda x: x.astype(numpy.int32), {"x": EncryptedTensor(Float(64), (1,))}
    )

    eval_result = op_graph(a)
    assert numpy.array_equal(numpy.array([1], dtype=numpy.int32), eval_result)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(
            {"x": EncryptedScalar(Integer(32, is_signed=True))},
        ),
    ],
)
@pytest.mark.parametrize(
    "function_to_trace",
    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint is not happy
    # with it
    [lambda x: numpy.invert(x), lambda x: numpy.bitwise_not(x)],
)
def test_trace_numpy_fails_for_invert(inputs, function_to_trace):
    """Check we catch calls to numpy.invert and tell user to change their code"""

    with pytest.raises(RuntimeError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert (
        "NPTracer does not manage the following func: invert. Please replace by calls to "
        "bitwise_xor with appropriate mask" in str(excinfo.value)
    )


@pytest.mark.parametrize(
    "inputs,expected_output_node",
    [
        pytest.param(
            {"x": EncryptedScalar(Integer(7, is_signed=False))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(32, is_signed=True))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(64, is_signed=True))},
            ir.GenericFunction,
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(128, is_signed=True))},
            ir.GenericFunction,
            marks=pytest.mark.xfail(strict=True, raises=NotImplementedError),
        ),
        pytest.param(
            {"x": EncryptedScalar(Float(64))},
            ir.GenericFunction,
        ),
    ],
)
@pytest.mark.parametrize(
    "function_to_trace_def",
    [f for f in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC if f.nin == 1],
)
def test_trace_numpy_supported_unary_ufuncs(inputs, expected_output_node, function_to_trace_def):
    """Function to trace supported numpy ufuncs"""

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    # pylint: disable=cell-var-from-loop
    function_to_trace = lambda x: function_to_trace_def(x)  # noqa: E731
    # pylint: enable=cell-var-from-loop

    op_graph = tracing.trace_numpy_function(function_to_trace, inputs)

    assert len(op_graph.output_nodes) == 1
    assert isinstance(op_graph.output_nodes[0], expected_output_node)
    assert len(op_graph.output_nodes[0].outputs) == 1

    if function_to_trace_def in LIST_OF_UFUNC_WHOSE_OUTPUT_IS_FLOAT64:
        assert op_graph.output_nodes[0].outputs[0] == EncryptedScalar(Float(64))
    elif function_to_trace_def in LIST_OF_UFUNC_WHOSE_OUTPUT_IS_BOOL:

        # Boolean function
        assert op_graph.output_nodes[0].outputs[0] == EncryptedScalar(Integer(8, is_signed=False))
    else:

        # Function keeping more or less input type
        input_node_type = inputs["x"]

        expected_output_node_type = deepcopy(input_node_type)

        expected_output_node_type.dtype.bit_width = max(
            expected_output_node_type.dtype.bit_width, 32
        )

        assert op_graph.output_nodes[0].outputs[0] == expected_output_node_type


def test_trace_numpy_ufuncs_not_supported():
    """Testing a failure case of trace_numpy_function"""
    inputs = {"x": EncryptedScalar(Integer(128, is_signed=True))}

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x: numpy.add.reduce(x)  # noqa: E731

    with pytest.raises(NotImplementedError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert "Only __call__ method is supported currently" in str(excinfo.value)


def test_trace_numpy_ufuncs_no_kwargs_no_extra_args():
    """Test a case where kwargs are not allowed and too many inputs are passed"""
    inputs = {
        "x": EncryptedScalar(Integer(32, is_signed=True)),
        "y": EncryptedScalar(Integer(32, is_signed=True)),
        "z": EncryptedScalar(Integer(32, is_signed=True)),
    }

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x, y, z: numpy.add(x, y, z)  # noqa: E731

    with pytest.raises(AssertionError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    # numpy only passes ufunc.nin tracers so the extra arguments are passed as kwargs
    assert "**kwargs are currently not supported for numpy ufuncs, ufunc: add" in str(excinfo.value)

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    function_to_trace = lambda x, y, z: numpy.add(x, y, out=z)  # noqa: E731

    with pytest.raises(AssertionError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert "**kwargs are currently not supported for numpy ufuncs, ufunc: add" in str(excinfo.value)


@pytest.mark.parametrize(
    "function_to_trace,inputs,expected_output_node,expected_output_value",
    [
        pytest.param(
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(Integer(7, is_signed=False), shape=(10,)),
                "y": EncryptedTensor(Integer(7, is_signed=False), shape=(10,)),
            },
            ir.Dot,
            EncryptedScalar(Integer(32, False)),
        ),
        pytest.param(
            lambda x, y: numpy.dot(x, y),
            {
                "x": EncryptedTensor(Float(64), shape=(10,)),
                "y": EncryptedTensor(Float(64), shape=(10,)),
            },
            ir.Dot,
            EncryptedScalar(Float(64)),
        ),
        pytest.param(
            lambda x, y: numpy.dot(x, y),
            {
                "x": ClearTensor(Integer(64, is_signed=True), shape=(6,)),
                "y": ClearTensor(Integer(64, is_signed=True), shape=(6,)),
            },
            ir.Dot,
            ClearScalar(Integer(64, is_signed=True)),
        ),
        pytest.param(
            lambda x: numpy.dot(x, numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)),
            {
                "x": EncryptedTensor(Integer(64, is_signed=True), shape=(5,)),
            },
            ir.Dot,
            EncryptedScalar(Integer(64, True)),
        ),
        pytest.param(
            lambda x: x.dot(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)),
            {
                "x": EncryptedTensor(Integer(64, is_signed=True), shape=(5,)),
            },
            ir.Dot,
            EncryptedScalar(Integer(64, True)),
        ),
    ],
)
def test_trace_numpy_dot(function_to_trace, inputs, expected_output_node, expected_output_value):
    """Function to test dot tracing"""

    op_graph = tracing.trace_numpy_function(function_to_trace, inputs)

    assert len(op_graph.output_nodes) == 1
    assert isinstance(op_graph.output_nodes[0], expected_output_node)
    assert len(op_graph.output_nodes[0].outputs) == 1
    assert op_graph.output_nodes[0].outputs[0] == expected_output_value


@pytest.mark.parametrize("np_function", tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC)
def test_nptracer_get_tracing_func_for_np_functions(np_function):
    """Test NPTracer get_tracing_func_for_np_function"""

    expected_tracing_func = tracing.NPTracer.UFUNC_ROUTING[np_function]

    assert tracing.NPTracer.get_tracing_func_for_np_function(np_function) == expected_tracing_func


def test_nptracer_get_tracing_func_for_np_functions_not_implemented():
    """Check NPTracer in case of not-implemented function"""
    with pytest.raises(NotImplementedError) as excinfo:
        tracing.NPTracer.get_tracing_func_for_np_function(numpy.conjugate)

    assert "NPTracer does not yet manage the following func: conjugate" in str(excinfo.value)


@pytest.mark.parametrize(
    "tracer",
    [
        tracing.NPTracer([], ir.Input(ClearScalar(Integer(32, True)), "x", 0), 0),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x + "fail",
        lambda x: "fail" + x,
        lambda x: x - "fail",
        lambda x: "fail" - x,
        lambda x: x * "fail",
        lambda x: "fail" * x,
    ],
)
def test_nptracer_unsupported_operands(operation, tracer):
    """Test cases where NPTracer cannot be used with other operands."""
    with pytest.raises(TypeError):
        tracer = operation(tracer)


@pytest.mark.parametrize(
    "function_to_trace,input_value_input_and_expected_output_tuples",
    [
        # Indirect calls, like numpy.function(x, ...)
        (
            lambda x: numpy.transpose(x),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([[0, 2], [1, 3]]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4, 8).reshape(2, 2),
                    numpy.array([[4, 6], [5, 7]]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(42),
                ),
            ],
        ),
        (
            lambda x: numpy.transpose(x) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(3, 5).transpose(),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(84),
                ),
            ],
        ),
        (
            lambda x: numpy.ravel(x),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.array([42], dtype=numpy.int64),
                ),
            ],
        ),
        (
            lambda x: numpy.reshape(x, (5, 3)) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(5, 3),
                ),
            ],
        ),
        # Direct calls, like x.function(...)
        (
            lambda x: x.transpose() + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(3, 5).transpose(),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.int64(84),
                ),
            ],
        ),
        (
            lambda x: x.ravel(),
            [
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(4, is_signed=False), shape=(2, 2)),
                    numpy.arange(4).reshape(2, 2),
                    numpy.array([0, 1, 2, 3]),
                ),
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    numpy.array([42], dtype=numpy.int64),
                ),
            ],
        ),
        (
            lambda x: x.reshape((5, 3)) + 42,
            [
                (
                    EncryptedTensor(Integer(32, is_signed=False), shape=(3, 5)),
                    numpy.arange(15).reshape(3, 5),
                    numpy.arange(42, 57).reshape(5, 3),
                ),
            ],
        ),
        pytest.param(
            lambda x: x.reshape((5, 3)),
            [
                (
                    EncryptedTensor(Integer(6, is_signed=False), shape=()),
                    numpy.int64(42),
                    None,
                )
            ],
            marks=pytest.mark.xfail(strict=True, raises=AssertionError),
        ),
    ],
)
def test_tracing_generic_function_memory_ops(
    function_to_trace,
    input_value_input_and_expected_output_tuples,
):
    """Test memory function managed by GenericFunction node"""
    for input_value, input_, expected_output in input_value_input_and_expected_output_tuples:

        op_graph = tracing.trace_numpy_function(function_to_trace, {"x": input_value})
        output_node = op_graph.output_nodes[0]

        node_results = op_graph.evaluate({0: input_})
        evaluated_output = node_results[output_node]
        assert isinstance(evaluated_output, type(expected_output)), type(evaluated_output)
        assert numpy.array_equal(expected_output, evaluated_output)


@pytest.mark.parametrize(
    "lambda_f,params",
    [
        (
            lambda x: numpy.reshape(x, (5, 3)),
            {
                "x": EncryptedTensor(Integer(2, is_signed=False), shape=(7, 5)),
            },
        ),
    ],
)
def test_errors_with_generic_function(lambda_f, params):
    "Test some errors with generic function"
    with pytest.raises(AssertionError) as excinfo:
        tracing.trace_numpy_function(lambda_f, params)

    assert "shapes are not compatible (old shape (7, 5), new shape (5, 3))" in str(excinfo.value)
