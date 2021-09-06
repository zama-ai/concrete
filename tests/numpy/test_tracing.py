"""Test file for numpy tracing"""

import networkx as nx
import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.representation import intermediate as ir
from concrete.common.values import (
    ClearScalar,
    ClearTensor,
    EncryptedScalar,
    EncryptedTensor,
)
from concrete.numpy import tracing

OPERATIONS_TO_TEST = [ir.Add, ir.Sub, ir.Mul]


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

    ref_graph.add_edge(input_x, add_node_z, input_idx=0)
    ref_graph.add_edge(input_x, add_node_z, input_idx=1)

    ref_graph.add_edge(add_node_z, returned_final_node, input_idx=0)
    ref_graph.add_edge(input_y, returned_final_node, input_idx=1)

    assert test_helpers.digraphs_are_equivalent(ref_graph, op_graph.graph)


@pytest.mark.parametrize(
    "tensor_constructor",
    [
        EncryptedTensor,
        ClearTensor,
    ],
)
def test_numpy_tracing_tensor_constant(tensor_constructor):
    "Test numpy tracing tensor constant"

    def simple_add_tensor(x):
        return x + numpy.array([[1, 2], [3, 4]], dtype=numpy.int32)

    op_graph = tracing.trace_numpy_function(
        simple_add_tensor, {"x": tensor_constructor(Integer(32, True), shape=(2, 2))}
    )

    constant_inputs = [node for node in op_graph.graph.nodes() if isinstance(node, ir.Constant)]
    assert len(constant_inputs) == 1

    constant_input_data = constant_inputs[0].constant_data

    assert (constant_input_data == numpy.array([[1, 2], [3, 4]], dtype=numpy.int32)).all()
    assert op_graph.get_ordered_outputs()[0].outputs[0].shape == constant_input_data.shape


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
        assert op_graph_expected_output_type == output_node.outputs[0].data_type

        node_results = op_graph.evaluate({0: numpy.array(input_)})
        evaluated_output = node_results[output_node]
        assert isinstance(evaluated_output, type(expected_output))
        assert expected_output == evaluated_output


@pytest.mark.parametrize(
    "inputs,expected_output_node,expected_output_value",
    [
        pytest.param(
            {"x": EncryptedScalar(Integer(7, is_signed=False))},
            ir.ArbitraryFunction,
            EncryptedScalar(Float(64)),
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(32, is_signed=True))},
            ir.ArbitraryFunction,
            EncryptedScalar(Float(64)),
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(64, is_signed=True))},
            ir.ArbitraryFunction,
            EncryptedScalar(Float(64)),
        ),
        pytest.param(
            {"x": EncryptedScalar(Integer(128, is_signed=True))},
            ir.ArbitraryFunction,
            None,
            marks=pytest.mark.xfail(strict=True, raises=NotImplementedError),
        ),
        pytest.param(
            {"x": EncryptedScalar(Float(64))},
            ir.ArbitraryFunction,
            EncryptedScalar(Float(64)),
        ),
    ],
)
def test_trace_numpy_supported_ufuncs(inputs, expected_output_node, expected_output_value):
    """Function to trace supported numpy ufuncs"""
    for function_to_trace_def in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC:

        # We really need a lambda (because numpy functions are not playing
        # nice with inspect.signature), but pylint and flake8 are not happy
        # with it
        # pylint: disable=unnecessary-lambda,cell-var-from-loop
        function_to_trace = lambda x: function_to_trace_def(x)  # noqa: E731
        # pylint: enable=unnecessary-lambda,cell-var-from-loop

        op_graph = tracing.trace_numpy_function(function_to_trace, inputs)

        assert len(op_graph.output_nodes) == 1
        assert isinstance(op_graph.output_nodes[0], expected_output_node)
        assert len(op_graph.output_nodes[0].outputs) == 1
        assert op_graph.output_nodes[0].outputs[0] == expected_output_value


def test_trace_numpy_ufuncs_not_supported():
    """Testing a failure case of trace_numpy_function"""
    inputs = {"x": EncryptedScalar(Integer(128, is_signed=True))}

    # We really need a lambda (because numpy functions are not playing
    # nice with inspect.signature), but pylint and flake8 are not happy
    # with it
    # pylint: disable=unnecessary-lambda
    function_to_trace = lambda x: numpy.add.reduce(x)  # noqa: E731
    # pylint: enable=unnecessary-lambda

    with pytest.raises(NotImplementedError) as excinfo:
        tracing.trace_numpy_function(function_to_trace, inputs)

    assert "Only __call__ method is supported currently" in str(excinfo.value)


@pytest.mark.parametrize(
    "function_to_trace,inputs,expected_output_node,expected_output_value",
    [
        # pylint: disable=unnecessary-lambda
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
                "x": EncryptedTensor(Float(64), shape=(42,)),
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
        # pylint: enable=unnecessary-lambda
    ],
)
def test_trace_numpy_dot(function_to_trace, inputs, expected_output_node, expected_output_value):
    """Function to test dot tracing"""

    op_graph = tracing.trace_numpy_function(function_to_trace, inputs)

    assert len(op_graph.output_nodes) == 1
    assert isinstance(op_graph.output_nodes[0], expected_output_node)
    assert len(op_graph.output_nodes[0].outputs) == 1
    assert op_graph.output_nodes[0].outputs[0] == expected_output_value


def test_nptracer_get_tracing_func_for_np_functions():
    """Test NPTracer get_tracing_func_for_np_function"""

    for np_function in tracing.NPTracer.LIST_OF_SUPPORTED_UFUNC:
        expected_tracing_func = tracing.NPTracer.UFUNC_ROUTING[np_function]

        assert (
            tracing.NPTracer.get_tracing_func_for_np_function(np_function) == expected_tracing_func
        )


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
