"""Test file for formatting"""

from concrete.common.data_types.integers import Integer
from concrete.common.debugging import format_operation_graph
from concrete.common.values import EncryptedScalar
from concrete.numpy.compile import compile_numpy_function_into_op_graph_and_measure_bounds


def test_format_operation_graph_with_multiple_edges(default_compilation_configuration):
    """Test format_operation_graph with multiple edges"""

    def function(x):
        return x + x

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        {"x": EncryptedScalar(Integer(4, True))},
        [(i,) for i in range(0, 10)],
        default_compilation_configuration,
    )

    formatted_graph = format_operation_graph(op_graph)
    assert (
        formatted_graph
        == """

%0 = x                  # EncryptedScalar<uint4>
%1 = add(%0, %0)        # EncryptedScalar<uint5>
return %1

""".strip()
    )


def test_format_operation_graph_with_offending_nodes(default_compilation_configuration):
    """Test format_operation_graph with offending nodes"""

    def function(x):
        return x + 42

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        {"x": EncryptedScalar(Integer(7, True))},
        [(i,) for i in range(-5, 5)],
        default_compilation_configuration,
    )

    highlighted_nodes = {op_graph.input_nodes[0]: ["foo"]}
    formatted_graph = format_operation_graph(op_graph, highlighted_nodes=highlighted_nodes).strip()
    assert (
        formatted_graph
        == """

%0 = x                  # EncryptedScalar<int4>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ foo
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedScalar<uint6>
return %2

""".strip()
    )

    highlighted_nodes = {op_graph.input_nodes[0]: ["foo"], op_graph.output_nodes[0]: ["bar", "baz"]}
    formatted_graph = format_operation_graph(op_graph, highlighted_nodes=highlighted_nodes).strip()
    assert (
        formatted_graph
        == """

%0 = x                  # EncryptedScalar<int4>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ foo
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedScalar<uint6>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ bar
                                                 baz
return %2

""".strip()
    )
