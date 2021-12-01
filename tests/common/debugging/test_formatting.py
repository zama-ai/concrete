"""Test file for formatting"""

import numpy

from concrete.common.data_types.integers import Integer, UnsignedInteger
from concrete.common.debugging import format_operation_graph
from concrete.common.values import EncryptedScalar
from concrete.numpy.compile import (
    compile_numpy_function,
    compile_numpy_function_into_op_graph_and_measure_bounds,
)


def test_format_operation_graph_with_multiple_edges(default_compilation_configuration):
    """Test format_operation_graph with multiple edges"""

    def function(x):
        return x + x

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        {"x": EncryptedScalar(Integer(4, True))},
        range(0, 10),
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
        range(-5, 5),
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


def test_format_operation_graph_with_fusing(default_compilation_configuration):
    """Test format_operation_graph with fusing"""

    def function(x):
        return (10 * (numpy.cos(x + 1) + 1)).astype(numpy.uint32)

    circuit = compile_numpy_function(
        function,
        {
            "x": EncryptedScalar(UnsignedInteger(3)),
        },
        range(2 ** 3),
        default_compilation_configuration,
    )

    assert (
        str(circuit)
        == """

%0 = x                   # EncryptedScalar<uint5>
%1 = 1                   # ClearScalar<uint6>
%2 = add(%0, %1)         # EncryptedScalar<uint5>
%3 = subgraph(%2)        # EncryptedScalar<uint5>
return %3

Subgraphs:

    %3 = subgraph(%2):

        %0 = 10                              # ClearScalar<uint4>
        %1 = 1                               # ClearScalar<uint1>
        %2 = float_subgraph_input            # EncryptedScalar<uint3>
        %3 = cos(%2)                         # EncryptedScalar<float64>
        %4 = add(%3, %1)                     # EncryptedScalar<float64>
        %5 = mul(%4, %0)                     # EncryptedScalar<float64>
        %6 = astype(%5, dtype=uint32)        # EncryptedScalar<uint5>
        return %6

""".strip()
    ), str(circuit)
