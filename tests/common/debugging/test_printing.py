"""Test file for printing"""

from concrete.common.data_types.integers import Integer
from concrete.common.debugging import format_operation_graph
from concrete.common.values import EncryptedScalar
from concrete.numpy.compile import compile_numpy_function_into_op_graph


def test_format_operation_graph_with_offending_nodes(default_compilation_configuration):
    """Test format_operation_graph with offending nodes"""

    def function(x):
        return x + 42

    opgraph = compile_numpy_function_into_op_graph(
        function,
        {"x": EncryptedScalar(Integer(7, True))},
        [(i,) for i in range(-5, 5)],
        default_compilation_configuration,
    )

    highlighted_nodes = {opgraph.input_nodes[0]: ["foo"]}

    without_types = format_operation_graph(
        opgraph, show_data_types=False, highlighted_nodes=highlighted_nodes
    ).strip()
    with_types = format_operation_graph(
        opgraph, show_data_types=True, highlighted_nodes=highlighted_nodes
    ).strip()

    assert (
        without_types
        == """

%0 = x
^^^^^^ foo
%1 = Constant(42)
%2 = Add(%0, %1)
return(%2)

""".strip()
    )

    assert (
        with_types
        == """

%0 = x                                             # EncryptedScalar<Integer<signed, 4 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ foo
%1 = Constant(42)                                  # ClearScalar<Integer<unsigned, 6 bits>>
%2 = Add(%0, %1)                                   # EncryptedScalar<Integer<unsigned, 6 bits>>
return(%2)

""".strip()
    )

    highlighted_nodes = {opgraph.input_nodes[0]: ["foo"], opgraph.output_nodes[0]: ["bar", "baz"]}

    without_types = format_operation_graph(
        opgraph, show_data_types=False, highlighted_nodes=highlighted_nodes
    ).strip()
    with_types = format_operation_graph(
        opgraph, show_data_types=True, highlighted_nodes=highlighted_nodes
    ).strip()

    assert (
        without_types
        == """

%0 = x
^^^^^^ foo
%1 = Constant(42)
%2 = Add(%0, %1)
^^^^^^^^^^^^^^^^ bar
                 baz
return(%2)

""".strip()
    )

    assert (
        with_types
        == """

%0 = x                                             # EncryptedScalar<Integer<signed, 4 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ foo
%1 = Constant(42)                                  # ClearScalar<Integer<unsigned, 6 bits>>
%2 = Add(%0, %1)                                   # EncryptedScalar<Integer<unsigned, 6 bits>>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ bar
                                                                                                baz
return(%2)

""".strip()
    )
