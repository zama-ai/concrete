"""Test file for printing"""

from concrete.common.data_types.integers import Integer
from concrete.common.debugging import get_printable_graph
from concrete.common.values import EncryptedScalar
from concrete.numpy.compile import compile_numpy_function_into_op_graph


def test_get_printable_graph_with_offending_nodes(default_compilation_configuration):
    """Test get_printable_graph with offending nodes"""

    def function(x):
        return x + 42

    opgraph = compile_numpy_function_into_op_graph(
        function,
        {"x": EncryptedScalar(Integer(7, True))},
        [(i,) for i in range(-5, 5)],
        default_compilation_configuration,
    )

    highlighted_nodes = {opgraph.input_nodes[0]: "foo"}

    without_types = get_printable_graph(
        opgraph, show_data_types=False, highlighted_nodes=highlighted_nodes
    ).strip()
    with_types = get_printable_graph(
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

    highlighted_nodes = {opgraph.input_nodes[0]: "foo", opgraph.output_nodes[0]: "bar"}

    without_types = get_printable_graph(
        opgraph, show_data_types=False, highlighted_nodes=highlighted_nodes
    ).strip()
    with_types = get_printable_graph(
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
return(%2)

""".strip()
    )
