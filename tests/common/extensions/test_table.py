"""Test file for direct table lookups"""

from copy import deepcopy

import networkx as nx
import pytest

from concrete.common import is_a_power_of_2
from concrete.common.data_types.integers import Integer
from concrete.common.extensions.table import LookupTable
from concrete.common.representation import intermediate as ir
from concrete.common.values import EncryptedScalar
from concrete.numpy import tracing


def test_lookup_table_size_constraints():
    """Test function to make sure lookup tables have correct size"""

    table = []

    # creating empty lookup table is not acceptable
    with pytest.raises(ValueError):
        LookupTable(table)

    for _ in range(512):
        table.append(0)

        if is_a_power_of_2(len(table)):
            # creating lookup table with 2^N entries are acceptable
            LookupTable(table)
        else:
            # creating lookup table with anything other than 2^N entries are not acceptable
            with pytest.raises(ValueError):
                LookupTable(table)


def test_lookup_table_encrypted_lookup(test_helpers):
    """Test function for tracing with explicit table lookups using encrypted inputs"""

    table = LookupTable([3, 6, 0, 2])

    def f(x):
        return table[x]

    x = EncryptedScalar(Integer(2, is_signed=False))
    op_graph = tracing.trace_numpy_function(f, {"x": x})

    table_node = op_graph.output_nodes[0]

    assert table_node.get_table(op_graph.get_ordered_preds(table_node)) == [3, 6, 0, 2]

    ref_graph = nx.MultiDiGraph()
    # Here is the ASCII drawing of the expected graph:
    #   (x) - (TLU)

    input_x = ir.Input(input_value=x, input_name="x", program_input_idx=0)
    ref_graph.add_node(input_x)

    generic_function_output_value = deepcopy(x)
    generic_function_output_value.dtype = table.output_dtype

    # pylint: disable=protected-access
    # Need access to _checked_indexing to have is_equivalent_to work for ir.GenericFunction
    output_arbitrary_function = ir.GenericFunction(
        inputs=[x],
        arbitrary_func=LookupTable._checked_indexing,
        output_value=generic_function_output_value,
        op_kind="TLU",
        op_kwargs={"table": deepcopy(table.table)},
        op_name="TLU",
    )
    # pylint: enable=protected-access
    ref_graph.add_node(output_arbitrary_function)

    ref_graph.add_edge(input_x, output_arbitrary_function, input_idx=0, output_idx=0)

    # TODO: discuss if this check is enough as == is not overloaded properly for GenericFunction
    assert test_helpers.digraphs_are_equivalent(ref_graph, op_graph.graph)


def test_lookup_table_encrypted_and_plain_lookup(test_helpers):
    """Test function for tracing with explicit table lookups using encrypted and plain inputs"""

    table = LookupTable([3, 6, 0, 2, 1, 4, 5, 7])

    def f(x):
        return table[x] + table[0]

    x = EncryptedScalar(Integer(3, is_signed=False))
    op_graph = tracing.trace_numpy_function(f, {"x": x})

    ref_graph = nx.MultiDiGraph()
    # Here is the ASCII drawing of the expected graph:
    #   (x) - (TLU)
    #              \
    #              (+)
    #              /
    #           (3)

    input_x = ir.Input(input_value=x, input_name="x", program_input_idx=0)
    ref_graph.add_node(input_x)

    generic_function_output_value = deepcopy(x)
    generic_function_output_value.dtype = table.output_dtype

    # pylint: disable=protected-access
    # Need access to _checked_indexing to have is_equivalent_to work for ir.GenericFunction
    intermediate_arbitrary_function = ir.GenericFunction(
        inputs=[x],
        arbitrary_func=LookupTable._checked_indexing,
        output_value=generic_function_output_value,
        op_kind="TLU",
        op_kwargs={"table": deepcopy(table.table)},
        op_name="TLU",
    )
    # pylint: enable=protected-access
    ref_graph.add_node(intermediate_arbitrary_function)

    constant_3 = ir.Constant(3)
    ref_graph.add_node(constant_3)

    output_add = ir.Add((intermediate_arbitrary_function.outputs[0], constant_3.outputs[0]))
    ref_graph.add_node(output_add)

    ref_graph.add_edge(input_x, intermediate_arbitrary_function, input_idx=0, output_idx=0)

    ref_graph.add_edge(intermediate_arbitrary_function, output_add, input_idx=0, output_idx=0)
    ref_graph.add_edge(constant_3, output_add, input_idx=1, output_idx=0)

    # TODO: discuss if this check is enough as == is not overloaded properly for GenericFunction
    assert test_helpers.digraphs_are_equivalent(ref_graph, op_graph.graph)
