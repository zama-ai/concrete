"""Test file for HDK's hnumpy tracing"""

import networkx as nx
import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.representation import intermediate as ir
from hdk.hnumpy import tracing


@pytest.mark.parametrize(
    "operation",
    [ir.Add, ir.Sub, ir.Mul],
)
@pytest.mark.parametrize(
    "x",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="x: Encrypted uint"),
        pytest.param(
            EncryptedValue(Integer(64, is_signed=True)),
            id="x: Encrypted int",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=False)),
            id="x: Clear uint",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=True)),
            id="x: Clear int",
        ),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="y: Encrypted uint"),
        pytest.param(
            EncryptedValue(Integer(64, is_signed=True)),
            id="y: Encrypted int",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=False)),
            id="y: Clear uint",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=True)),
            id="y: Clear int",
        ),
    ],
)
def test_hnumpy_tracing_binary_op(operation, x, y, test_helpers):
    "Test hnumpy tracing a binary operation (in the supported ops)"

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

    if operation == ir.Add:
        function_to_compile = simple_add_function
    elif operation == ir.Sub:
        function_to_compile = simple_sub_function
    elif operation == ir.Mul:
        function_to_compile = simple_mul_function
    else:
        assert False, f"unknown operation {operation}"

    graph = tracing.trace_numpy_function(function_to_compile, {"x": x, "y": y})

    ref_graph = nx.MultiDiGraph()

    input_x = ir.Input(x, input_name="x")
    input_y = ir.Input(y, input_name="y")

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

    ref_graph.add_node(input_x, content=input_x)
    ref_graph.add_node(input_y, content=input_y)
    ref_graph.add_node(add_node_z, content=add_node_z)
    ref_graph.add_node(returned_final_node, content=returned_final_node)

    ref_graph.add_edge(input_x, add_node_z, input_idx=0)
    ref_graph.add_edge(input_x, add_node_z, input_idx=1)

    ref_graph.add_edge(add_node_z, returned_final_node, input_idx=0)
    ref_graph.add_edge(input_y, returned_final_node, input_idx=1)

    assert test_helpers.digraphs_are_equivalent(ref_graph, graph)
