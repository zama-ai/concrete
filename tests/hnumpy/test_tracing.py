"""Test file for HDK's hnumpy tracing"""

import networkx as nx
import pytest

from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import ClearValue, EncryptedValue
from hdk.common.representation import intermediate as ir
from hdk.hnumpy import tracing


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="Encrypted uint"),
        pytest.param(
            EncryptedValue(Integer(64, is_signed=True)),
            id="Encrypted int",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=False)),
            id="Clear uint",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=True)),
            id="Clear int",
        ),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        pytest.param(EncryptedValue(Integer(64, is_signed=False)), id="Encrypted uint"),
        pytest.param(
            EncryptedValue(Integer(64, is_signed=True)),
            id="Encrypted int",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=False)),
            id="Clear uint",
        ),
        pytest.param(
            ClearValue(Integer(64, is_signed=True)),
            id="Clear int",
        ),
    ],
)
def test_hnumpy_tracing_add(x, y, test_helpers):
    "Test hnumpy tracing __add__"

    def simple_add_function(x, y):
        z = x + x
        return z + y

    graph = tracing.trace_numpy_function(simple_add_function, {"x": x, "y": y})

    ref_graph = nx.MultiDiGraph()

    input_x = ir.Input((x,))
    input_y = ir.Input((y,))

    add_node_z = ir.Add(
        (
            input_x.outputs[0],
            input_x.outputs[0],
        )
    )

    return_add_node = ir.Add(
        (
            add_node_z.outputs[0],
            input_y.outputs[0],
        )
    )

    ref_graph.add_node(input_x, content=input_x)
    ref_graph.add_node(input_y, content=input_y)
    ref_graph.add_node(add_node_z, content=add_node_z)
    ref_graph.add_node(return_add_node, content=return_add_node)

    ref_graph.add_edge(input_x, add_node_z, input_idx=0)
    ref_graph.add_edge(input_x, add_node_z, input_idx=1)

    ref_graph.add_edge(add_node_z, return_add_node, input_idx=0)
    ref_graph.add_edge(input_y, return_add_node, input_idx=1)

    assert test_helpers.digraphs_are_equivalent(ref_graph, graph)
