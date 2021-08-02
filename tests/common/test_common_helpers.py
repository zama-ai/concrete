"""Test file for common helpers"""

from copy import deepcopy

from hdk.common import check_op_graph_is_integer_program
from hdk.common.data_types.base import BaseDataType
from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import EncryptedValue
from hdk.hnumpy.tracing import trace_numpy_function


class DummyNotInteger(BaseDataType):
    """Dummy helper data type class"""


def test_check_op_graph_is_integer_program():
    """Test function for check_op_graph_is_integer_program"""

    def function(x, y):
        return x + y - y * y + x * y

    op_graph = trace_numpy_function(
        function, {"x": EncryptedValue(Integer(64, True)), "y": EncryptedValue(Integer(64, True))}
    )

    # Test without and with output list
    offending_nodes = []
    assert check_op_graph_is_integer_program(op_graph)
    assert check_op_graph_is_integer_program(op_graph, offending_nodes)
    assert len(offending_nodes) == 0

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.output_nodes[0].outputs[0].data_type = DummyNotInteger()

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 1
    assert offending_nodes == [op_graph_copy.output_nodes[0]]

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.input_nodes[0].inputs[0].data_type = DummyNotInteger()

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 1
    assert offending_nodes == [op_graph_copy.input_nodes[0]]

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.input_nodes[0].inputs[0].data_type = DummyNotInteger()
    op_graph_copy.input_nodes[1].inputs[0].data_type = DummyNotInteger()

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 2
    assert set(offending_nodes) == set([op_graph_copy.input_nodes[0], op_graph_copy.input_nodes[1]])
