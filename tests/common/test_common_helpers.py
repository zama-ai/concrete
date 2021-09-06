"""Test file for common helpers"""

from copy import deepcopy

import pytest

from concrete.common import check_op_graph_is_integer_program, is_a_power_of_2
from concrete.common.data_types.floats import Float64
from concrete.common.data_types.integers import Integer
from concrete.common.values import EncryptedScalar
from concrete.numpy.tracing import trace_numpy_function


@pytest.mark.parametrize(
    "x,result",
    [
        (0, False),
        (1, True),
        (2, True),
        (3, False),
        (4, True),
        (10, False),
        (16, True),
    ],
)
def test_is_a_power_of_2(x, result):
    """Test function for test_is_a_power_of_2"""

    assert is_a_power_of_2(x) == result


def test_check_op_graph_is_integer_program():
    """Test function for check_op_graph_is_integer_program"""

    def function(x, y):
        return x + y - y * y + x * y

    op_graph = trace_numpy_function(
        function, {"x": EncryptedScalar(Integer(64, True)), "y": EncryptedScalar(Integer(64, True))}
    )

    # Test without and with output list
    offending_nodes = []
    assert check_op_graph_is_integer_program(op_graph)
    assert check_op_graph_is_integer_program(op_graph, offending_nodes)
    assert len(offending_nodes) == 0

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.output_nodes[0].outputs[0].data_type = Float64

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 1
    assert offending_nodes == [op_graph_copy.output_nodes[0]]

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.input_nodes[0].inputs[0].data_type = Float64

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 1
    assert offending_nodes == [op_graph_copy.input_nodes[0]]

    op_graph_copy = deepcopy(op_graph)
    op_graph_copy.input_nodes[0].inputs[0].data_type = Float64
    op_graph_copy.input_nodes[1].inputs[0].data_type = Float64

    offending_nodes = []
    assert not check_op_graph_is_integer_program(op_graph_copy)
    assert not check_op_graph_is_integer_program(op_graph_copy, offending_nodes)
    assert len(offending_nodes) == 2
    assert set(offending_nodes) == set([op_graph_copy.input_nodes[0], op_graph_copy.input_nodes[1]])
