"""Test file for bounds evaluation with a dataset"""

import pytest

from hdk.common.bounds_measurement.dataset_eval import eval_op_graph_bounds_on_dataset
from hdk.common.data_types.integers import Integer
from hdk.common.data_types.values import EncryptedValue
from hdk.hnumpy.tracing import trace_numpy_function


@pytest.mark.parametrize(
    "function,input_ranges,expected_output_bounds,expected_output_data_type",
    [
        pytest.param(
            lambda x, y: x + y,
            ((-10, 10), (-10, 10)),
            (-20, 20),
            Integer(6, is_signed=True),
            id="x + y, (-10, 10), (-10, 10), (-20, 20)",
        ),
        pytest.param(
            lambda x, y: x + y,
            ((-10, 2), (-4, 5)),
            (-14, 7),
            Integer(5, is_signed=True),
            id="x + y, (-10, 2), (-4, 5), (-14, 9)",
        ),
        pytest.param(
            lambda x, y: x - y,
            ((-10, 10), (-10, 10)),
            (-20, 20),
            Integer(6, is_signed=True),
            id="x - y, (-10, 10), (-10, 10), (-20, 20)",
        ),
        pytest.param(
            lambda x, y: x - y,
            ((-10, 2), (-4, 5)),
            (-15, 6),
            Integer(5, is_signed=True),
            id="x - y, (-10, 2), (-4, 5), (-15, 6)",
        ),
        pytest.param(
            lambda x, y: x * y,
            ((-10, 10), (-10, 10)),
            (-100, 100),
            Integer(8, is_signed=True),
            id="x * y, (-10, 10), (-10, 10), (-100, 100)",
        ),
        pytest.param(
            lambda x, y: x * y,
            ((-10, 2), (-4, 5)),
            (-50, 40),
            Integer(7, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-50, 40)",
        ),
        pytest.param(
            lambda x, y: x + x + y,
            ((-10, 10), (-10, 10)),
            (-30, 30),
            Integer(6, is_signed=True),
            id="x + x + y, (-10, 10), (-10, 10), (-30, 30)",
        ),
        pytest.param(
            lambda x, y: x - x + y,
            ((-10, 10), (-10, 10)),
            (-10, 10),
            Integer(5, is_signed=True),
            id="x - x + y, (-10, 10), (-10, 10), (-10, 10)",
        ),
        pytest.param(
            lambda x, y: x - x + y,
            ((-10, 2), (-4, 5)),
            (-4, 5),
            Integer(4, is_signed=True),
            id="x - x + y, (-10, 2), (-4, 5), (-4, 5)",
        ),
        pytest.param(
            lambda x, y: x * y - x,
            ((-10, 10), (-10, 10)),
            (-110, 110),
            Integer(8, is_signed=True),
            id="x * y - x, (-10, 10), (-10, 10), (-110, 110)",
        ),
        pytest.param(
            lambda x, y: x * y - x,
            ((-10, 2), (-4, 5)),
            (-40, 50),
            Integer(7, is_signed=True),
            id="x * y - x, (-10, 2), (-4, 5), (-40, 50),",
        ),
    ],
)
def test_eval_op_graph_bounds_on_dataset(
    function,
    input_ranges,
    expected_output_bounds,
    expected_output_data_type: Integer,
):
    """Test function for eval_op_graph_bounds_on_dataset"""

    op_graph = trace_numpy_function(
        function, {"x": EncryptedValue(Integer(64, True)), "y": EncryptedValue(Integer(64, True))}
    )

    def data_gen(range_x, range_y):
        for x_gen in range_x:
            for y_gen in range_y:
                yield (x_gen, y_gen)

    node_bounds = eval_op_graph_bounds_on_dataset(
        op_graph, data_gen(*tuple(map(lambda x: range(x[0], x[1] + 1), input_ranges)))
    )

    output_node = op_graph.output_nodes[0]
    output_node_bounds = node_bounds[output_node]

    assert (output_node_bounds["min"], output_node_bounds["max"]) == expected_output_bounds

    assert EncryptedValue(Integer(64, True)) == output_node.outputs[0]
    op_graph.update_values_with_bounds(node_bounds)

    assert expected_output_data_type == output_node.outputs[0].data_type
