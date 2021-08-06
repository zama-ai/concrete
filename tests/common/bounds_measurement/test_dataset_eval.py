"""Test file for bounds evaluation with a dataset"""

from typing import Tuple

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
            id="x + y, (-10, 2), (-4, 5), (-14, 7)",
        ),
        pytest.param(
            lambda x, y: x + y + 1,
            ((-10, 2), (-4, 5)),
            (-13, 8),
            Integer(5, is_signed=True),
            id="x + y + 1, (-10, 2), (-4, 5), (-13, 8)",
        ),
        pytest.param(
            lambda x, y: x + y + (-3),
            ((-10, 2), (-4, 5)),
            (-17, 4),
            Integer(6, is_signed=True),
            id="x + y + 1, (-10, 2), (-4, 5), (-17, 4)",
        ),
        pytest.param(
            lambda x, y: (1 + x) + y,
            ((-10, 2), (-4, 5)),
            (-13, 8),
            Integer(5, is_signed=True),
            id="(1 + x) + y, (-10, 2), (-4, 5), (-13, 8)",
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
            lambda x, y: x - y - 42,
            ((-10, 2), (-4, 5)),
            (-57, -36),
            Integer(7, is_signed=True),
            id="x - y, (-10, 2), (-4, 5), (-57, -36)",
        ),
        pytest.param(
            lambda x, y: 3 - x + y,
            ((-10, 2), (-4, 5)),
            (-3, 18),
            Integer(6, is_signed=True),
            id="x - y, (-10, 2), (-4, 5), (-3, 18)",
        ),
        pytest.param(
            lambda x, y: (-13) - x + y,
            ((-10, 2), (-4, 5)),
            (-19, 2),
            Integer(6, is_signed=True),
            id="x - y, (-10, 2), (-4, 5), (-16, 2)",
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
            lambda x, y: (3 * x) * y,
            ((-10, 2), (-4, 5)),
            (-150, 120),
            Integer(9, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-150, 120)",
        ),
        pytest.param(
            lambda x, y: (x * 11) * y,
            ((-10, 2), (-4, 5)),
            (-550, 440),
            Integer(11, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-550, 440)",
        ),
        pytest.param(
            lambda x, y: (x * (-11)) * y,
            ((-10, 2), (-4, 5)),
            (-440, 550),
            Integer(11, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-440, 550)",
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
        pytest.param(
            lambda x, y: (x * 3) * y - (x + 3) + (y - 13) + x * (11 + y) * (12 + y) + (15 - x),
            ((-10, 2), (-4, 5)),
            (-2846, 574),
            Integer(13, is_signed=True),
            id="x * y - x, (-10, 2), (-4, 5), (-2846, 574),",
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

    test_eval_op_graph_bounds_on_dataset_multiple_output(
        function,
        input_ranges,
        (expected_output_bounds,),
        (expected_output_data_type,),
    )


@pytest.mark.parametrize(
    "function,input_ranges,expected_output_bounds,expected_output_data_type",
    [
        pytest.param(
            lambda x, y: (x + 1, y + 10),
            ((-1, 1), (3, 4)),
            ((0, 2), (13, 14)),
            (Integer(2, is_signed=False), Integer(4, is_signed=False)),
        ),
        pytest.param(
            lambda x, y: (x + y + 1, x * y + 42),
            ((-1, 1), (3, 4)),
            ((3, 6), (38, 46)),
            (Integer(3, is_signed=False), Integer(6, is_signed=False)),
        ),
    ],
)
def test_eval_op_graph_bounds_on_dataset_multiple_output(
    function,
    input_ranges,
    expected_output_bounds,
    expected_output_data_type: Tuple[Integer],
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
        op_graph, data_gen(*tuple(range(x[0], x[1] + 1) for x in input_ranges))
    )

    for i, output_node in op_graph.output_nodes.items():
        output_node_bounds = node_bounds[output_node]

        assert (output_node_bounds["min"], output_node_bounds["max"]) == expected_output_bounds[i]

        assert EncryptedValue(Integer(64, True)) == output_node.outputs[0]

    op_graph.update_values_with_bounds(node_bounds)

    for i, output_node in op_graph.output_nodes.items():
        assert expected_output_data_type[i] == output_node.outputs[0].data_type
