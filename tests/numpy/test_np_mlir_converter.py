"""Test file for numpy mlir converter"""

import math

import numpy
import pytest

import concrete.numpy as hnp
from concrete.common.representation.intermediate import GenericFunction
from concrete.numpy.np_mlir_converter import generate_deduplicated_tables


def multi_tlu_func(x, cst):
    """Multi TLU function"""
    y = x + cst
    return y.astype(numpy.int32)


RESNET_BIGGEST_SHAPE = (64, 112, 112)
RESNET_BIGGEST_SIZE = math.prod(RESNET_BIGGEST_SHAPE)


@pytest.mark.parametrize(
    "function,expected_number_of_tables",
    [
        (
            lambda x: multi_tlu_func(x, numpy.zeros(RESNET_BIGGEST_SHAPE, dtype=numpy.float64)),
            1,
        ),
        (
            lambda x: multi_tlu_func(
                x,
                numpy.arange(RESNET_BIGGEST_SIZE, dtype=numpy.float64).reshape(
                    RESNET_BIGGEST_SHAPE
                ),
            ),
            RESNET_BIGGEST_SIZE,
        ),
    ],
)
def test_generate_deduplicated_tables(
    function, expected_number_of_tables, default_compilation_configuration
):
    """Test function for generate_deduplicated_tables"""
    op_graph = hnp.compile_numpy_function_into_op_graph_and_measure_bounds(
        function,
        {"x": hnp.EncryptedTensor(hnp.Integer(7, False), RESNET_BIGGEST_SHAPE)},
        (i * numpy.ones(RESNET_BIGGEST_SHAPE, dtype=numpy.int32) for i in range(128)),
        default_compilation_configuration,
    )

    univariate_function_nodes = [
        node for node in op_graph.graph.nodes() if isinstance(node, GenericFunction)
    ]

    assert len(univariate_function_nodes) == 1

    tlu_node = univariate_function_nodes[0]

    deduplication_result = generate_deduplicated_tables(
        tlu_node, op_graph.get_ordered_preds(tlu_node)
    )

    assert len(deduplication_result) == expected_number_of_tables


def test_deduplicated_tables_correctness(default_compilation_configuration):
    """Check the deduplicated tables are the expected ones"""

    tensor_shape = (2, 2)

    op_graph = hnp.compile_numpy_function_into_op_graph_and_measure_bounds(
        lambda x: multi_tlu_func(x, numpy.arange(4, dtype=numpy.float64).reshape(tensor_shape)),
        {"x": hnp.EncryptedTensor(hnp.Integer(2, False), tensor_shape)},
        (i * numpy.ones(tensor_shape, dtype=numpy.int32) for i in range(4)),
        default_compilation_configuration,
    )

    univariate_function_nodes = [
        node for node in op_graph.graph.nodes() if isinstance(node, GenericFunction)
    ]

    assert len(univariate_function_nodes) == 1

    tlu_node = univariate_function_nodes[0]

    deduplication_result = generate_deduplicated_tables(
        tlu_node, op_graph.get_ordered_preds(tlu_node)
    )

    expected_result = tuple(
        (
            numpy.arange(i, 4 + i, dtype=numpy.int32),
            [
                numpy.unravel_index(i, tensor_shape),
            ],
        )
        for i in range(4)
    )

    assert len(deduplication_result) == len(expected_result)
    for computed_array, computed_idx in deduplication_result:
        for expected_array, expected_idx in expected_result:
            if numpy.array_equal(computed_array, expected_array) and computed_idx == expected_idx:
                break
        else:
            raise AssertionError(
                f"Could not find {(computed_array, computed_idx)} "
                f"in expected_result: {expected_result}"
            )
