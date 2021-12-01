"""Test module for constant indexing."""

import numpy as np
import pytest

from concrete.common.data_types import UnsignedInteger
from concrete.common.values import EncryptedScalar, EncryptedTensor
from concrete.numpy import (
    compile_numpy_function,
    compile_numpy_function_into_op_graph_and_measure_bounds,
)


@pytest.mark.parametrize(
    "input_value,function_with_indexing,output_value",
    [
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2:],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1:],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1:],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2:],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:-2],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:2],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:3],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:-2],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:2],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3:3],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2:2],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2:3],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1:3],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:-2],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:2],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:3],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1:2],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1:3],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2:3],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-3::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-2::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2::-1],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:-3:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:-2:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:0:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:1:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2:0:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2:1:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1:1:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(1,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[-1:0:-1],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[:, :, :],
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0, :, :],
            EncryptedTensor(UnsignedInteger(1), shape=(4, 5)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[:, 0, :],
            EncryptedTensor(UnsignedInteger(1), shape=(3, 5)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[:, :, 0],
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0, 0, :],
            EncryptedTensor(UnsignedInteger(1), shape=(5,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0, :, 0],
            EncryptedTensor(UnsignedInteger(1), shape=(4,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[:, 0, 0],
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0:, 1:, 2:],
            EncryptedTensor(UnsignedInteger(1), shape=(3, 3, 3)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[2:, 1:, 0:],
            EncryptedTensor(UnsignedInteger(1), shape=(1, 3, 5)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0],
            EncryptedTensor(UnsignedInteger(1), shape=(4, 5)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0, 0],
            EncryptedTensor(UnsignedInteger(1), shape=(5,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3, 4, 5)),
            lambda x: x[0, 0, 0],
            EncryptedScalar(UnsignedInteger(1)),
        ),
    ],
)
def test_constant_indexing(
    default_compilation_configuration,
    input_value,
    function_with_indexing,
    output_value,
):
    """Test compile_numpy_function_into_op_graph with constant indexing"""

    inputset = [
        np.random.randint(
            input_value.dtype.min_value(),
            input_value.dtype.max_value() + 1,
            size=input_value.shape,
        )
        for _ in range(10)
    ]

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function_with_indexing,
        {"x": input_value},
        inputset,
        default_compilation_configuration,
    )

    assert len(op_graph.output_nodes) == 1
    output_node = op_graph.output_nodes[0]

    assert len(output_node.outputs) == 1
    assert output_value == output_node.outputs[0]


@pytest.mark.parametrize(
    "input_value,function_with_indexing,expected_error_type,expected_error_message",
    [
        pytest.param(
            EncryptedScalar(UnsignedInteger(1)),
            lambda x: x[0],
            TypeError,
            "Only tensors can be indexed but you tried to index EncryptedScalar<uint1>",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0.5],
            TypeError,
            "Only integers and integer slices can be used for indexing "
            "but you tried to use 0.5 for indexing",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[1:5:0.5],  # type: ignore
            TypeError,
            "Only integers and integer slices can be used for indexing "
            "but you tried to use 1:5:0.5 for indexing",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0, 1],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [0, 1] "
            "as the index has more elements than the number of dimensions of the tensor",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[5],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [5] "
            "because index is out of range for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[5:],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [5:] "
            "because start index is out of range for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:10],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [:10] "
            "because stop index is out of range for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[2:0],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [2:0] "
            "because start index is not less than stop index for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[5::-1],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [5::-1] "
            "because start index is out of range for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[:10:-1],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [:10:-1] "
            "because stop index is out of range for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[0:2:-1],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [0:2:-1] "
            "because step is negative and stop index is not less than start index for dimension 0",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[::0],
            ValueError,
            "Tensor of shape (3,) cannot be indexed with [::0] "
            "because step is zero for dimension 0",
        ),
    ],
)
def test_invalid_constant_indexing(
    default_compilation_configuration,
    input_value,
    function_with_indexing,
    expected_error_type,
    expected_error_message,
):
    """Test compile_numpy_function_into_op_graph with invalid constant indexing"""

    with pytest.raises(expected_error_type):
        try:
            inputset = [
                (
                    np.random.randint(
                        input_value.dtype.min_value(),
                        input_value.dtype.max_value() + 1,
                        size=input_value.shape,
                    ),
                )
                for _ in range(10)
            ]
            compile_numpy_function_into_op_graph_and_measure_bounds(
                function_with_indexing,
                {"x": input_value},
                inputset,
                default_compilation_configuration,
            )
        except Exception as error:
            assert str(error) == expected_error_message
            raise


@pytest.mark.parametrize(
    "input_value,function_with_indexing,output_value",
    [
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[np.uint32(0)],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[slice(np.uint32(2), np.int32(0), np.int8(-1))],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[np.array(0)],
            EncryptedScalar(UnsignedInteger(1)),
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[slice(np.array(2), np.array(0), np.array(-1))],
            EncryptedTensor(UnsignedInteger(1), shape=(2,)),
        ),
    ],
)
def test_constant_indexing_with_numpy_integers(
    default_compilation_configuration,
    input_value,
    function_with_indexing,
    output_value,
):
    """Test compile_numpy_function_into_op_graph with constant indexing with numpy integers"""

    inputset = [
        np.random.randint(
            input_value.dtype.min_value(),
            input_value.dtype.max_value() + 1,
            size=input_value.shape,
        )
        for _ in range(10)
    ]

    op_graph = compile_numpy_function_into_op_graph_and_measure_bounds(
        function_with_indexing,
        {"x": input_value},
        inputset,
        default_compilation_configuration,
    )

    assert len(op_graph.output_nodes) == 1
    output_node = op_graph.output_nodes[0]

    assert len(output_node.outputs) == 1
    assert output_value == output_node.outputs[0]


@pytest.mark.parametrize(
    "input_value,function_with_indexing,expected_error_type,expected_error_message",
    [
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[np.float32(1.5)],
            TypeError,
            "Only integers and integer slices can be used for indexing "
            "but you tried to use 1.5 for indexing",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[np.array(1.5)],
            TypeError,
            "Only integers and integer slices can be used for indexing "
            "but you tried to use 1.5 for indexing",
        ),
        pytest.param(
            EncryptedTensor(UnsignedInteger(1), shape=(3,)),
            lambda x: x[np.array([1, 2])],
            TypeError,
            "Only integers and integer slices can be used for indexing "
            "but you tried to use [1 2] for indexing",
        ),
    ],
)
def test_invalid_constant_indexing_with_numpy_values(
    default_compilation_configuration,
    input_value,
    function_with_indexing,
    expected_error_type,
    expected_error_message,
):
    """Test compile_numpy_function_into_op_graph with invalid constant indexing with numpy values"""

    with pytest.raises(expected_error_type):
        try:
            inputset = [
                (
                    np.random.randint(
                        input_value.dtype.min_value(),
                        input_value.dtype.max_value() + 1,
                        size=input_value.shape,
                    ),
                )
                for _ in range(10)
            ]
            compile_numpy_function_into_op_graph_and_measure_bounds(
                function_with_indexing,
                {"x": input_value},
                inputset,
                default_compilation_configuration,
            )
        except Exception as error:
            assert str(error) == expected_error_message
            raise


@pytest.mark.parametrize(
    "function,parameters,inputset,test_input,expected_output",
    [
        pytest.param(
            lambda x: x[0],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [np.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            ([4, 2, 6],),
            4,
        ),
        pytest.param(
            lambda x: x[-1],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [np.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            ([4, 2, 6],),
            6,
        ),
        pytest.param(
            lambda x: x[:3],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4,)),
            },
            [np.random.randint(0, 2 ** 3, size=(4,)) for _ in range(10)],
            ([4, 2, 6, 1],),
            [4, 2, 6],
        ),
        pytest.param(
            lambda x: x[2:],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4,)),
            },
            [np.random.randint(0, 2 ** 3, size=(4,)) for _ in range(10)],
            ([4, 2, 6, 1],),
            [6, 1],
        ),
        pytest.param(
            lambda x: x[1:3],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4,)),
            },
            [np.random.randint(0, 2 ** 3, size=(4,)) for _ in range(10)],
            ([4, 2, 6, 1],),
            [2, 6],
        ),
        pytest.param(
            lambda x: x[::2],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4,)),
            },
            [np.random.randint(0, 2 ** 3, size=(4,)) for _ in range(10)],
            ([4, 2, 6, 1],),
            [4, 6],
        ),
        pytest.param(
            lambda x: x[::-1],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(4,)),
            },
            [np.random.randint(0, 2 ** 3, size=(4,)) for _ in range(10)],
            ([4, 2, 6, 1],),
            [1, 6, 2, 4],
        ),
        pytest.param(
            lambda x: x[1, 0],
            {
                "x": EncryptedTensor(UnsignedInteger(6), shape=(3, 2)),
            },
            [np.random.randint(0, 2 ** 6, size=(3, 2)) for _ in range(10)],
            ([[11, 12], [21, 22], [31, 32]],),
            21,
        ),
        pytest.param(
            lambda x: x[:, :],
            {
                "x": EncryptedTensor(UnsignedInteger(6), shape=(3, 2)),
            },
            [np.random.randint(0, 2 ** 6, size=(3, 2)) for _ in range(10)],
            ([[11, 12], [21, 22], [31, 32]],),
            [[11, 12], [21, 22], [31, 32]],
        ),
        pytest.param(
            lambda x: x[0, :],
            {
                "x": EncryptedTensor(UnsignedInteger(6), shape=(3, 2)),
            },
            [np.random.randint(0, 2 ** 6, size=(3, 2)) for _ in range(10)],
            ([[11, 12], [21, 22], [31, 32]],),
            [11, 12],
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            lambda x: x[:, 0],
            {
                "x": EncryptedTensor(UnsignedInteger(6), shape=(3, 2)),
            },
            [np.random.randint(0, 2 ** 6, size=(3, 2)) for _ in range(10)],
            ([[11, 12], [21, 22], [31, 32]],),
            [11, 21, 31],
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_constant_indexing_run_correctness(
    function,
    parameters,
    inputset,
    test_input,
    expected_output,
    default_compilation_configuration,
):
    """Test correctness of results when running a compiled function with tensor operators"""
    circuit = compile_numpy_function(
        function,
        parameters,
        inputset,
        default_compilation_configuration,
    )

    numpy_test_input = tuple(
        item if isinstance(item, int) else np.array(item, dtype=np.uint8) for item in test_input
    )

    output = circuit.run(*numpy_test_input)
    expected = np.array(expected_output, dtype=np.uint8)

    assert np.array_equal(
        output, expected
    ), f"""

Actual Output
=============
{output}

Expected Output
===============
{expected}

        """


@pytest.mark.parametrize(
    "function,parameters,inputset,match",
    [
        pytest.param(
            lambda x: x[0:1],
            {
                "x": EncryptedTensor(UnsignedInteger(3), shape=(3,)),
            },
            [np.random.randint(0, 2 ** 3, size=(3,)) for _ in range(10)],
            (
                "Indexing of EncryptedTensor<uint3, shape=(3,)> with [0:1] "
                "cannot be converted to MLIR yet"
            ),
        ),
    ],
)
def test_constant_indexing_failed_compilation(
    function,
    parameters,
    inputset,
    match,
    default_compilation_configuration,
):
    """Test compilation failures of compiled function with constant indexing"""

    with pytest.raises(RuntimeError) as excinfo:
        compile_numpy_function(
            function,
            parameters,
            inputset,
            default_compilation_configuration,
        )

    assert str(excinfo.value) == match, str(excinfo.value)
