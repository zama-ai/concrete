"""
Tests of `GraphConverter` class.
"""

import numpy as np
import pytest

import concrete.numpy as cnp
import concrete.onnx as connx

# pylint: disable=line-too-long


def assign(x):
    """
    Simple assignment to a vector.
    """

    x[0] = 0
    return x


@pytest.mark.parametrize(
    "function,encryption_statuses,inputset,expected_error,expected_message",
    [
        pytest.param(
            lambda x, y: (x - y, x + y),
            {"x": "encrypted", "y": "clear"},
            [(0, 0), (7, 7), (0, 7), (7, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                       # EncryptedScalar<uint3>        ∈ [0, 7]
%1 = y                       # ClearScalar<uint3>            ∈ [0, 7]
%2 = subtract(%0, %1)        # EncryptedScalar<int4>         ∈ [-7, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only a single output is supported
%3 = add(%0, %1)             # EncryptedScalar<uint4>        ∈ [0, 14]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only a single output is supported
return (%2, %3)

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x,
            {"x": "clear"},
            range(-10, 10),
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x        # ClearScalar<int5>        ∈ [-10, 9]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted signed integer inputs are supported
return %0

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x * 1.5,
            {"x": "encrypted"},
            [2.5 * x for x in range(100)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                       # EncryptedScalar<float64>        ∈ [0.0, 247.5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer inputs are supported
%1 = 1.5                     # ClearScalar<float64>            ∈ [1.5, 1.5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%2 = multiply(%0, %1)        # EncryptedScalar<float64>        ∈ [0.0, 371.25]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.sin(x),
            {"x": "encrypted"},
            range(100),
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x              # EncryptedScalar<uint7>          ∈ [0, 99]
%1 = sin(%0)        # EncryptedScalar<float64>        ∈ [-0.9999902065507035, 0.9999118601072672]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {"x": "encrypted", "y": "clear"},
            [
                (
                    np.random.randint(0, 2**3, size=(3, 2)),
                    np.random.randint(0, 2**3, size=(3, 2)),
                )
                for _ in range(100)
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                            # EncryptedTensor<uint3, shape=(3, 2)>        ∈ [0, 7]
%1 = y                            # ClearTensor<uint3, shape=(3, 2)>            ∈ [0, 7]
%2 = concatenate((%0, %1))        # EncryptedTensor<uint3, shape=(6, 2)>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only all encrypted concatenate is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, w: connx.conv(x, w),
            {"x": "encrypted", "w": "encrypted"},
            [
                (
                    np.random.randint(0, 2, size=(1, 1, 4)),
                    np.random.randint(0, 2, size=(1, 1, 1)),
                )
                for _ in range(100)
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                                                              # EncryptedTensor<uint1, shape=(1, 1, 4)>        ∈ [0, 1]
%1 = w                                                                              # EncryptedTensor<uint1, shape=(1, 1, 1)>        ∈ [0, 1]
%2 = conv1d(%0, %1, [0], pads=(0, 0), strides=(1,), dilations=(1,), group=1)        # EncryptedTensor<uint1, shape=(1, 1, 4)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only conv1d with encrypted input and clear weight is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, w: connx.conv(x, w),
            {"x": "encrypted", "w": "encrypted"},
            [
                (
                    np.random.randint(0, 2, size=(1, 1, 4, 4)),
                    np.random.randint(0, 2, size=(1, 1, 1, 1)),
                )
                for _ in range(100)
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                                                                        # EncryptedTensor<uint1, shape=(1, 1, 4, 4)>        ∈ [0, 1]
%1 = w                                                                                        # EncryptedTensor<uint1, shape=(1, 1, 1, 1)>        ∈ [0, 1]
%2 = conv2d(%0, %1, [0], pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), group=1)        # EncryptedTensor<uint1, shape=(1, 1, 4, 4)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only conv2d with encrypted input and clear weight is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, w: connx.conv(x, w),
            {"x": "encrypted", "w": "encrypted"},
            [
                (
                    np.random.randint(0, 2, size=(1, 1, 4, 4, 4)),
                    np.random.randint(0, 2, size=(1, 1, 1, 1, 1)),
                )
                for _ in range(100)
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                                                                                    # EncryptedTensor<uint1, shape=(1, 1, 4, 4, 4)>        ∈ [0, 1]
%1 = w                                                                                                    # EncryptedTensor<uint1, shape=(1, 1, 1, 1, 1)>        ∈ [0, 1]
%2 = conv3d(%0, %1, [0], pads=(0, 0, 0, 0, 0, 0), strides=(1, 1, 1), dilations=(1, 1, 1), group=1)        # EncryptedTensor<uint1, shape=(1, 1, 4, 4, 4)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only conv3d with encrypted input and clear weight is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: np.dot(x, y),
            {"x": "encrypted", "y": "encrypted"},
            [([0], [0]), ([3], [3]), ([3], [0]), ([0], [3]), ([1], [1])],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                  # EncryptedTensor<uint2, shape=(1,)>        ∈ [0, 3]
%1 = y                  # EncryptedTensor<uint2, shape=(1,)>        ∈ [0, 3]
%2 = dot(%0, %1)        # EncryptedScalar<uint4>                    ∈ [0, 9]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only dot product between encrypted and clear is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x[0],
            {"x": "clear"},
            [[0, 1, 2, 3], [7, 6, 5, 4]],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x            # ClearTensor<uint3, shape=(4,)>        ∈ [0, 7]
%1 = %0[0]        # ClearScalar<uint3>                    ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted indexing supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x @ y,
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    np.random.randint(0, 2**1, size=(1, 1)),
                    np.random.randint(0, 2**1, size=(1, 1)),
                )
                for _ in range(100)
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                     # EncryptedTensor<uint1, shape=(1, 1)>        ∈ [0, 1]
%1 = y                     # EncryptedTensor<uint1, shape=(1, 1)>        ∈ [0, 1]
%2 = matmul(%0, %1)        # EncryptedTensor<uint1, shape=(1, 1)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only matrix multiplication between encrypted and clear is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x * y,
            {"x": "encrypted", "y": "encrypted"},
            [(0, 0), (7, 7), (0, 7), (7, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                       # EncryptedScalar<uint3>        ∈ [0, 7]
%1 = y                       # EncryptedScalar<uint3>        ∈ [0, 7]
%2 = multiply(%0, %1)        # EncryptedScalar<uint6>        ∈ [0, 49]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only multiplication between encrypted and clear is supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: -x,
            {"x": "clear"},
            [0, 7],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                   # ClearScalar<uint3>        ∈ [0, 7]
%1 = negative(%0)        # ClearScalar<int4>         ∈ [-7, 0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted negation is supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x.reshape((3, 2)),
            {"x": "clear"},
            [np.random.randint(0, 2**3, size=(2, 3)) for _ in range(100)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                   # ClearTensor<uint3, shape=(2, 3)>        ∈ [0, 7]
%1 = reshape(%0, newshape=(3, 2))        # ClearTensor<uint3, shape=(3, 2)>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted reshape is supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.sum(x),
            {"x": "clear"},
            [np.random.randint(0, 2, size=(1,)) for _ in range(100)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x              # ClearTensor<uint1, shape=(1,)>        ∈ [0, 1]
%1 = sum(%0)        # ClearScalar<uint1>                    ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted sum is supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.maximum(x, np.array([3])),
            {"x": "clear"},
            [[0], [1]],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                      # ClearTensor<uint1, shape=(1,)>        ∈ [0, 1]
%1 = [3]                    # ClearTensor<uint2, shape=(1,)>        ∈ [3, 3]
%2 = maximum(%0, %1)        # ClearTensor<uint2, shape=(1,)>        ∈ [3, 3]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ one of the operands must be encrypted
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.transpose(x),
            {"x": "clear"},
            [np.random.randint(0, 2, size=(3, 2)) for _ in range(10)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                    # ClearTensor<uint1, shape=(3, 2)>        ∈ [0, 1]
%1 = transpose(%0)        # ClearTensor<uint1, shape=(2, 3)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted transpose is supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.broadcast_to(x, shape=(3, 2)),
            {"x": "clear"},
            [np.random.randint(0, 2, size=(2,)) for _ in range(10)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                     # ClearTensor<uint1, shape=(2,)>          ∈ [0, 1]
%1 = broadcast_to(%0, shape=(3, 2))        # ClearTensor<uint1, shape=(3, 2)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted broadcasting is supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            assign,
            {"x": "clear"},
            [np.random.randint(0, 2, size=(3,)) for _ in range(10)],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR

%0 = x                   # ClearTensor<uint1, shape=(3,)>        ∈ [0, 1]
%1 = 0                   # ClearScalar<uint1>                    ∈ [0, 0]
%2 = (%0[0] = %1)        # ClearTensor<uint1, shape=(3,)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only assignment to encrypted tensors are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.abs(10 * np.sin(x + 300)).astype(np.int64),
            {"x": "encrypted"},
            [200000],
            RuntimeError,
            """

Function you are trying to compile cannot be converted to MLIR:

%0 = x                   # EncryptedScalar<uint18>        ∈ [200000, 200000]
%1 = 300                 # ClearScalar<uint9>             ∈ [300, 300]
%2 = add(%0, %1)         # EncryptedScalar<uint18>        ∈ [200300, 200300]
%3 = subgraph(%2)        # EncryptedScalar<uint4>         ∈ [9, 9]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table lookups are only supported on circuits with up to 16-bit integers
return %3

Subgraphs:

    %3 = subgraph(%2):

        %0 = input                         # EncryptedScalar<uint2>
        %1 = sin(%0)                       # EncryptedScalar<float64>
        %2 = 10                            # ClearScalar<uint4>
        %3 = multiply(%2, %1)              # EncryptedScalar<float64>
        %4 = absolute(%3)                  # EncryptedScalar<float64>
        %5 = astype(%4, dtype=int_)        # EncryptedScalar<uint1>
        return %5

            """,  # noqa: E501
        ),
    ],
)
def test_graph_converter_bad_convert(
    function,
    encryption_statuses,
    inputset,
    expected_error,
    expected_message,
    helpers,
):
    """
    Test unsupported graph conversion.
    """

    configuration = helpers.configuration()
    compiler = cnp.Compiler(function, encryption_statuses)

    with pytest.raises(expected_error) as excinfo:
        compiler.compile(inputset, configuration)

    helpers.check_str(expected_message, str(excinfo.value))


@pytest.mark.parametrize(
    "function,inputset,expected_mlir",
    [
        pytest.param(
            lambda x: 1 + cnp.LookupTable([4, 1, 2, 3])[x] + cnp.LookupTable([4, 1, 2, 3])[x + 1],
            range(3),
            """

module {
  func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
    %c1_i4 = arith.constant 1 : i4
    %cst = arith.constant dense<[4, 1, 2, 3, 3, 3, 3, 3]> : tensor<8xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<3>
    %1 = "FHE.add_eint_int"(%arg0, %c1_i4) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
    %2 = "FHE.add_eint_int"(%0, %c1_i4) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
    %3 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<3>
    %4 = "FHE.add_eint"(%2, %3) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
    return %4 : !FHE.eint<3>
  }
}

            """,  # noqa: E501
            # Notice that there is only a single 1 and a single table cst above
        ),
    ],
)
def test_constant_cache(function, inputset, expected_mlir, helpers):
    """
    Test caching MLIR constants.
    """

    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, {"x": "encrypted"})
    circuit = compiler.compile(inputset, configuration)

    helpers.check_str(expected_mlir, circuit.mlir)


# pylint: enable=line-too-long
