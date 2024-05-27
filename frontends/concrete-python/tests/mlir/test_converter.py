"""
Tests of `Converter` class.
"""

# pylint: disable=import-error,no-name-in-module

import numpy as np
import pytest
from concrete.compiler import CompilationContext

from concrete import fhe
from concrete.fhe.compilation.configuration import ParameterSelectionStrategy
from concrete.fhe.mlir import GraphConverter

from ..conftest import USE_MULTI_PRECISION

# pylint: enable=import-error,no-name-in-module


def assign(x, y):
    """
    Assign scalar `y` into vector `x`.
    """

    x[0] = y
    return x


@pytest.mark.parametrize(
    "function,encryption_statuses,inputset,expected_error,expected_message",
    [
        pytest.param(
            lambda x, y: x + y,
            {"x": "encrypted", "y": "encrypted"},
            [(0.0, 0), (7.0, 7), (0.0, 7), (7.0, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                  # EncryptedScalar<float64>        ∈ [0.0, 7.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integers are supported
%1 = y                  # EncryptedScalar<uint3>          ∈ [0, 7]
%2 = add(%0, %1)        # EncryptedScalar<float64>        ∈ [0.0, 14.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integers are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.conv(x, [[[3, 1, 0, 2]]]),
            {"x": "encrypted"},
            [np.ones(shape=(1, 1, 10), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                              # EncryptedTensor<uint1, shape=(1, 1, 10)>        ∈ [1, 1]
%1 = [[[3 1 0 2]]]                                                                  # ClearTensor<uint2, shape=(1, 1, 4)>             ∈ [0, 3]
%2 = conv1d(%0, %1, [0], pads=(0, 0), strides=(1,), dilations=(1,), group=1)        # EncryptedTensor<uint3, shape=(1, 1, 7)>         ∈ [6, 6]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 1-dimensional convolutions are not supported at the moment
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.conv(x, [[[[[1, 3], [4, 2]]]]]),
            {"x": "encrypted"},
            [np.ones(shape=(1, 1, 3, 4, 5), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                                    # EncryptedTensor<uint1, shape=(1, 1, 3, 4, 5)>        ∈ [1, 1]
%1 = [[[[[1 3]    [4 2]]]]]                                                                               # ClearTensor<uint3, shape=(1, 1, 1, 2, 2)>            ∈ [1, 4]
%2 = conv3d(%0, %1, [0], pads=(0, 0, 0, 0, 0, 0), strides=(1, 1, 1), dilations=(1, 1, 1), group=1)        # EncryptedTensor<uint4, shape=(1, 1, 3, 3, 4)>        ∈ [10, 10]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 3-dimensional convolutions are not supported at the moment
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.maxpool(x, kernel_shape=(3,)),
            {"x": "encrypted"},
            [np.ones(shape=(1, 1, 10), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                                   # EncryptedTensor<uint1, shape=(1, 1, 10)>        ∈ [1, 1]
%1 = maxpool1d(%0, kernel_shape=(3,), strides=(1,), pads=(0, 0), dilations=(1,), ceil_mode=False)        # EncryptedTensor<uint1, shape=(1, 1, 8)>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 1-dimensional maxpooling is not supported at the moment
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.maxpool(x, kernel_shape=(3, 1, 2)),
            {"x": "encrypted"},
            [np.ones(shape=(1, 1, 3, 4, 5), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                                                              # EncryptedTensor<uint1, shape=(1, 1, 3, 4, 5)>        ∈ [1, 1]
%1 = maxpool3d(%0, kernel_shape=(3, 1, 2), strides=(1, 1, 1), pads=(0, 0, 0, 0, 0, 0), dilations=(1, 1, 1), ceil_mode=False)        # EncryptedTensor<uint1, shape=(1, 1, 1, 4, 4)>        ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 3-dimensional maxpooling is not supported at the moment
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x + y,
            {"x": "clear", "y": "clear"},
            [(0, 0), (7, 7), (0, 7), (7, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                  # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is clear
%1 = y                  # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is clear
%2 = add(%0, %1)        # ClearScalar<uint4>        ∈ [0, 14]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear-clear additions are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x - y,
            {"x": "clear", "y": "clear"},
            [(0, 0), (7, 7), (0, 7), (7, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is clear
%1 = y                       # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is clear
%2 = subtract(%0, %1)        # ClearScalar<int4>         ∈ [-7, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear-clear subtractions are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x * y,
            {"x": "clear", "y": "clear"},
            [(0, 0), (7, 7), (0, 7), (7, 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is clear
%1 = y                       # ClearScalar<uint3>        ∈ [0, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is clear
%2 = multiply(%0, %1)        # ClearScalar<uint6>        ∈ [0, 49]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear-clear multiplications are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: np.dot(x, y),
            {"x": "clear", "y": "clear"},
            [([1, 2], [3, 4])],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                  # ClearTensor<uint2, shape=(2,)>        ∈ [1, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is clear
%1 = y                  # ClearTensor<uint3, shape=(2,)>        ∈ [3, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is clear
%2 = dot(%0, %1)        # ClearScalar<uint4>                    ∈ [11, 11]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear-clear dot products are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.broadcast_to(x, shape=(2, 2)),
            {"x": "clear"},
            [[1, 2], [3, 4]],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                     # ClearTensor<uint3, shape=(2,)>          ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ value is clear
%1 = broadcast_to(%0, shape=(2, 2))        # ClearTensor<uint3, shape=(2, 2)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear values cannot be broadcasted
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            assign,
            {"x": "clear", "y": "encrypted"},
            [([1, 2, 3], 0)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                   # ClearTensor<uint2, shape=(3,)>        ∈ [1, 3]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ tensor is clear
%1 = y                   # EncryptedScalar<uint1>                ∈ [0, 0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ assigned value is encrypted
%2 = (%0[0] = %1)        # ClearTensor<uint2, shape=(3,)>        ∈ [0, 3]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but encrypted values cannot be assigned to clear tensors
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x**2 + (x + 1_000_000),
            {"x": "encrypted"},
            [100_000],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                    # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 34-bit value is used as an input to a table lookup
                                                                              (note that it's assigned 34-bits during compilation because of its relation with other operations)
%1 = 2                    # ClearScalar<uint2>             ∈ [2, 2]
%2 = power(%0, %1)        # EncryptedScalar<uint34>        ∈ [10000000000, 10000000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit table lookups are supported
%3 = 1000000              # ClearScalar<uint20>            ∈ [1000000, 1000000]
%4 = add(%0, %3)          # EncryptedScalar<uint21>        ∈ [1100000, 1100000]
%5 = add(%2, %4)          # EncryptedScalar<uint34>        ∈ [10001100000, 10001100000]
return %5

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x & y,
            {"x": "encrypted", "y": "encrypted"},
            [(-2, 4)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                          # EncryptedScalar<int2>         ∈ [-2, -2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is signed
%1 = y                          # EncryptedScalar<uint3>        ∈ [4, 4]
%2 = bitwise_and(%0, %1)        # EncryptedScalar<uint3>        ∈ [4, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only unsigned-unsigned bitwise operations are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x & y,
            {"x": "encrypted", "y": "encrypted"},
            [(4, -2)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                          # EncryptedScalar<uint3>        ∈ [4, 4]
%1 = y                          # EncryptedScalar<int2>         ∈ [-2, -2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is signed
%2 = bitwise_and(%0, %1)        # EncryptedScalar<uint3>        ∈ [4, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only unsigned-unsigned bitwise operations are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {"x": "clear", "y": "clear"},
            [([1, 2], [3, 4])],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                            # ClearTensor<uint2, shape=(2,)>        ∈ [1, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ value is clear
%1 = y                            # ClearTensor<uint3, shape=(2,)>        ∈ [3, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ value is clear
%2 = concatenate((%0, %1))        # ClearTensor<uint3, shape=(4,)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear concatenation is not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.conv(x, [[[[2, 1], [0, 3]]]]),
            {"x": "clear"},
            [np.ones(shape=(1, 1, 10, 10), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                        # ClearTensor<uint1, shape=(1, 1, 10, 10)>          ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = [[[[2 1]   [0 3]]]]                                                                      # ClearTensor<uint2, shape=(1, 1, 2, 2)>            ∈ [0, 3]
%2 = conv2d(%0, %1, [0], pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), group=1)        # EncryptedTensor<uint3, shape=(1, 1, 9, 9)>        ∈ [6, 6]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear convolutions are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.conv(x, weight=y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    np.ones(shape=(1, 1, 10, 10), dtype=np.int64),
                    np.ones(shape=(1, 1, 2, 2), dtype=np.int64),
                )
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                        # EncryptedTensor<uint1, shape=(1, 1, 10, 10)>        ∈ [1, 1]
%1 = y                                                                                        # EncryptedTensor<uint1, shape=(1, 1, 2, 2)>          ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weight is encrypted
%2 = conv2d(%0, %1, [0], pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), group=1)        # EncryptedTensor<uint3, shape=(1, 1, 9, 9)>          ∈ [4, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but convolutions with encrypted weights are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.conv(x, weight=[[[[2, 1], [0, 3]]]], bias=y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    np.ones(shape=(1, 1, 10, 10), dtype=np.int64),
                    np.ones(shape=(1,), dtype=np.int64),
                )
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                       # EncryptedTensor<uint1, shape=(1, 1, 10, 10)>        ∈ [1, 1]
%1 = y                                                                                       # EncryptedTensor<uint1, shape=(1,)>                  ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ bias is encrypted
%2 = [[[[2 1]   [0 3]]]]                                                                     # ClearTensor<uint2, shape=(1, 1, 2, 2)>              ∈ [0, 3]
%3 = conv2d(%0, %2, %1, pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), group=1)        # EncryptedTensor<uint3, shape=(1, 1, 9, 9)>          ∈ [7, 7]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but convolutions with encrypted biases are not supported
return %3

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x @ y,
            {"x": "clear", "y": "clear"},
            [([[1, 2], [3, 4]], [[4, 3], [2, 1]])],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                     # ClearTensor<uint3, shape=(2, 2)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is clear
%1 = y                     # ClearTensor<uint3, shape=(2, 2)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is clear
%2 = matmul(%0, %1)        # ClearTensor<uint5, shape=(2, 2)>        ∈ [5, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear-clear matrix multiplications are not supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.maxpool(x, kernel_shape=(3, 2)),
            {"x": "clear"},
            [np.ones(shape=(1, 1, 10, 5), dtype=np.int64)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                                               # ClearTensor<uint1, shape=(1, 1, 10, 5)>        ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = maxpool2d(%0, kernel_shape=(3, 2), strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1), ceil_mode=False)        # ClearTensor<uint1, shape=(1, 1, 8, 4)>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear maxpooling is not supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x**2,
            {"x": "clear"},
            [3, 4, 5],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                    # ClearScalar<uint3>        ∈ [3, 5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this clear value is used as an input to a table lookup
%1 = 2                    # ClearScalar<uint2>        ∈ [2, 2]
%2 = power(%0, %1)        # ClearScalar<uint5>        ∈ [9, 25]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only encrypted table lookups are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: np.sum(x),
            {"x": "clear"},
            [[1, 2]],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x              # ClearTensor<uint2, shape=(2,)>        ∈ [1, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = sum(%0)        # ClearScalar<uint2>                    ∈ [3, 3]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear summation is not supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x << y,
            {"x": "encrypted", "y": "encrypted"},
            [(-2, 4)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                         # EncryptedScalar<int2>         ∈ [-2, -2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lhs is signed
%1 = y                         # EncryptedScalar<uint3>        ∈ [4, 4]
%2 = left_shift(%0, %1)        # EncryptedScalar<int6>         ∈ [-32, -32]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only unsigned-unsigned bitwise shifts are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x >> y,
            {"x": "encrypted", "y": "encrypted"},
            [(4, -2)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                          # EncryptedScalar<uint3>        ∈ [4, 4]
%1 = y                          # EncryptedScalar<int2>         ∈ [-2, -2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rhs is signed
%2 = right_shift(%0, %1)        # EncryptedScalar<uint1>        ∈ [0, 0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only unsigned-unsigned bitwise shifts are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: -x,
            {"x": "clear"},
            [10],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                   # ClearScalar<uint4>        ∈ [10, 10]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = negative(%0)        # ClearScalar<int5>         ∈ [-10, -10]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear negations are not supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.LookupTable([fhe.LookupTable([0, 1]), fhe.LookupTable([1, 0])])[x],
            {"x": "clear"},
            [[1, 1], [1, 0], [0, 1], [0, 0]],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                     # ClearTensor<uint1, shape=(2,)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this clear value is used as an input to a table lookup
%1 = tlu(%0, table=[[0, 1] [1, 0]])        # ClearTensor<uint1, shape=(2,)>        ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only encrypted table lookups are supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.round_bit_pattern(x, lsbs_to_remove=2),
            {"x": "clear"},
            [10, 20, 30],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                                                        # ClearScalar<uint5>        ∈ [10, 30]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = round_bit_pattern(%0, lsbs_to_remove=2, overflow_protection=True, exactness=None)        # ClearScalar<uint6>        ∈ [12, 32]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear round bit pattern is not supported
%2 = reinterpret(%1)                                                                          # ClearScalar<uint6>
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x | y,
            {"x": "encrypted", "y": "encrypted"},
            [(100_000, 300_000)],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                         # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to a bitwise operation
%1 = y                         # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a bitwise operation
%2 = bitwise_or(%0, %1)        # EncryptedScalar<uint19>        ∈ [366560, 366560]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit bitwise operations are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                         # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a bitwise operation
                                                                                   (note that it's assigned 19-bits during compilation because of its relation with other operations)
%1 = y                         # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a bitwise operation
%2 = bitwise_or(%0, %1)        # EncryptedScalar<uint19>        ∈ [366560, 366560]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit bitwise operations are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: x != y,
            {"x": "encrypted", "y": "encrypted"},
            [(300_000, 100_000)],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                        # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
%1 = y                        # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to a comparison operation
%2 = not_equal(%0, %1)        # EncryptedScalar<uint1>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit comparison operations are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                        # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
%1 = y                        # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
                                                                                  (note that it's assigned 19-bits during compilation because of its relation with other operations)
%2 = not_equal(%0, %1)        # EncryptedScalar<uint1>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit comparison operations are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: x >= y,
            {"x": "encrypted", "y": "encrypted"},
            [(300_000, 100_000)],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                            # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
%1 = y                            # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to a comparison operation
%2 = greater_equal(%0, %1)        # EncryptedScalar<uint1>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit comparison operations are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                            # EncryptedScalar<uint19>        ∈ [300000, 300000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
%1 = y                            # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to a comparison operation
                                                                                      (note that it's assigned 19-bits during compilation because of its relation with other operations)
%2 = greater_equal(%0, %1)        # EncryptedScalar<uint1>         ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit comparison operations are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: x << y,
            {"x": "encrypted", "y": "encrypted"},
            [(100_000, 20)],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                         # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 37-bit value is used as the operand of a shift operation
                                                                                   (note that it's assigned 37-bits during compilation because of its relation with other operations)
%1 = y                         # EncryptedScalar<uint5>         ∈ [20, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 5-bit value is used as the shift amount of a shift operation
%2 = left_shift(%0, %1)        # EncryptedScalar<uint37>        ∈ [104857600000, 104857600000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this shift operation resulted in 37-bits but only up to 16-bit shift operations are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                         # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 37-bit value is used as the operand of a shift operation
                                                                                   (note that it's assigned 37-bits during compilation because of its relation with other operations)
%1 = y                         # EncryptedScalar<uint5>         ∈ [20, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 37-bit value is used as the shift amount of a shift operation
                                                                           (note that it's assigned 37-bits during compilation because of its relation with other operations)
%2 = left_shift(%0, %1)        # EncryptedScalar<uint37>        ∈ [104857600000, 104857600000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this shift operation resulted in 37-bits but only up to 16-bit shift operations are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: x * y,
            {"x": "encrypted", "y": "encrypted"},
            [(100_000, 20)],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                       # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to an encrypted multiplication
                                                                                 (note that it's assigned 18-bits during compilation because of its relation with other operations)
%1 = y                       # EncryptedScalar<uint5>         ∈ [20, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to an encrypted multiplication
                                                                         (note that it's assigned 18-bits during compilation because of its relation with other operations)
%2 = multiply(%0, %1)        # EncryptedScalar<uint21>        ∈ [2000000, 2000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted multiplications are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                       # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 21-bit value is used as an operand to an encrypted multiplication
                                                                                 (note that it's assigned 21-bits during compilation because of its relation with other operations)
%1 = y                       # EncryptedScalar<uint5>         ∈ [20, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 21-bit value is used as an operand to an encrypted multiplication
                                                                         (note that it's assigned 21-bits during compilation because of its relation with other operations)
%2 = multiply(%0, %1)        # EncryptedScalar<uint21>        ∈ [2000000, 2000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted multiplications are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: np.dot(x, y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    [100_000, 200_000],
                    [200_000, 100_000],
                )
            ],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                  # EncryptedTensor<uint18, shape=(2,)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to an encrypted dot products
                                                                                        (note that it's assigned 19-bits during compilation because of its relation with other operations)
%1 = y                  # EncryptedTensor<uint18, shape=(2,)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to an encrypted dot products
                                                                                        (note that it's assigned 19-bits during compilation because of its relation with other operations)
%2 = dot(%0, %1)        # EncryptedScalar<uint36>                    ∈ [40000000000, 40000000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted dot products are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                  # EncryptedTensor<uint18, shape=(2,)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 36-bit value is used as an operand to an encrypted dot products
                                                                                        (note that it's assigned 36-bits during compilation because of its relation with other operations)
%1 = y                  # EncryptedTensor<uint18, shape=(2,)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 36-bit value is used as an operand to an encrypted dot products
                                                                                        (note that it's assigned 36-bits during compilation because of its relation with other operations)
%2 = dot(%0, %1)        # EncryptedScalar<uint36>                    ∈ [40000000000, 40000000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted dot products are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: x @ y,
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    [
                        [100_000, 200_000],
                        [200_000, 100_000],
                    ],
                    [
                        [100_000, 200_000],
                        [200_000, 100_000],
                    ],
                )
            ],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                     # EncryptedTensor<uint18, shape=(2, 2)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to an encrypted matrix multiplication
                                                                                             (note that it's assigned 19-bits during compilation because of its relation with other operations)
%1 = y                     # EncryptedTensor<uint18, shape=(2, 2)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 19-bit value is used as an operand to an encrypted matrix multiplication
                                                                                             (note that it's assigned 19-bits during compilation because of its relation with other operations)
%2 = matmul(%0, %1)        # EncryptedTensor<uint36, shape=(2, 2)>        ∈ [40000000000, 50000000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted matrix multiplications are supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                     # EncryptedTensor<uint18, shape=(2, 2)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 36-bit value is used as an operand to an encrypted matrix multiplication
                                                                                             (note that it's assigned 36-bits during compilation because of its relation with other operations)
%1 = y                     # EncryptedTensor<uint18, shape=(2, 2)>        ∈ [100000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 36-bit value is used as an operand to an encrypted matrix multiplication
                                                                                             (note that it's assigned 36-bits during compilation because of its relation with other operations)
%2 = matmul(%0, %1)        # EncryptedTensor<uint36, shape=(2, 2)>        ∈ [40000000000, 50000000000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted matrix multiplications are supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: np.dot(x, y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    [50_000, 0],
                    [1, 10_000],
                )
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                  # EncryptedTensor<uint16, shape=(2,)>        ∈ [0, 50000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to an encrypted dot products
                                                                                  (note that it's assigned 17-bits during compilation because of its relation with other operations)
%1 = y                  # EncryptedTensor<uint14, shape=(2,)>        ∈ [1, 10000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to an encrypted dot products
                                                                                  (note that it's assigned 17-bits during compilation because of its relation with other operations)
%2 = dot(%0, %1)        # EncryptedScalar<uint16>                    ∈ [50000, 50000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted dot products are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x @ y,
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    [
                        [50_000, 3],
                        [3, 10_000],
                    ],
                    [
                        [1, 0],
                        [0, 1],
                    ],
                )
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                     # EncryptedTensor<uint16, shape=(2, 2)>        ∈ [3, 50000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to an encrypted matrix multiplication
                                                                                       (note that it's assigned 17-bits during compilation because of its relation with other operations)
%1 = y                     # EncryptedTensor<uint1, shape=(2, 2)>         ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to an encrypted matrix multiplication
                                                                                   (note that it's assigned 17-bits during compilation because of its relation with other operations)
%2 = matmul(%0, %1)        # EncryptedTensor<uint16, shape=(2, 2)>        ∈ [3, 50000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit encrypted matrix multiplications are supported
return %2

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: y[x],
            {"x": "encrypted", "y": "clear"},
            [
                (
                    1,
                    [1, 2, 3, 4],
                )
            ],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                          # EncryptedScalar<uint1>                ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table lookup input is 1-bits
%1 = y                          # ClearTensor<uint3, shape=(4,)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table has the shape (4,)
%2 = dynamic_tlu(%0, %1)        # EncryptedScalar<uint2>                ∈ [2, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ so table cannot be looked up with this input
                                                                                 table shape should have been (2,)
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                          # EncryptedScalar<uint1>                ∈ [1, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table lookup input is 3-bits
                                                                                 (note that it's assigned 3-bits during compilation because of its relation with other operations)
%1 = y                          # ClearTensor<uint3, shape=(4,)>        ∈ [1, 4]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table has the shape (4,)
%2 = dynamic_tlu(%0, %1)        # EncryptedScalar<uint2>                ∈ [2, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ so table cannot be looked up with this input
                                                                                 table shape should have been (8,)
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y, z: fhe.multivariate(lambda x, y, z: x + y // z)(x, y, z),
            {"x": "encrypted", "y": "encrypted", "z": "encrypted"},
            [
                (
                    100_000,
                    200_000,
                    10,
                )
            ],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                           # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is one of the inputs
%1 = y                           # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is one of the inputs
%2 = z                           # EncryptedScalar<uint4>         ∈ [10, 10]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 4-bit value is one of the inputs
%3 = <lambda>(%0, %1, %2)        # EncryptedScalar<uint17>        ∈ [120000, 120000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ which means the inputs would be packed to 39-bits for the table lookup
                                                                                     but only up to 16-bit table lookups are supported
return %3

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: np.minimum(x, y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    200_000,
                    100_000,
                )
            ],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                      # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a minimum operation
%1 = y                      # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to a minimum operation
%2 = minimum(%0, %1)        # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit minimum operation is supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                      # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a minimum operation
%1 = y                      # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a minimum operation
                                                                                (note that it's assigned 18-bits during compilation because of its relation with other operations)
%2 = minimum(%0, %1)        # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit minimum operation is supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x, y: np.maximum(x, y),
            {"x": "encrypted", "y": "encrypted"},
            [
                (
                    200_000,
                    100_000,
                )
            ],
            RuntimeError,
            (
                """

Function you are trying to compile cannot be compiled

%0 = x                      # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a maximum operation
%1 = y                      # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an operand to a maximum operation
%2 = maximum(%0, %1)        # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit maximum operation is supported
return %2

                """  # noqa: E501
                if USE_MULTI_PRECISION
                else """

Function you are trying to compile cannot be compiled

%0 = x                      # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a maximum operation
%1 = y                      # EncryptedScalar<uint17>        ∈ [100000, 100000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 18-bit value is used as an operand to a maximum operation
                                                                                (note that it's assigned 18-bits during compilation because of its relation with other operations)
%2 = maximum(%0, %1)        # EncryptedScalar<uint18>        ∈ [200000, 200000]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but only up to 16-bit maximum operation is supported
return %2

                """  # noqa: E501
            ),
        ),
        pytest.param(
            lambda x: fhe.truncate_bit_pattern(x, lsbs_to_remove=2),
            {"x": "clear"},
            [10, 20, 30],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                                                 # ClearScalar<uint5>        ∈ [10, 30]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = truncate_bit_pattern(%0, lsbs_to_remove=2)        # ClearScalar<uint5>        ∈ [8, 28]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear truncate bit pattern is not supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.bits(x)[0],
            {"x": "clear"},
            [10, 20, 30],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                  # ClearScalar<uint5>        ∈ [10, 30]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ operand is clear
%1 = bits(%0)[0]        # ClearScalar<uint1>        ∈ [0, 0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but clear bit extraction is not supported
return %1

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y, z: np.where(x, y, z),
            {"x": "encrypted", "y": "encrypted", "z": "encrypted"},
            [(10, 2, 3), (20, 1, 5)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # EncryptedScalar<uint5>        ∈ [10, 20]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ condition is not uint1
%1 = y                       # EncryptedScalar<uint2>        ∈ [1, 2]
%2 = z                       # EncryptedScalar<uint3>        ∈ [3, 5]
%3 = %1 if %0 else %2        # EncryptedScalar<uint2>        ∈ [1, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but it needs to be for where operation
return %3

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y, z: np.where(x, y, z),
            {"x": "encrypted", "y": "clear", "z": "encrypted"},
            [(1, 2, 3), (0, 1, 5)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # EncryptedScalar<uint1>        ∈ [0, 1]
%1 = y                       # ClearScalar<uint2>            ∈ [1, 2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ outcome of true condition is not encrypted
%2 = z                       # EncryptedScalar<uint3>        ∈ [3, 5]
%3 = %1 if %0 else %2        # EncryptedScalar<uint3>        ∈ [2, 5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but it needs to be for where operation
return %3

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y, z: np.where(x, y, z),
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
            [(1, 2, 3), (0, 1, 5)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # EncryptedScalar<uint1>        ∈ [0, 1]
%1 = y                       # EncryptedScalar<uint2>        ∈ [1, 2]
%2 = z                       # ClearScalar<uint3>            ∈ [3, 5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ outcome of false condition is not encrypted
%3 = %1 if %0 else %2        # EncryptedScalar<uint3>        ∈ [2, 5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but it needs to be for where operation
return %3

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y, z: np.where(x, y, z),
            {"x": "clear", "y": "encrypted", "z": "encrypted"},
            [(1, 2, 3), (0, 1, 5)],
            RuntimeError,
            """

Function you are trying to compile cannot be compiled

%0 = x                       # ClearScalar<uint1>            ∈ [0, 1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ condition is not encrypted
%1 = y                       # EncryptedScalar<uint2>        ∈ [1, 2]
%2 = z                       # EncryptedScalar<uint3>        ∈ [3, 5]
%3 = %1 if %0 else %2        # EncryptedScalar<uint3>        ∈ [2, 5]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ but it needs to be for where operation
return %3

            """,  # noqa: E501
        ),
    ],
)
def test_converter_bad_convert(
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
    compiler = fhe.Compiler(function, encryption_statuses)

    with pytest.raises(expected_error) as excinfo:
        compiler.compile(inputset, configuration)

    helpers.check_str(expected_message, str(excinfo.value))


@pytest.mark.parametrize(
    "function,parameters,configuration_overrides,expected_mlir",
    [
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 7], "status": "encrypted"},
                "y": {"range": [0, 7], "status": "encrypted"},
            },
            {},
            """

module {
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<6> {
    %0 = "FHE.mul_eint"(%arg0, %arg1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<6>
    return %0 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
            },
            {},
            """

module {
  func.func @main(%arg0: tensor<3x2x!FHE.eint<4>>, %arg1: tensor<3x2x!FHE.eint<4>>) -> tensor<3x2x!FHE.eint<6>> {
    %0 = "FHELinalg.mul_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<4>>, tensor<3x2x!FHE.eint<4>>) -> tensor<3x2x!FHE.eint<6>>
    return %0 : tensor<3x2x!FHE.eint<6>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.hint(np.dot(x, y), bit_width=7),
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (2,)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (2,)},
            },
            {},
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<4>>, %arg1: tensor<2x!FHE.eint<4>>) -> !FHE.eint<7> {
    %0 = "FHELinalg.dot_eint_eint"(%arg0, %arg1) : (tensor<2x!FHE.eint<4>>, tensor<2x!FHE.eint<4>>) -> !FHE.eint<7>
    return %0 : !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x @ y,
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (2, 4)},
            },
            {},
            """

module {
  func.func @main(%arg0: tensor<3x2x!FHE.eint<4>>, %arg1: tensor<2x4x!FHE.eint<4>>) -> tensor<3x4x!FHE.eint<7>> {
    %0 = "FHELinalg.matmul_eint_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<4>>, tensor<2x4x!FHE.eint<4>>) -> tensor<3x4x!FHE.eint<7>>
    return %0 : tensor<3x4x!FHE.eint<7>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.hint(np.sum((x - y) == 0), can_store=len(x)) == len(x),
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (10,)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (10,)},
            },
            {},
            """

module {
  func.func @main(%arg0: tensor<10x!FHE.eint<4>>, %arg1: tensor<10x!FHE.eint<4>>) -> !FHE.eint<1> {
    %0 = "FHELinalg.to_signed"(%arg0) : (tensor<10x!FHE.eint<4>>) -> tensor<10x!FHE.esint<4>>
    %1 = "FHELinalg.to_signed"(%arg1) : (tensor<10x!FHE.eint<4>>) -> tensor<10x!FHE.esint<4>>
    %2 = "FHELinalg.sub_eint"(%0, %1) : (tensor<10x!FHE.esint<4>>, tensor<10x!FHE.esint<4>>) -> tensor<10x!FHE.esint<4>>
    %c0_i2 = arith.constant 0 : i2
    %cst = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>
    %3 = "FHELinalg.apply_lookup_table"(%2, %cst) : (tensor<10x!FHE.esint<4>>, tensor<16xi64>) -> tensor<10x!FHE.eint<4>>
    %4 = "FHELinalg.sum"(%3) {axes = [], keep_dims = false} : (tensor<10x!FHE.eint<4>>) -> !FHE.eint<4>
    %c10_i5 = arith.constant 10 : i5
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi64>
    %5 = "FHE.apply_lookup_table"(%4, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<1>
    return %5 : !FHE.eint<1>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.hint(assign(x**2, y), bit_width=6),
            {
                "x": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": ()},
            },
            {},
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<2>>, %arg1: !FHE.eint<6>) -> tensor<2x!FHE.eint<6>> {
    %c2_i3 = arith.constant 2 : i3
    %cst = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
    %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<6>>
    %from_elements = tensor.from_elements %arg1 : tensor<1x!FHE.eint<6>>
    %inserted_slice = tensor.insert_slice %from_elements into %0[0] [1] [1] : tensor<1x!FHE.eint<6>> into tensor<2x!FHE.eint<6>>
    return %inserted_slice : tensor<2x!FHE.eint<6>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x**2, x + 100),
            {
                "x": {"range": [12, 13], "status": "encrypted", "shape": ()},
            },
            {},
            """

module {
  func.func @main(%arg0: !FHE.eint<7>) -> (!FHE.eint<8>, !FHE.eint<7>) {
    %c2_i3 = arith.constant 2 : i3
    %c8_i8 = arith.constant 8 : i8
    %0 = "FHE.mul_eint_int"(%arg0, %c8_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<7>) -> !FHE.eint<4>
    %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]> : tensor<16xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<8>
    %c100_i8 = arith.constant 100 : i8
    %3 = "FHE.add_eint_int"(%arg0, %c100_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    return %2, %3 : !FHE.eint<8>, !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x**2, x + 100),
            {
                "x": {"range": [12, 13], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<7>) -> (!FHE.eint<8>, !FHE.eint<7>) {
    %c2_i3 = arith.constant 2 : i3
    %c12_i8 = arith.constant 12 : i8
    %0 = "FHE.sub_eint_int"(%arg0, %c12_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    %c64_i8 = arith.constant 64 : i8
    %1 = "FHE.mul_eint_int"(%0, %c64_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    %2 = "FHE.reinterpret_precision"(%1) : (!FHE.eint<7>) -> !FHE.eint<1>
    %cst = arith.constant dense<[144, 169]> : tensor<2xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.eint<1>, tensor<2xi64>) -> !FHE.eint<8>
    %c100_i8 = arith.constant 100 : i8
    %4 = "FHE.add_eint_int"(%arg0, %c100_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    return %3, %4 : !FHE.eint<8>, !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.univariate(lambda x: x // np.array([2, 3]))(x),
            {
                "x": {"range": [12, 15], "status": "encrypted", "shape": (2,)},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<4>>) -> tensor<2x!FHE.eint<3>> {
    %c12_i5 = arith.constant 12 : i5
    %from_elements = tensor.from_elements %c12_i5 : tensor<1xi5>
    %0 = "FHELinalg.sub_eint_int"(%arg0, %from_elements) : (tensor<2x!FHE.eint<4>>, tensor<1xi5>) -> tensor<2x!FHE.eint<4>>
    %c4_i5 = arith.constant 4 : i5
    %from_elements_0 = tensor.from_elements %c4_i5 : tensor<1xi5>
    %1 = "FHELinalg.mul_eint_int"(%0, %from_elements_0) : (tensor<2x!FHE.eint<4>>, tensor<1xi5>) -> tensor<2x!FHE.eint<4>>
    %2 = "FHELinalg.reinterpret_precision"(%1) : (tensor<2x!FHE.eint<4>>) -> tensor<2x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
    %cst_1 = arith.constant dense<[[6, 6, 7, 7], [4, 4, 4, 5]]> : tensor<2x4xi64>
    %3 = "FHELinalg.apply_mapped_lookup_table"(%2, %cst_1, %cst) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<3>>
    return %3 : tensor<2x!FHE.eint<3>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (fhe.hint(x, bit_width=5) ** 2, x + 100),
            {
                "x": {"range": [12, 13], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<7>) -> (!FHE.eint<8>, !FHE.eint<7>) {
    %c2_i3 = arith.constant 2 : i3
    %c4_i8 = arith.constant 4 : i8
    %0 = "FHE.mul_eint_int"(%arg0, %c4_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<7>) -> !FHE.eint<5>
    %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<8>
    %c100_i8 = arith.constant 100 : i8
    %3 = "FHE.add_eint_int"(%arg0, %c100_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    return %2, %3 : !FHE.eint<8>, !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.univariate(lambda x: x // np.array([2, 3]))(x),
            {
                "x": {"range": [12, 15], "status": "encrypted", "shape": (2,)},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<4>>) -> tensor<2x!FHE.eint<3>> {
    %c12_i5 = arith.constant 12 : i5
    %from_elements = tensor.from_elements %c12_i5 : tensor<1xi5>
    %0 = "FHELinalg.sub_eint_int"(%arg0, %from_elements) : (tensor<2x!FHE.eint<4>>, tensor<1xi5>) -> tensor<2x!FHE.eint<4>>
    %c4_i5 = arith.constant 4 : i5
    %from_elements_0 = tensor.from_elements %c4_i5 : tensor<1xi5>
    %1 = "FHELinalg.mul_eint_int"(%0, %from_elements_0) : (tensor<2x!FHE.eint<4>>, tensor<1xi5>) -> tensor<2x!FHE.eint<4>>
    %2 = "FHELinalg.reinterpret_precision"(%1) : (tensor<2x!FHE.eint<4>>) -> tensor<2x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
    %cst_1 = arith.constant dense<[[6, 6, 7, 7], [4, 4, 4, 5]]> : tensor<2x4xi64>
    %3 = "FHELinalg.apply_mapped_lookup_table"(%2, %cst_1, %cst) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<3>>
    return %3 : tensor<2x!FHE.eint<3>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: fhe.univariate(lambda x: x // np.array([2, 3]))(fhe.hint(x, bit_width=5)),
            {
                "x": {"range": [12, 15], "status": "encrypted", "shape": (2,)},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<5>>) -> tensor<2x!FHE.eint<3>> {
    %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
    %cst_0 = arith.constant dense<[[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]]> : tensor<2x32xi64>
    %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst_0, %cst) : (tensor<2x!FHE.eint<5>>, tensor<2x32xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<3>>
    return %0 : tensor<2x!FHE.eint<3>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x // 2, x + 100),
            {
                "x": {"range": [1, 63], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<8>) -> (!FHE.eint<5>, !FHE.eint<8>) {
    %c2_i3 = arith.constant 2 : i3
    %c4_i9 = arith.constant 4 : i9
    %0 = "FHE.mul_eint_int"(%arg0, %c4_i9) : (!FHE.eint<8>, i9) -> !FHE.eint<8>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<8>) -> !FHE.eint<6>
    %cst = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31]> : tensor<64xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<5>
    %c100_i8 = arith.constant 100 : i8
    %3 = "FHE.add_eint_int"(%arg0, %c100_i8) : (!FHE.eint<8>, i8) -> !FHE.eint<8>
    return %2, %3 : !FHE.eint<5>, !FHE.eint<8>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x**2, x + 100),
            {
                "x": {"range": [-13, -12], "status": "encrypted", "shape": ()},
            },
            {},
            """

module {
  func.func @main(%arg0: !FHE.esint<7>) -> (!FHE.eint<8>, !FHE.eint<7>) {
    %c2_i3 = arith.constant 2 : i3
    %c4_i8 = arith.constant 4 : i8
    %0 = "FHE.mul_eint_int"(%arg0, %c4_i8) : (!FHE.esint<7>, i8) -> !FHE.esint<7>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.esint<7>) -> !FHE.esint<5>
    %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<8>
    %c100_i8 = arith.constant 100 : i8
    %3 = "FHE.to_unsigned"(%arg0) : (!FHE.esint<7>) -> !FHE.eint<7>
    %4 = "FHE.add_eint_int"(%3, %c100_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    return %2, %4 : !FHE.eint<8>, !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x**2, x + 100),
            {
                "x": {"range": [-13, -12], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.esint<7>) -> (!FHE.eint<8>, !FHE.eint<7>) {
    %c2_i3 = arith.constant 2 : i3
    %c13_i8 = arith.constant 13 : i8
    %0 = "FHE.add_eint_int"(%arg0, %c13_i8) : (!FHE.esint<7>, i8) -> !FHE.esint<7>
    %c64_i8 = arith.constant 64 : i8
    %1 = "FHE.mul_eint_int"(%0, %c64_i8) : (!FHE.esint<7>, i8) -> !FHE.esint<7>
    %2 = "FHE.reinterpret_precision"(%1) : (!FHE.esint<7>) -> !FHE.esint<1>
    %cst = arith.constant dense<[169, 144]> : tensor<2xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.esint<1>, tensor<2xi64>) -> !FHE.eint<8>
    %c100_i8 = arith.constant 100 : i8
    %4 = "FHE.to_unsigned"(%arg0) : (!FHE.esint<7>) -> !FHE.eint<7>
    %5 = "FHE.add_eint_int"(%4, %c100_i8) : (!FHE.eint<7>, i8) -> !FHE.eint<7>
    return %3, %5 : !FHE.eint<8>, !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: (x // 2, x + 100),
            {
                "x": {"range": [-32, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_measured_bounds": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.esint<8>) -> (!FHE.esint<5>, !FHE.eint<8>) {
    %c2_i3 = arith.constant 2 : i3
    %c4_i9 = arith.constant 4 : i9
    %0 = "FHE.mul_eint_int"(%arg0, %c4_i9) : (!FHE.esint<8>, i9) -> !FHE.esint<8>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.esint<8>) -> !FHE.esint<6>
    %cst = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, -16, -16, -15, -15, -14, -14, -13, -13, -12, -12, -11, -11, -10, -10, -9, -9, -8, -8, -7, -7, -6, -6, -5, -5, -4, -4, -3, -3, -2, -2, -1, -1]> : tensor<64xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.esint<6>, tensor<64xi64>) -> !FHE.esint<5>
    %c100_i8 = arith.constant 100 : i8
    %3 = "FHE.to_unsigned"(%arg0) : (!FHE.esint<8>) -> !FHE.eint<8>
    %4 = "FHE.add_eint_int"(%3, %c100_i8) : (!FHE.eint<8>, i8) -> !FHE.eint<8>
    return %2, %4 : !FHE.esint<5>, !FHE.eint<8>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": True,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %c2_i7 = arith.constant 2 : i7
    %0 = "FHE.mul_eint_int"(%arg0, %c2_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<6>) -> !FHE.eint<5>
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<6>
    %3 = "FHE.add_eint"(%arg0, %2) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %3 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": False,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21]> : tensor<64xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<6>
    %1 = "FHE.add_eint"(%arg0, %0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %1 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": 6,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %c2_i7 = arith.constant 2 : i7
    %0 = "FHE.mul_eint_int"(%arg0, %c2_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<6>) -> !FHE.eint<5>
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<6>
    %3 = "FHE.add_eint"(%arg0, %2) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %3 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": 5,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %c2_i7 = arith.constant 2 : i7
    %0 = "FHE.mul_eint_int"(%arg0, %c2_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
    %1 = "FHE.reinterpret_precision"(%0) : (!FHE.eint<6>) -> !FHE.eint<5>
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]> : tensor<32xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<6>
    %3 = "FHE.add_eint"(%arg0, %2) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %3 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": 4,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21]> : tensor<64xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<6>
    %1 = "FHE.add_eint"(%arg0, %0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %1 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x: x + (x // 3),
            {
                "x": {"range": [0, 31], "status": "encrypted", "shape": ()},
            },
            {
                "optimize_tlu_based_on_original_bit_width": 3,
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c3_i3 = arith.constant 3 : i3
    %cst = arith.constant dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21]> : tensor<64xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<6>
    %1 = "FHE.add_eint"(%arg0, %0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %1 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
    ],
)
def test_converter_convert_multi_precision(
    function,
    parameters,
    configuration_overrides,
    expected_mlir,
    helpers,
):
    """
    Test `convert` method of `Converter` with multi precision.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration().fork(single_precision=False, **configuration_overrides)

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    graph = compiler.trace(inputset, configuration)

    compilation_context = CompilationContext.new()
    mlir_context = compilation_context.mlir_context()

    module = GraphConverter(configuration).convert(graph, mlir_context)
    helpers.check_str(expected_mlir.strip(), str(module).strip())


@pytest.mark.parametrize(
    "function,parameters,expected_mlir",
    [
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 7], "status": "encrypted"},
                "y": {"range": [0, 7], "status": "encrypted"},
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<6>) -> !FHE.eint<6> {
    %0 = "FHE.mul_eint"(%arg0, %arg1) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    return %0 : !FHE.eint<6>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
            },
            """

module {
  func.func @main(%arg0: tensor<3x2x!FHE.eint<6>>, %arg1: tensor<3x2x!FHE.eint<6>>) -> tensor<3x2x!FHE.eint<6>> {
    %0 = "FHELinalg.mul_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<6>>, tensor<3x2x!FHE.eint<6>>) -> tensor<3x2x!FHE.eint<6>>
    return %0 : tensor<3x2x!FHE.eint<6>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.hint(np.dot(x, y), bit_width=7),
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (2,)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (2,)},
            },
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<7>>, %arg1: tensor<2x!FHE.eint<7>>) -> !FHE.eint<7> {
    %0 = "FHELinalg.dot_eint_eint"(%arg0, %arg1) : (tensor<2x!FHE.eint<7>>, tensor<2x!FHE.eint<7>>) -> !FHE.eint<7>
    return %0 : !FHE.eint<7>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: x @ y,
            {
                "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 2)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": (2, 4)},
            },
            """

module {
  func.func @main(%arg0: tensor<3x2x!FHE.eint<7>>, %arg1: tensor<2x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
    %0 = "FHELinalg.matmul_eint_eint"(%arg0, %arg1) : (tensor<3x2x!FHE.eint<7>>, tensor<2x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>
    return %0 : tensor<3x4x!FHE.eint<7>>
  }
}

            """,  # noqa: E501
        ),
        pytest.param(
            lambda x, y: fhe.hint(assign(x**2, y), bit_width=6),
            {
                "x": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
                "y": {"range": [0, 7], "status": "encrypted", "shape": ()},
            },
            """

module {
  func.func @main(%arg0: tensor<2x!FHE.eint<6>>, %arg1: !FHE.eint<6>) -> tensor<2x!FHE.eint<6>> {
    %c2_i3 = arith.constant 2 : i3
    %c16_i7 = arith.constant 16 : i7
    %from_elements = tensor.from_elements %c16_i7 : tensor<1xi7>
    %0 = "FHELinalg.mul_eint_int"(%arg0, %from_elements) : (tensor<2x!FHE.eint<6>>, tensor<1xi7>) -> tensor<2x!FHE.eint<6>>
    %1 = "FHELinalg.reinterpret_precision"(%0) : (tensor<2x!FHE.eint<6>>) -> tensor<2x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<6>>
    %from_elements_0 = tensor.from_elements %arg1 : tensor<1x!FHE.eint<6>>
    %inserted_slice = tensor.insert_slice %from_elements_0 into %2[0] [1] [1] : tensor<1x!FHE.eint<6>> into tensor<2x!FHE.eint<6>>
    return %inserted_slice : tensor<2x!FHE.eint<6>>
  }
}

            """,  # noqa: E501
        ),
    ],
)
def test_converter_convert_single_precision(function, parameters, expected_mlir, helpers):
    """
    Test `convert` method of `Converter` with multi precision.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration().fork(single_precision=True)

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    graph = compiler.trace(inputset, configuration)

    compilation_context = CompilationContext.new()
    mlir_context = compilation_context.mlir_context()

    module = GraphConverter(configuration).convert(graph, mlir_context)
    helpers.check_str(expected_mlir.strip(), str(module).strip())


@pytest.mark.parametrize(
    "function,parameters,expected_mlir",
    [
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 70], "status": "encrypted"},
                "y": {"range": [0, 7], "status": "encrypted"},
            },
            """

module {
  func.func @main(%arg0: !FHE.eint<9>, %arg1: !FHE.eint<9>) -> !FHE.eint<9> {
    %0 = "FHE.mul_eint"(%arg0, %arg1) : (!FHE.eint<9>, !FHE.eint<9>) -> !FHE.eint<9>
    return %0 : !FHE.eint<9>
  }
}

            """,  # noqa: E501
        ),
    ],
)
def test_converter_convert_composition(function, parameters, expected_mlir, helpers):
    """
    Test `convert` method of `Converter` with multi precision and composition activated.
    """
    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration().fork(
        parameter_selection_strategy=ParameterSelectionStrategy.MULTI,
        single_precision=False,
        composable=True,
    )

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    helpers.check_str(expected_mlir.strip(), circuit.mlir.strip())


@pytest.mark.parametrize(
    "function,parameters,expected_graph",
    [
        pytest.param(
            lambda x: (x**2) + 100,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
            },
            """

%0 = x                    # EncryptedScalar<uint4>        ∈ [0, 10]
%1 = 2                    # ClearScalar<uint3>            ∈ [2, 2]
%2 = power(%0, %1)        # EncryptedScalar<uint8>        ∈ [0, 100]
%3 = 100                  # ClearScalar<uint8>            ∈ [100, 100]
%4 = add(%2, %3)          # EncryptedScalar<uint8>        ∈ [100, 200]
return %4

            """,
        )
    ],
)
def test_converter_process_multi_precision(function, parameters, expected_graph, helpers):
    """
    Test `process` method of `Converter` with multi precision.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration().fork(single_precision=False)

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    graph = compiler.trace(inputset, configuration)

    GraphConverter(configuration).process({"main": graph})
    for node in graph.query_nodes():
        if "original_bit_width" in node.properties:
            del node.properties["original_bit_width"]

    helpers.check_str(expected_graph, graph.format())


@pytest.mark.parametrize(
    "function,parameters,expected_graph",
    [
        pytest.param(
            lambda x: (x**2) + 100,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
            },
            """

%0 = x                    # EncryptedScalar<uint8>        ∈ [0, 10]
%1 = 2                    # ClearScalar<uint3>            ∈ [2, 2]
%2 = power(%0, %1)        # EncryptedScalar<uint8>        ∈ [0, 100]
%3 = 100                  # ClearScalar<uint8>            ∈ [100, 100]
%4 = add(%2, %3)          # EncryptedScalar<uint8>        ∈ [100, 200]
return %4

            """,
        )
    ],
)
def test_converter_process_single_precision(function, parameters, expected_graph, helpers):
    """
    Test `process` method of `Converter` with single precision.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration().fork(single_precision=True)

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    graph = compiler.trace(inputset, configuration)

    GraphConverter(configuration).process({"main": graph})
    for node in graph.query_nodes():
        if "original_bit_width" in node.properties:
            del node.properties["original_bit_width"]

    helpers.check_str(expected_graph, graph.format())
