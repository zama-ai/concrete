"""
Tests of execution of static indexing operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "shape,function",
    [
        pytest.param(
            (3,),
            lambda x: x[0],
            id="x[0] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[1],
            id="x[1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[2],
            id="x[2] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[-1],
            id="x[-1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[-2],
            id="x[-2] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[-3],
            id="x[-3] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[0:1],
            id="x[0:1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[1:2],
            id="x[1:2] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[2:3],
            id="x[2:3] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[0:2],
            id="x[0:2] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[1:3],
            id="x[1:3] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[0:3],
            id="x[0:3] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[2:0:-1],
            id="x[2:0:-1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[::-1],
            id="x[::-1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[:-1],
            id="x[:-1] where x.shape == (3,)",
        ),
        pytest.param(
            (3,),
            lambda x: x[-2:],
            id="x[-2:] where x.shape == (3,)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[0, 0],
            id="x[0, 0] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[0, -1],
            id="x[0, -1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[-1, 0],
            id="x[-1, 0] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[-1, -1],
            id="x[-1, -1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[2, 1],
            id="x[2, 1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[0],
            id="x[0] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[:, 0],
            id="x[:, 0] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[-1],
            id="x[-1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[:, -1],
            id="x[:, -1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[1:3, 1:3],
            id="x[1:3, 1:3] where x.shape == (3, 4)",
        ),
        pytest.param(
            (3, 4),
            lambda x: x[::-1],
            id="x[::-1] where x.shape == (3, 4)",
        ),
        pytest.param(
            (10,),
            lambda x: x[slice(np.int64(8), np.int64(2), np.int64(-2))],
            id="x[8:2:-2] where x.shape == (10,)",
        ),
        pytest.param(
            (5,),
            lambda x: x[[3, 1, 2]],
            id="x[[3, 1, 2]] where x.shape == (5,)",
        ),
        pytest.param(
            (5,),
            lambda x: x[
                [
                    [3, 0],
                    [1, 2],
                ]
            ],
            id="x[[[3, 0], [1, 2]]] where x.shape == (5,)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[
                [0, 0, 4, 4],
                [0, 3, 0, 3],
            ],
            id="x[[0, 0, 4, 4], [0, 3, 0, 3]] where x.shape == (5, 4)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[
                0,
                [0, 3, 0, 3],
            ],
            id="x[0, [0, 3, 0, 3]] where x.shape == (5, 4)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[
                [0, 0, 4, 4],
                0,
            ],
            id="x[[0, 0, 4, 4], 0] where x.shape == (5, 4)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[
                [[0, 0], [4, 4]],
                [[0, 3], [0, 3]],
            ],
            id="x[[[0, 0], [4, 4]], [[0, 3], [0, 3]]] where x.shape == (5, 4)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[0, [[0, 3], [0, 3]]],
            id="x[0, [[0, 3], [0, 3]]] where x.shape == (5, 4)",
        ),
        pytest.param(
            (5, 4),
            lambda x: x[[[0, 3], [0, 3]], 0],
            id="x[[[0, 3], [0, 3]], 0] where x.shape == (5, 4)",
        ),
    ],
)
def test_static_indexing(shape, function, helpers):
    """
    Test static indexing.
    """

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [np.random.randint(0, 2**5, size=shape) for _ in range(100)]
    circuit = compiler.compile(inputset, configuration)

    sample = np.random.randint(0, 2**5, size=shape)
    helpers.check_execution(circuit, function, sample)


def test_bad_static_indexing(helpers):
    """
    Test static indexing with bad parameters.
    """

    configuration = helpers.configuration()

    # with float
    # ----------

    compiler = fhe.Compiler(lambda x: x[1.5], {"x": "encrypted"})

    inputset = [[5, 2, 4]]
    with pytest.raises(ValueError) as excinfo:
        compiler.compile(inputset, configuration)

    assert str(excinfo.value) == (
        "Tracer<output=EncryptedTensor<uint3, shape=(3,)>> cannot be indexed with 1.5"
    )

    # with bad slice
    # --------------

    compiler = fhe.Compiler(lambda x: x[slice(1.5, 2.5, None)], {"x": "encrypted"})

    inputset = [[5, 2, 4]]
    with pytest.raises(ValueError) as excinfo:
        compiler.compile(inputset, configuration)

    assert str(excinfo.value) == (
        "Tracer<output=EncryptedTensor<uint3, shape=(3,)>> cannot be indexed with 1.5:2.5"
    )

    # with multiple values
    # --------------------

    compiler = fhe.Compiler(lambda x: x[0, 1.5], {"x": "encrypted"})

    inputset = [[5, 2, 4]]
    with pytest.raises(ValueError) as excinfo:
        compiler.compile(inputset, configuration)

    assert str(excinfo.value) == (
        "Tracer<output=EncryptedTensor<uint3, shape=(3,)>> cannot be indexed with 0, 1.5"
    )
