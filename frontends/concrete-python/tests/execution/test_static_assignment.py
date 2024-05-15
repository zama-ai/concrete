"""
Tests of execution of static assignment operation.
"""

import numpy as np
import pytest

from concrete import fhe


def assignment_case_0():
    """
    Assignment test case.
    """

    shape = (3,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[:] = value
        return x

    return shape, assign


def assignment_case_1():
    """
    Assignment test case.
    """

    shape = (3,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[0] = value
        return x

    return shape, assign


def assignment_case_2():
    """
    Assignment test case.
    """

    shape = (3,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[1] = value
        return x

    return shape, assign


def assignment_case_3():
    """
    Assignment test case.
    """

    shape = (3,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[2] = value
        return x

    return shape, assign


def assignment_case_4():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[0:3] = value
        return x

    return shape, assign


def assignment_case_5():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[1:4] = value
        return x

    return shape, assign


def assignment_case_6():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[1:4:2] = value
        return x

    return shape, assign


def assignment_case_7():
    """
    Assignment test case.
    """

    shape = (10,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[::2] = value
        return x

    return shape, assign


def assignment_case_8():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[2:0:-1] = value
        return x

    return shape, assign


def assignment_case_9():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[4:0:-2] = value
        return x

    return shape, assign


def assignment_case_10():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[1:4] = value
        return x

    return shape, assign


def assignment_case_11():
    """
    Assignment test case.
    """

    shape = (5,)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[4:1:-1] = value
        return x

    return shape, assign


def assignment_case_12():
    """
    Assignment test case.
    """

    shape = (10,)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[1:7:2] = value
        return x

    return shape, assign


def assignment_case_13():
    """
    Assignment test case.
    """

    shape = (10,)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[7:1:-2] = value
        return x

    return shape, assign


def assignment_case_14():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[0, 0] = value
        return x

    return shape, assign


def assignment_case_15():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[3, 1] = value
        return x

    return shape, assign


def assignment_case_16():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[0] = value
        return x

    return shape, assign


def assignment_case_17():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(4,))

    def assign(x):
        x[0] = value
        return x

    return shape, assign


def assignment_case_18():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(5,))

    def assign(x):
        x[:, 0] = value
        return x

    return shape, assign


def assignment_case_19():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(5,))

    def assign(x):
        x[:, 1] = value
        return x

    return shape, assign


def assignment_case_20():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[0:3, :] = value
        return x

    return shape, assign


def assignment_case_21():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(3, 4))

    def assign(x):
        x[0:3, :] = value
        return x

    return shape, assign


def assignment_case_22():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(4,))

    def assign(x):
        x[0:3, :] = value
        return x

    return shape, assign


def assignment_case_23():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[0:3, 1:4] = value
        return x

    return shape, assign


def assignment_case_24():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(3, 3))

    def assign(x):
        x[0:3, 1:4] = value
        return x

    return shape, assign


def assignment_case_25():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(3, 3))

    def assign(x):
        x[4:1:-1, 3:0:-1] = value
        return x

    return shape, assign


def assignment_case_26():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(3,))

    def assign(x):
        x[3:0:-1, 0] = value
        return x

    return shape, assign


def assignment_case_27():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=(2,))

    def assign(x):
        x[0, 1:3] = value
        return x

    return shape, assign


def assignment_case_28():
    """
    Assignment test case.
    """

    shape = (5, 4)
    value = np.random.randint(0, 2**7, size=())

    def assign(x):
        x[2:4, 1:3] = value
        return x

    return shape, assign


def assignment_case_29():
    """
    Assignment test case.
    """

    shape = (5,)
    value = -20

    def assign(x):
        x[0] = value
        return x

    return shape, assign


def assignment_case_30():
    """
    Assignment test case.
    """

    shape = (5,)
    index = [0, 2, 4]
    value = 10

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_31():
    """
    Assignment test case.
    """

    shape = (5,)
    index = [0, 2, 4]
    value = [10, 20, 30]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_32():
    """
    Assignment test case.
    """

    shape = (5,)
    index = [
        [3, 0],
        [1, 2],
    ]
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_33():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [0, 0, 4, 4],
        [0, 3, 0, 3],
    )
    value = [10, 20, 30, 40]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_34():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        0,
        [0, 3, 1, 2],
    )
    value = [10, 20, 30, 40]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_35():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [3, 3, 1, 2],
        0,
    )
    value = [10, 20, 30, 40]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_36():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [[0, 0], [4, 4]],
        [[0, 3], [3, 0]],
    )
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_37():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        0,
        [[1, 3], [2, 0]],
    )
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_38():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [[1, 3], [2, 4]],
        0,
    )
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_39():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [[1, 3], [2, 4]],
        [0, 2],
    )
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


def assignment_case_40():
    """
    Assignment test case.
    """

    shape = (5, 4)
    index = (
        [2, 4],
        [[1, 3], [2, 0]],
    )
    value = [
        [10, 20],
        [30, 40],
    ]

    def assign(x):
        x[index] = value
        return x

    return shape, assign


@pytest.mark.parametrize(
    "shape,function",
    [
        pytest.param(*assignment_case_0()),
        pytest.param(*assignment_case_1()),
        pytest.param(*assignment_case_2()),
        pytest.param(*assignment_case_3()),
        pytest.param(*assignment_case_4()),
        pytest.param(*assignment_case_5()),
        pytest.param(*assignment_case_6()),
        pytest.param(*assignment_case_7()),
        pytest.param(*assignment_case_8()),
        pytest.param(*assignment_case_9()),
        pytest.param(*assignment_case_10()),
        pytest.param(*assignment_case_11()),
        pytest.param(*assignment_case_12()),
        pytest.param(*assignment_case_13()),
        pytest.param(*assignment_case_14()),
        pytest.param(*assignment_case_15()),
        pytest.param(*assignment_case_16()),
        pytest.param(*assignment_case_17()),
        pytest.param(*assignment_case_18()),
        pytest.param(*assignment_case_19()),
        pytest.param(*assignment_case_20()),
        pytest.param(*assignment_case_21()),
        pytest.param(*assignment_case_22()),
        pytest.param(*assignment_case_23()),
        pytest.param(*assignment_case_24()),
        pytest.param(*assignment_case_25()),
        pytest.param(*assignment_case_26()),
        pytest.param(*assignment_case_27()),
        pytest.param(*assignment_case_28()),
        pytest.param(*assignment_case_29()),
        pytest.param(*assignment_case_30()),
        pytest.param(*assignment_case_31()),
        pytest.param(*assignment_case_32()),
        pytest.param(*assignment_case_33()),
        pytest.param(*assignment_case_34()),
        pytest.param(*assignment_case_35()),
        pytest.param(*assignment_case_36()),
        pytest.param(*assignment_case_37()),
        pytest.param(*assignment_case_38()),
        pytest.param(*assignment_case_39()),
        pytest.param(*assignment_case_40()),
    ],
)
def test_static_assignment(shape, function, helpers):
    """
    Test static assignment.
    """

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [np.random.randint(0, 2**7, size=shape) for _ in range(100)]
    circuit = compiler.compile(inputset, configuration)

    sample = np.random.randint(0, 2**7, size=shape)
    helpers.check_execution(circuit, function, sample)


def test_bad_static_assignment(helpers):
    """
    Test static assignment with bad parameters.
    """

    configuration = helpers.configuration()

    # with float
    # ----------

    def f(x):
        x[1.5] = 0
        return x

    compiler = fhe.Compiler(f, {"x": "encrypted"})

    inputset = [[1, 2, 3]]
    with pytest.raises(ValueError) as excinfo:
        compiler.compile(inputset, configuration)

    assert (
        str(excinfo.value)
        == "Tracer<output=EncryptedTensor<uint2, shape=(3,)>>[1.5] cannot be assigned 0"
    )

    # with bad slice
    # --------------

    def g(x):
        x[slice(1.5, 2.5, None)] = 0
        return x

    compiler = fhe.Compiler(g, {"x": "encrypted"})

    inputset = [[1, 2, 3]]
    with pytest.raises(ValueError) as excinfo:
        compiler.compile(inputset, configuration)

    assert (
        str(excinfo.value)
        == "Tracer<output=EncryptedTensor<uint2, shape=(3,)>>[1.5:2.5] cannot be assigned 0"
    )
