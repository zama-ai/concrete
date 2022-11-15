"""
Tests of execution of maxpool operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp
import concrete.onnx as connx


@pytest.mark.parametrize(
    "operation,sample_input,expected_output",
    [
        pytest.param(
            {"kernel_shape": (3,)},
            [1, 2, 2, 3, 2, 2, 2, 4, 1, 5, 2, 6],
            [2, 3, 3, 3, 2, 4, 4, 5, 5, 6],
        ),
        pytest.param(
            {"kernel_shape": (3,), "strides": (2,)},
            [1, 2, 2, 3, 2, 2, 2, 4, 1, 5, 2, 6, 7],
            [2, 3, 2, 4, 5, 7],
        ),
        pytest.param(
            {
                "kernel_shape": (2, 2),
            },
            [
                [3, 1, 2],
                [1, 1, 1],
                [2, 3, 4],
                [4, 1, 2],
            ],
            [
                [3, 2],
                [3, 4],
                [4, 4],
            ],
        ),
        pytest.param(
            {
                "kernel_shape": (2, 2),
                "strides": (2, 1),
            },
            [
                [3, 1, 2],
                [1, 1, 1],
                [2, 3, 4],
                [4, 1, 2],
            ],
            [
                [3, 2],
                [4, 4],
            ],
        ),
    ],
)
def test_maxpool(
    operation,
    sample_input,
    expected_output,
    helpers,
):
    """
    Test maxpool.
    """

    sample_input = np.expand_dims(np.array(sample_input), axis=(0, 1))
    expected_output = np.expand_dims(np.array(expected_output), axis=(0, 1))

    assert np.array_equal(connx.maxpool(sample_input, **operation), expected_output)

    @cnp.compiler({"x": "encrypted"})
    def function(x):
        return connx.maxpool(x, **operation)

    circuit = function.compile([sample_input], helpers.configuration(), virtual=True)
    helpers.check_execution(circuit, function, sample_input)


@pytest.mark.parametrize(
    "input_shape,operation,expected_error,expected_message",
    [
        pytest.param(
            (10, 10),
            {
                "kernel_shape": (),
            },
            ValueError,
            "Expected input to have at least 3 dimensions (N, C, D1, ...) but it only has 2",
        ),
        pytest.param(
            (1, 1, 5, 4, 3, 2),
            {
                "kernel_shape": (),
            },
            NotImplementedError,
            "4D maximum pooling is not supported yet",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": "",
            },
            TypeError,
            "Expected kernel_shape to be a tuple or a list but it's str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": ["0"],
            },
            TypeError,
            "Expected kernel_shape to consist of integers but it has an element of type str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (3,),
            },
            ValueError,
            "Expected kernel_shape to have 2 elements but it has 1",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "strides": "",
            },
            TypeError,
            "Expected strides to be a tuple or a list but it's str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "strides": ["0"],
            },
            TypeError,
            "Expected strides to consist of integers but it has an element of type str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "strides": (3,),
            },
            ValueError,
            "Expected strides to have 2 elements but it has 1",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "auto_pad": True,
            },
            TypeError,
            "Expected auto_pad to be of type str but it's bool",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "auto_pad": "YES_PLEASE",
            },
            ValueError,
            "Expected auto_pad to be one of NOTSET, SAME_LOWER, SAME_UPPER, VALID "
            "but it's YES_PLEASE",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "auto_pad": "VALID",
            },
            NotImplementedError,
            "Desired auto_pad of VALID is not supported yet",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "pads": "",
            },
            TypeError,
            "Expected pads to be a tuple or a list but it's str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "pads": ["0"],
            },
            TypeError,
            "Expected pads to consist of integers but it has an element of type str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "pads": (3,),
            },
            ValueError,
            "Expected pads to have 4 elements but it has 1",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "pads": (1, 1, 2, 2),
            },
            NotImplementedError,
            "Desired pads of (1, 1, 2, 2) is not supported yet because of uneven padding",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "dilations": "",
            },
            TypeError,
            "Expected dilations to be a tuple or a list but it's str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "dilations": ["0"],
            },
            TypeError,
            "Expected dilations to consist of integers but it has an element of type str",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "dilations": (3,),
            },
            ValueError,
            "Expected dilations to have 2 elements but it has 1",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "ceil_mode": None,
            },
            TypeError,
            "Expected ceil_mode to be of type int but it's NoneType",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "ceil_mode": 10,
            },
            ValueError,
            "Expected ceil_mode to be one of 0, 1 but it's 10",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "ceil_mode": 1,
            },
            NotImplementedError,
            "Desired ceil_mode of 1 is not supported yet",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "storage_order": None,
            },
            TypeError,
            "Expected storage_order to be of type int but it's NoneType",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "storage_order": 10,
            },
            ValueError,
            "Expected storage_order to be one of 0, 1 but it's 10",
        ),
        pytest.param(
            (1, 1, 5, 4),
            {
                "kernel_shape": (2, 3),
                "storage_order": 1,
            },
            NotImplementedError,
            "Desired storage_order of 1 is not supported yet",
        ),
    ],
)
def test_bad_maxpool(
    input_shape,
    operation,
    expected_error,
    expected_message,
    helpers,
):
    """
    Test maxpool with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        connx.maxpool(np.random.randint(0, 10, size=input_shape), **operation)

    helpers.check_str(expected_message, str(excinfo.value))


def test_bad_maxpool_special(helpers):
    """
    Test maxpool with bad parameters for special cases.
    """

    # without virtual
    # ---------------

    @cnp.compiler({"x": "encrypted"})
    def without_virtual(x):
        return connx.maxpool(x, kernel_shape=(4, 3))

    inputset = [np.random.randint(0, 10, size=(1, 1, 10, 10)) for i in range(100)]
    with pytest.raises(NotImplementedError) as excinfo:
        without_virtual.compile(inputset, helpers.configuration())

    helpers.check_str("MaxPool operation cannot be compiled yet", str(excinfo.value))

    # clear input
    # -----------

    @cnp.compiler({"x": "clear"})
    def clear_input(x):
        return connx.maxpool(x, kernel_shape=(4, 3, 2))

    inputset = [np.zeros((1, 1, 10, 10, 10), dtype=np.int64)]
    with pytest.raises(RuntimeError) as excinfo:
        clear_input.compile(inputset, helpers.configuration())

    helpers.check_str(
        # pylint: disable=line-too-long
        """

Function you are trying to compile cannot be converted to MLIR

%0 = x                                                                                                                            # ClearTensor<uint1, shape=(1, 1, 10, 10, 10)>        ∈ [0, 0]
%1 = maxpool(%0, kernel_shape=(4, 3, 2), strides=(1, 1, 1), pads=(0, 0, 0, 0, 0, 0), dilations=(1, 1, 1), ceil_mode=False)        # ClearTensor<uint1, shape=(1, 1, 7, 8, 9)>           ∈ [0, 0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only encrypted maxpool is supported
return %1

        """.strip(),  # noqa: E501
        # pylint: enable=line-too-long
        str(excinfo.value),
    )

    # badly typed ndarray input
    # -------------------------

    with pytest.raises(TypeError) as excinfo:
        connx.maxpool(np.array([{}, None]), ())

    helpers.check_str(
        # pylint: disable=line-too-long
        """

Expected input elements to be of type np.integer, np.floating, or np.bool_ but it's dtype[object_]

        """.strip(),  # noqa: E501
        # pylint: enable=line-too-long
        str(excinfo.value),
    )

    # badly typed input
    # -----------------

    with pytest.raises(TypeError) as excinfo:
        connx.maxpool("", ())

    helpers.check_str(
        # pylint: disable=line-too-long
        """

Expected input to be of type np.ndarray or Tracer but it's str

        """.strip(),  # noqa: E501
        # pylint: enable=line-too-long
        str(excinfo.value),
    )
