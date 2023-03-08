"""
Test type annotations.
"""

import pytest

import concrete.numpy as cnp


def test_bad_tensor():
    """
    Test `tensor` type with bad parameters
    """

    # invalid dtype
    # -------------

    with pytest.raises(ValueError) as excinfo:

        def case1(x: cnp.tensor[int]):
            return x

        case1(None)

    assert str(excinfo.value) == (
        "First argument to tensor annotations should be a "
        "concrete-numpy data type (e.g., cnp.uint4) not int"
    )

    # no shape
    # --------

    with pytest.raises(ValueError) as excinfo:

        def case2(x: cnp.tensor[cnp.uint3]):
            return x

        case2(None)

    assert str(excinfo.value) == (
        "Tensor annotations should have a shape (e.g., cnp.tensor[cnp.uint4, 3, 2])"
    )

    # bad shape
    # ---------

    with pytest.raises(ValueError) as excinfo:

        def case3(x: cnp.tensor[cnp.uint3, 1.5]):
            return x

        case3(None)

    assert str(excinfo.value) == "Tensor annotation shape elements must be 'int'"
