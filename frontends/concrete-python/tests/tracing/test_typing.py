"""
Test type annotations.
"""

import pytest

from concrete import fhe


def test_bad_tensor():
    """
    Test `tensor` type with bad parameters
    """

    # invalid dtype
    # -------------

    with pytest.raises(ValueError) as excinfo:

        def case1(x: fhe.tensor[int]):
            return x

        case1(None)

    assert str(excinfo.value) == (
        "First argument to tensor annotations should be "
        "an fhe data type (e.g., fhe.uint4) not int"
    )

    # no shape
    # --------

    with pytest.raises(ValueError) as excinfo:

        def case2(x: fhe.tensor[fhe.uint3]):
            return x

        case2(None)

    assert str(excinfo.value) == (
        "Tensor annotations should have a shape (e.g., fhe.tensor[fhe.uint4, 3, 2])"
    )

    # bad shape
    # ---------

    with pytest.raises(ValueError) as excinfo:

        def case3(x: fhe.tensor[fhe.uint3, 1.5]):
            return x

        case3(None)

    assert str(excinfo.value) == "Tensor annotation shape elements must be 'int'"
