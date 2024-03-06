"""
Tests of compilation utilities.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "inputs,ranges,shapes",
    [
        (
            (fhe.uint3,),  # type: ignore
            ([0, 2**3],),
            ((),),
        ),
        (
            (fhe.int3,),
            ([-(2**2), 2**2],),  # type: ignore
            ((),),
        ),
        (
            (fhe.tensor[fhe.uint3, 3, 2],),  # type: ignore
            ([0, 2**3],),
            ((3, 2),),
        ),
        (
            (fhe.tensor[fhe.int3, 3, 2],),  # type: ignore
            ([-(2**2), 2**2],),
            ((3, 2),),
        ),
        (
            (fhe.uint3, fhe.uint4),  # type: ignore
            ([0, 2**3], [0, 2**4]),
            ((), ()),
        ),
        (
            (fhe.int4, fhe.int3),  # type: ignore
            ([-(2**3), 2**3], [-(2**2), 2**2]),
            ((), ()),
        ),
        (
            (fhe.tensor[fhe.uint6, 3, 2], fhe.tensor[fhe.int6, 5]),  # type: ignore
            ([0, 2**6], [-(2**5), 2**5]),
            ((3, 2), (5,)),
        ),
        (
            (fhe.f32,),  # type: ignore
            ([0.0, 1.0],),
            ((),),
        ),
        (
            (fhe.tensor[fhe.f32, 3, 2],),  # type: ignore
            ([0.0, 1.0],),
            ((3, 2),),
        ),
        (
            (lambda _index: np.random.randint(10, 20),),
            ([10, 20],),
            ((),),
        ),
        (
            (lambda _index: np.random.randint(10, 20, size=(3, 2)),),
            ([10, 20],),
            ((3, 2),),
        ),
    ],
)
@pytest.mark.parametrize("size", [10, 100])
def test_inputset(inputs, ranges, shapes, size):
    """
    Test `inputset` utility.
    """

    inputset = fhe.inputset(*inputs, size=size)

    assert isinstance(inputset, list)
    assert len(inputset) == size

    for sample in inputset:
        assert isinstance(sample, tuple)
        assert len(sample) == len(inputs)
        for value, range_, shape in zip(sample, ranges, shapes):
            assert isinstance(value, (int, float, np.ndarray))
            if isinstance(value, (int, float)):
                assert shape == ()
                assert range_[0] <= value < range_[1]
            else:
                assert shape == value.shape
                assert value.min() >= range_[0]
                assert value.max() < range_[1]
