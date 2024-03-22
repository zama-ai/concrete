"""
Tests of 'round_bit_pattern' extension.
"""

import pytest

from concrete import fhe


def test_dump_load_auto_rounder():
    """
    Test 'dump_dict' and 'load_dict' methods of AutoRounder.
    """

    rounder = fhe.AutoRounder(target_msbs=3)
    rounder.is_adjusted = True
    rounder.input_min = 10
    rounder.input_max = 20
    rounder.input_bit_width = 5
    rounder.lsbs_to_remove = 2

    dumped = rounder.dump_dict()
    assert dumped == {
        "target_msbs": 3,
        "is_adjusted": True,
        "input_min": 10,
        "input_max": 20,
        "input_bit_width": 5,
        "lsbs_to_remove": 2,
    }

    loaded = fhe.AutoRounder.load_dict(dumped)

    assert loaded.target_msbs == 3
    assert loaded.is_adjusted
    assert loaded.input_min == 10
    assert loaded.input_max == 20
    assert loaded.input_bit_width == 5
    assert loaded.lsbs_to_remove == 2


def test_bad_exactness():
    """
    Test for incorrect 'exactness' argument.
    """

    @fhe.compiler({"a": "encrypted"})
    def f(a):
        return fhe.round_bit_pattern(a, lsbs_to_remove=1, exactness=True)

    with pytest.raises(TypeError):
        f.compile([0])
