"""
Tests of 'truncate_bit_pattern' extension.
"""

from concrete import fhe


def test_dump_load_auto_truncator():
    """
    Test 'dump_dict' and 'load_dict' methods of AutoTruncator.
    """

    truncator = fhe.AutoTruncator(target_msbs=3)
    truncator.is_adjusted = True
    truncator.input_min = 10
    truncator.input_max = 20
    truncator.input_bit_width = 5
    truncator.lsbs_to_remove = 2

    dumped = truncator.dump_dict()
    assert dumped == {
        "target_msbs": 3,
        "is_adjusted": True,
        "input_min": 10,
        "input_max": 20,
        "input_bit_width": 5,
        "lsbs_to_remove": 2,
    }

    loaded = fhe.AutoTruncator.load_dict(dumped)

    assert loaded.target_msbs == 3
    assert loaded.is_adjusted
    assert loaded.input_min == 10
    assert loaded.input_max == 20
    assert loaded.input_bit_width == 5
    assert loaded.lsbs_to_remove == 2
