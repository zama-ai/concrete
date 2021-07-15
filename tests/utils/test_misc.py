"""Test file for HDK's misc utils"""
import random

import hdk


def test_get_unique_id():
    """Test get_unique_id"""
    how_many_ids = random.randint(2, 100)
    generated_ids = [hdk.utils.get_unique_id() for __ in range(how_many_ids)]

    len_generated_ids = len(generated_ids)
    len_unique_ids = len(set(generated_ids))

    assert (
        len_generated_ids == len_unique_ids
    ), f"Expected to have uniques ids, generated {len_generated_ids}, only had {len_unique_ids}"
