"""
Tests of the examples.
"""

from typing import Optional

import numpy as np
import pytest

from examples.key_value_database.static_size import StaticKeyValueDatabase
from examples.levenshtein_distance.levenshtein_distance import Alphabet, LevenshteinDistance


def test_static_kvdb(helpers):
    """
    Test static key-value database example.
    """

    configuration = helpers.configuration()

    def inspect(db: StaticKeyValueDatabase) -> np.ndarray:
        encrypted_state = db.inspect.run(db.state)
        clear_state = db.inspect.decrypt(encrypted_state)
        return clear_state  # type: ignore

    def insert(db: StaticKeyValueDatabase, key: int, value: int):
        encoded_key, encoded_value = db.encode_key(key), db.encode_value(value)
        _, encrypted_key, encoded_value = db.insert.encrypt(  # type: ignore
            None,
            encoded_key,
            encoded_value,
        )
        db.state = db.insert.run(db.state, encrypted_key, encoded_value)  # type: ignore

    def replace(db: StaticKeyValueDatabase, key: int, value: int):
        encoded_key, encoded_value = db.encode_key(key), db.encode_value(value)
        _, encrypted_key, encoded_value = db.replace.encrypt(  # type: ignore
            None,
            encoded_key,
            encoded_value,
        )
        db.state = db.replace.run(db.state, encrypted_key, encoded_value)  # type: ignore

    def query(db: StaticKeyValueDatabase, key: int) -> Optional[int]:
        encoded_key = db.encode_key(key)
        _, encrypted_key = db.query.encrypt(None, encoded_key)  # type: ignore
        encrypted_found, encrypted_value = db.query.run(db.state, encrypted_key)  # type: ignore

        found, value = db.query.decrypt(encrypted_found, encrypted_value)  # type: ignore
        if not found:
            return None

        return db.decode_value(value)  # type: ignore

    db = StaticKeyValueDatabase(
        number_of_entries=4,
        key_size=8,
        value_size=8,
        chunk_size=2,
        configuration=configuration,
    )
    db.keygen()

    db.initialize()
    assert np.array_equal(
        inspect(db),
        [
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, 3) is None

    insert(db, 3, 4)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 3] + [0, 0, 1, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, 3) == 4

    replace(db, 3, 1)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 3] + [0, 0, 0, 1],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, 3) == 1

    insert(db, 25, 40)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 3] + [0, 0, 0, 1],
            [1] + [0, 1, 2, 1] + [0, 2, 2, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, 25) == 40

    minimum_key = 0
    maximum_key = 2**db.key_size - 1

    minimum_value = 0
    maximum_value = 2**db.value_size - 1

    db.initialize()
    assert np.array_equal(
        inspect(db),
        [
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    insert(db, minimum_key, minimum_value)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, minimum_key) == minimum_value

    replace(db, minimum_key, maximum_value)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 0] + [3, 3, 3, 3],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, minimum_key) == maximum_value

    insert(db, maximum_key, maximum_value)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 0] + [3, 3, 3, 3],
            [1] + [3, 3, 3, 3] + [3, 3, 3, 3],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, maximum_key) == maximum_value

    replace(db, maximum_key, minimum_value)
    assert np.array_equal(
        inspect(db),
        [
            [1] + [0, 0, 0, 0] + [3, 3, 3, 3],
            [1] + [3, 3, 3, 3] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
            [0] + [0, 0, 0, 0] + [0, 0, 0, 0],
        ],
    )

    assert query(db, maximum_key) == minimum_value
