"""
Tests of the examples.
"""

import os
from typing import Optional

import numpy as np
import pytest

from examples.game_of_life.game_of_life import GameOfLife
from examples.key_value_database.static_size import StaticKeyValueDatabase
from examples.levenshtein_distance.levenshtein_distance import (
    Alphabet,
    LevenshteinDistance,
    levenshtein_clear,
)


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


@pytest.mark.parametrize(
    "mode",
    [
        "simulate",
        "fhe",
    ],
)
def test_levenshtein_distance(mode, helpers):
    """
    Test levenshtein distance example.
    """

    configuration = helpers.configuration()
    if mode == "simulate":
        configuration = configuration.fork(fhe_execution=False, fhe_simulation=True)

    alphabet = Alphabet.lowercase()
    max_string_length = 5

    levenshtein_distance = LevenshteinDistance(alphabet, max_string_length, configuration)
    levenshtein_distance.module.keygen()

    samples = [
        # same
        ("hello", "hello", 0),
        # one character missing from the end
        ("hell", "hello", 1),
        ("hello", "hell", 1),
        # one character missing from the start
        ("ello", "hello", 1),
        ("hello", "ello", 1),
        # one character missing from the middle
        ("hllo", "hello", 1),
        ("hello", "hllo", 1),
        # two characters missing from the start and the end
        ("ell", "hello", 2),
        ("hello", "ell", 2),
        # three characters missing from the start, the end and the middle
        ("el", "hello", 3),
        ("hello", "el", 3),
        # shifted one character
        ("hello", "elloh", 2),
        ("elloh", "hello", 2),
        # shifted two characters
        ("hello", "llohe", 4),
        ("llohe", "hello", 4),
        # shifted three characters
        ("hello", "lohel", 4),
        ("lohel", "hello", 4),
        # shifted four characters
        ("hello", "ohell", 2),
        ("ohell", "hello", 2),
        # completely different
        ("hello", "numpy", 5),
    ]

    for str1, str2, expected_distance in samples:
        actual_distance = levenshtein_distance.calculate(str1, str2, mode, show_distance=True)
        assert actual_distance == expected_distance


@pytest.mark.parametrize(
    "alphabet_name",
    Alphabet.return_available_alphabets(),
)
@pytest.mark.parametrize(
    "max_length",
    [2, 3],
)
def test_levenshtein_distance_randomly(alphabet_name, max_length, helpers):
    """
    Test levenshtein distance example with randomly generated strings.
    """

    configuration = helpers.configuration().fork(fhe_execution=False, fhe_simulation=True)

    alphabet = Alphabet.init_by_name(alphabet_name)
    levenshtein_distance = LevenshteinDistance(alphabet, max_length, configuration)
    levenshtein_distance.module.keygen()

    for str1, str2 in alphabet.prepare_random_patterns(0, max_length, nb_strings=3):
        expected_distance = levenshtein_clear(str1, str2)
        actual_distance = levenshtein_distance.calculate(str1, str2, "simulate", show_distance=True)
        assert actual_distance == expected_distance


@pytest.mark.parametrize(
    "implementation",
    GameOfLife.implementations(),
)
@pytest.mark.parametrize(
    "dimension,sample_input_output_pairs",
    [
        pytest.param(
            4,
            [
                (
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                (
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                (
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                (
                    [
                        [1, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                (
                    [
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 1],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                (
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ],
                    # should become
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ),
            ],
        ),
    ],
)
def test_game_of_life(implementation, dimension, sample_input_output_pairs, helpers):
    """
    Test game of life implementation.
    """

    configuration = helpers.configuration()
    game_of_life = GameOfLife.implementation(implementation, dimension, configuration)
    game_of_life.circuit.keygen()

    for sample_input, expected_output in sample_input_output_pairs:
        sample = np.array(sample_input).reshape((1, 1, dimension, dimension))
        result = game_of_life.circuit.encrypt_run_decrypt(sample)
        actual_output = result.reshape((dimension, dimension))
        assert np.array_equal(
            actual_output, expected_output
        ), f"""

Expected Output
===============
{expected_output}

Actual Output
=============
{actual_output}

            """


def test_tfhers_example():
    """
    Test the TFHE-rs example.
    """
    path_to_test_script = f"{os.path.dirname(os.path.abspath(__file__))}/../../examples/tfhers/"
    test_script_filename = "test.sh"
    assert (
        os.system(f"cd {path_to_test_script} && sh {test_script_filename}") == 0  # noqa: S605
    ), "test script failed"


def test_tfhers_ml_example():
    """
    Test the TFHE-rs ML example.
    """
    path_to_test_script = f"{os.path.dirname(os.path.abspath(__file__))}/../../examples/tfhers-ml/"
    test_script_filename = "test.sh"
    assert (
        os.system(f"cd {path_to_test_script} && sh {test_script_filename}") == 0  # noqa: S605
    ), "test script failed"
