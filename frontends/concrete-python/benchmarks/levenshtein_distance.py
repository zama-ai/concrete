"""
Benchmarks of the levenshtein distance example.
"""

from functools import lru_cache
from pathlib import Path

import py_progress_tracker as progress

from concrete import fhe
from examples.levenshtein_distance.levenshtein_distance import Alphabet, LevenshteinDistance


@lru_cache
def levenshtein_on_server(
    server: fhe.Server,
    x: tuple[fhe.Value],
    y: tuple[fhe.Value],
    evaluation_keys: fhe.EvaluationKeys,
):
    """
    Compute levenshtein distance on a server.
    """

    if len(x) == 0:
        return server.run(
            len(y),
            function_name="constant",
            evaluation_keys=evaluation_keys,
        )

    if len(y) == 0:
        return server.run(
            len(x),
            function_name="constant",
            evaluation_keys=evaluation_keys,
        )

    if_equal = levenshtein_on_server(server, x[1:], y[1:], evaluation_keys)
    case_1 = levenshtein_on_server(server, x[1:], y, evaluation_keys)
    case_2 = levenshtein_on_server(server, x, y[1:], evaluation_keys)
    case_3 = if_equal

    is_equal = server.run(
        x[0],
        y[0],
        function_name="equal",
        evaluation_keys=evaluation_keys,
    )
    result = server.run(
        is_equal,
        if_equal,
        case_1,
        case_2,
        case_3,
        function_name="mix",
        evaluation_keys=evaluation_keys,
    )

    return result


def targets():
    """
    Generates targets to benchmark.
    """

    result = []
    for alphabet in ["ACTG", "string"]:
        for max_string_length in [2, 4, 8]:
            result.append(
                {
                    "id": (
                        f"levenshtein-distance :: "
                        f"Levenshtein distance "
                        f"| alphabet = {alphabet} "
                        f"| max_string_size = {max_string_length}"
                    ),
                    "name": (
                        f"Levenshtein distance between two strings "
                        f"of length {max_string_length} "
                        f"from {alphabet} alphabet"
                    ),
                    "parameters": {
                        "alphabet": alphabet,
                        "max_string_length": max_string_length,
                    },
                }
            )
    return result


@progress.track(targets())
def main(alphabet, max_string_length):
    """
    Benchmark a target.

    Args:
        alphabet:
            alphabet of the inputs
        max_string_length:
            maximum size of the inputs
    """

    cached_server = Path(f"levenshtein.{alphabet}.{max_string_length}.server.zip")
    alphabet = Alphabet.init_by_name(alphabet)

    print("Compiling...")
    if cached_server.exists():
        server = fhe.Server.load(cached_server)
        client = fhe.Client(server.client_specs, keyset_cache_directory=".keys")
    else:
        levenshtein_distance = LevenshteinDistance(
            alphabet,
            max_string_length,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )
        levenshtein_distance.module.server.save(cached_server)

        server = levenshtein_distance.module.server
        client = levenshtein_distance.module.client

    print("Generating keys...")
    client.keygen()

    print("Warming up...")

    sample_a = alphabet.random_string(max_string_length)
    sample_b = alphabet.random_string(max_string_length)

    encrypted_sample_a = tuple(
        client.encrypt(ai, None, function_name="equal")[0] for ai in alphabet.encode(sample_a)
    )
    encrypted_sample_b = tuple(
        client.encrypt(None, bi, function_name="equal")[1] for bi in alphabet.encode(sample_b)
    )

    levenshtein_on_server(
        server,
        encrypted_sample_a,
        encrypted_sample_b,
        client.evaluation_keys,
    )

    for i in range(5):
        print(f"Running subsample {i + 1} out of 5...")

        sample_a = alphabet.random_string(max_string_length)
        sample_b = alphabet.random_string(max_string_length)

        encrypted_sample_a = tuple(
            client.encrypt(ai, None, function_name="equal")[0] for ai in alphabet.encode(sample_a)
        )
        encrypted_sample_b = tuple(
            client.encrypt(None, bi, function_name="equal")[1] for bi in alphabet.encode(sample_b)
        )

        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            levenshtein_on_server(
                server,
                encrypted_sample_a,
                encrypted_sample_b,
                client.evaluation_keys,
            )
