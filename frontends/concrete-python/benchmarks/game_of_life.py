"""
Benchmarks of the game of life example.
"""

from pathlib import Path

import numpy as np
import py_progress_tracker as progress

from concrete import fhe
from examples.game_of_life.game_of_life import GameOfLife


def benchmark_computing_next_state(gol: GameOfLife, client: fhe.Client, server: fhe.Server):
    """
    Benchmark computing the next state in the game of life simulation.
    """

    print("Warming up...")

    sample_state = np.random.randint(0, 2, size=(1, 1, gol.dimension, gol.dimension))
    encrypted_sample_state = client.encrypt(sample_state)
    ran = server.run(  # noqa: F841  # pylint: disable=unused-variable
        encrypted_sample_state,
        evaluation_keys=client.evaluation_keys,
    )

    for i in range(5):
        print(f"Running subsample {i + 1} out of 5...")

        sample_state = np.random.randint(0, 2, size=(1, 1, gol.dimension, gol.dimension))
        encrypted_sample_state = client.encrypt(sample_state)

        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            ran = server.run(  # noqa: F841  # pylint: disable=unused-variable
                encrypted_sample_state,
                evaluation_keys=client.evaluation_keys,
            )


def targets():
    """
    Generates targets to benchmark.
    """

    result = []
    for dimension in [4, 8]:
        for implementation in GameOfLife.implementations():
            result.append(
                {
                    "id": (
                        f"game-of-life :: "
                        f"Game of Life "
                        f"| {dimension} x {dimension} | {implementation}"
                    ),
                    "name": (
                        f"Advancing Game of Life simulation "
                        f"of size {dimension} x {dimension} "
                        f"with {implementation.replace('_', ' ')}"
                    ),
                    "parameters": {
                        "dimension": dimension,
                        "implementation": implementation,
                    },
                }
            )
    return result


@progress.track(targets())
def main(dimension, implementation):
    """
    Benchmark a target.

    Args:
        dimension:
            dimension of the game of life simulation

        implementation:
            implementation of the game of life simulation
    """

    print("Compiling...")
    cached_server = Path(f"game_of_life.{dimension}.{implementation}.server.zip")
    if cached_server.exists():
        gol = GameOfLife.implementation(
            implementation,
            dimension,
            compiled=False,
        )
        server = fhe.Server.load(cached_server)
        client = fhe.Client(server.client_specs, keyset_cache_directory=".keys")
    else:
        gol = GameOfLife.implementation(
            implementation,
            dimension,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
            compiled=True,
        )
        server = gol.circuit.server
        client = gol.circuit.client

        server.save(cached_server)

    print("Generating keys...")
    client.keygen()

    benchmark_computing_next_state(gol, client, server)
