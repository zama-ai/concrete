import random
from pathlib import Path

import numpy as np
import py_progress_tracker as progress

from concrete import fhe
from examples.key_value_database.static_size import StaticKeyValueDatabase

targets = []

for number_of_entries in [8, 16]:
    for key_size in [8, 16]:
        for value_size in [8, 16]:
            for chunk_size in [2, 4]:
                targets.append(
                    {
                        "id": (
                            f"static-kvdb-insert :: "
                            f"{number_of_entries} * {key_size}->{value_size} ^ {chunk_size}"
                        ),
                        "name": (
                            f"Insertion to "
                            f"static key-value database "
                            f"from {key_size}b to {value_size}b "
                            f"with chunk size of {chunk_size} "
                            f"on {number_of_entries} entries"
                        ),
                        "parameters": {
                            "operation": "insert",
                            "number_of_entries": number_of_entries,
                            "key_size": key_size,
                            "value_size": value_size,
                            "chunk_size": chunk_size,
                        },
                    }
                )
                targets.append(
                    {
                        "id": (
                            f"static-kvdb-replace :: "
                            f"{number_of_entries} * {key_size}->{value_size} ^ {chunk_size}"
                        ),
                        "name": (
                            f"Replacement in "
                            f"static key-value database "
                            f"from {key_size}b to {value_size}b "
                            f"with chunk size of {chunk_size} "
                            f"on {number_of_entries} entries"
                        ),
                        "parameters": {
                            "operation": "replace",
                            "number_of_entries": number_of_entries,
                            "key_size": key_size,
                            "value_size": value_size,
                            "chunk_size": chunk_size,
                        },
                    }
                )
                targets.append(
                    {
                        "id": (
                            f"static-kvdb-query :: "
                            f"{number_of_entries} * {key_size}->{value_size} ^ {chunk_size}"
                        ),
                        "name": (
                            f"Query of "
                            f"static key-value database "
                            f"from {key_size}b to {value_size}b "
                            f"with chunk size of {chunk_size} "
                            f"on {number_of_entries} entries"
                        ),
                        "parameters": {
                            "operation": "query",
                            "number_of_entries": number_of_entries,
                            "key_size": key_size,
                            "value_size": value_size,
                            "chunk_size": chunk_size,
                        },
                    }
                )


def benchmark_insert(db: StaticKeyValueDatabase, client: fhe.Client, server: fhe.Server):
    print("Warming up...")

    sample_key = random.randint(0, 2**db.key_size - 1)
    sample_value = random.randint(0, 2**db.value_size - 1)

    # Initial state only contains odd keys for benchmarks.
    # To avoid collisions, we'll make sure that sample_key is even.
    if sample_key % 2 == 1:
        sample_key -= 1

    encoded_sample_key = db.encode_key(sample_key)
    encoded_sample_value = db.encode_value(sample_value)

    _, encrypted_sample_key, encrypted_sample_value = client.encrypt(
        None,
        encoded_sample_key,
        encoded_sample_value,
        function_name="insert",
    )
    ran = server.run(  # noqa: F841
        db.state,
        encrypted_sample_key,
        encrypted_sample_value,
        function_name="insert",
        evaluation_keys=client.evaluation_keys,
    )

    for i in range(5):
        print(f"Running subsample {i + 1} out of 5...")

        sample_key = random.randint(0, 2**db.key_size - 1)
        sample_value = random.randint(0, 2**db.value_size - 1)

        if sample_key % 2 == 1:
            sample_key -= 1

        encoded_sample_key = db.encode_key(sample_key)
        encoded_sample_value = db.encode_value(sample_value)

        _, encrypted_sample_key, encrypted_sample_value = client.encrypt(
            None,
            encoded_sample_key,
            encoded_sample_value,
            function_name="insert",
        )

        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            ran = server.run(  # noqa: F841
                db.state,
                encrypted_sample_key,
                encrypted_sample_value,
                function_name="insert",
                evaluation_keys=client.evaluation_keys,
            )


def benchmark_replace(db: StaticKeyValueDatabase, client: fhe.Client, server: fhe.Server):
    print("Warming up...")

    sample_key = random.randint(0, db.number_of_entries // 2) * 2
    sample_value = random.randint(0, db.number_of_entries // 2) * 2

    # Initial state only contains odd keys for benchmarks.
    # To actually replace, we'll make sure that sample_key is odd.
    if sample_key % 2 == 0:
        sample_key += 1

    encoded_sample_key = db.encode_key(sample_key)
    encoded_sample_value = db.encode_value(sample_value)

    _, encrypted_sample_key, encrypted_sample_value = client.encrypt(
        None,
        encoded_sample_key,
        encoded_sample_value,
        function_name="replace",
    )
    ran = server.run(  # noqa: F841
        db.state,
        encrypted_sample_key,
        encrypted_sample_value,
        function_name="replace",
        evaluation_keys=client.evaluation_keys,
    )

    for i in range(5):
        print(f"Running subsample {i + 1} out of 5...")

        sample_key = random.randint(0, db.number_of_entries - 1)
        sample_value = random.randint(0, db.number_of_entries - 1)

        if sample_key % 2 == 0:
            sample_key += 1

        encoded_sample_key = db.encode_key(sample_key)
        encoded_sample_value = db.encode_value(sample_value)

        _, encrypted_sample_key, encrypted_sample_value = client.encrypt(
            None,
            encoded_sample_key,
            encoded_sample_value,
            function_name="replace",
        )

        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            ran = server.run(  # noqa: F841
                db.state,
                encrypted_sample_key,
                encrypted_sample_value,
                function_name="replace",
                evaluation_keys=client.evaluation_keys,
            )


def benchmark_query(db: StaticKeyValueDatabase, client: fhe.Client, server: fhe.Server):
    print("Warming up...")

    sample_key = random.randint(0, db.number_of_entries - 1)
    encoded_sample_key = db.encode_key(sample_key)

    _, encrypted_sample_key = client.encrypt(
        None,
        encoded_sample_key,
        function_name="query",
    )
    ran = server.run(  # noqa: F841
        db.state,
        encrypted_sample_key,
        function_name="query",
        evaluation_keys=client.evaluation_keys,
    )

    for i in range(5):
        print(f"Running subsample {i + 1} out of 5...")

        sample_key = random.randint(0, db.number_of_entries - 1)
        encoded_sample_key = db.encode_key(sample_key)

        _, encrypted_sample_key = client.encrypt(None, encoded_sample_key, function_name="query")
        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            ran = server.run(  # noqa: F841
                db.state,
                encrypted_sample_key,
                function_name="query",
                evaluation_keys=client.evaluation_keys,
            )


@progress.track(targets)
def main(operation, number_of_entries, key_size, value_size, chunk_size):
    print("Compiling...")
    cached_server = Path(f"{number_of_entries}.{key_size}.{value_size}.{chunk_size}.server.zip")
    if cached_server.exists():
        db = StaticKeyValueDatabase(
            # database parameters
            number_of_entries,
            key_size,
            value_size,
            chunk_size,
            compiled=False,
        )
        server = fhe.Server.load(cached_server)
        client = fhe.Client(server.client_specs, keyset_cache_directory=".keys")
    else:
        configuration = {}
        db = StaticKeyValueDatabase(
            # database parameters
            number_of_entries,
            key_size,
            value_size,
            chunk_size,
            # configuration overrides
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
        )
        db.module.server.save(cached_server)

        server = db.module.server
        client = db.module.client

    db.state = server.run(
        client.encrypt(
            [
                np.array([1] + db.encode_key(i).tolist() + db.encode_value(i).tolist()) * (i % 2)
                for i in range(db.number_of_entries)
            ],
            function_name="reset",
        ),
        function_name="reset",
        evaluation_keys=client.evaluation_keys,
    )

    print("Generating keys...")
    client.keygen()

    if operation == "insert":
        benchmark_insert(db, client, server)
    elif operation == "replace":
        benchmark_replace(db, client, server)
    elif operation == "query":
        benchmark_query(db, client, server)
    else:
        raise ValueError(f"Invalid operation: {operation}")
