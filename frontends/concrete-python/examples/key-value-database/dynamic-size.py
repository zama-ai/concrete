import time

import concrete.numpy as cnp
import numpy as np


CHUNK_SIZE = 4

KEY_SIZE = 32
VALUE_SIZE = 32

assert KEY_SIZE % CHUNK_SIZE == 0
assert VALUE_SIZE % CHUNK_SIZE == 0

NUMBER_OF_KEY_CHUNKS = KEY_SIZE // CHUNK_SIZE
NUMBER_OF_VALUE_CHUNKS = VALUE_SIZE // CHUNK_SIZE


def encode(number, width):
    binary_repr = np.binary_repr(number, width=width)
    blocks = [binary_repr[i:i+CHUNK_SIZE] for i in range(0, len(binary_repr), CHUNK_SIZE)]
    return np.array([int(block, 2) for block in blocks])

def encode_key(number):
    return encode(number, width=KEY_SIZE)

def encode_value(number):
    return encode(number, width=VALUE_SIZE)

def decode(encoded_number):
    result = 0
    for i in range(len(encoded_number)):
        result += 2**(CHUNK_SIZE*i) * encoded_number[(len(encoded_number) - i) - 1]
    return result


keep_if_match_lut = cnp.LookupTable([0 for _ in range(16)] + [i for i in range(16)])
keep_if_no_match_lut = cnp.LookupTable([i for i in range(16)] + [0 for _ in range(16)])

def _replace_impl(key, value, candidate_key, candidate_value):
    match = np.sum((candidate_key - key) == 0) == NUMBER_OF_KEY_CHUNKS

    packed_match_and_value = (2 ** CHUNK_SIZE) * match + value
    value_if_match_else_zeros = keep_if_match_lut[packed_match_and_value]

    packed_match_and_candidate_value = (2 ** CHUNK_SIZE) * match + candidate_value
    zeros_if_match_else_candidate_value = keep_if_no_match_lut[packed_match_and_candidate_value]

    return value_if_match_else_zeros + zeros_if_match_else_candidate_value

def _query_impl(key, candidate_key, candidate_value):
    match = np.sum((candidate_key - key) == 0) == NUMBER_OF_KEY_CHUNKS

    packed_match_and_candidate_value = (2 ** CHUNK_SIZE) * match + candidate_value
    candidate_value_if_match_else_zeros = keep_if_match_lut[packed_match_and_candidate_value]

    return cnp.array([match, *candidate_value_if_match_else_zeros])


class KeyValueDatabase:

    _state: list[np.ndarray]

    _replace_circuit: cnp.Circuit
    _query_circuit: cnp.Circuit

    def __init__(self):
        self._state = []

        configuration = cnp.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
        )

        replace_compiler = cnp.Compiler(
            _replace_impl,
            {
                "key": "encrypted", "value": "encrypted",
                "candidate_key": "encrypted", "candidate_value": "encrypted",
            }
        )
        query_compiler = cnp.Compiler(
            _query_impl,
            {
                "key": "encrypted",
                "candidate_key": "encrypted", "candidate_value": "encrypted",
            }
        )

        replace_inputset = [
            (
                # key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # value
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # candidate_key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # candidate_value
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
            )
        ]
        query_inputset = [
            (
                # key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # candidate_key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # candidate_value
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
            )
        ]

        print("Compiling replacement circuit...")
        start = time.time()
        self._replace_circuit = replace_compiler.compile(replace_inputset, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Compiling query circuit...")
        start = time.time()
        self._query_circuit = query_compiler.compile(query_inputset, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Generating replacement keys...")
        start = time.time()
        self._replace_circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Generating query keys...")
        start = time.time()
        self._query_circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    def insert(self, key, value):
        print()
        print(f"Inserting...")
        start = time.time()

        self._state.append([encode_key(key), encode_value(value)])

        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    def replace(self, key, value):
        print()
        print(f"Replacing...")
        start = time.time()

        encoded_key = encode_key(key)
        encoded_value = encode_value(value)

        for entry in self._state:
            entry[1] = self._replace_circuit.encrypt_run_decrypt(encoded_key, encoded_value, *entry)

        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    def query(self, key):
        print()
        print(f"Querying...")
        start = time.time()

        encoded_key = encode_key(key)

        accumulation = np.zeros(1 + NUMBER_OF_VALUE_CHUNKS, dtype=np.uint64)
        for entry in self._state:
            accumulation += self._query_circuit.encrypt_run_decrypt(encoded_key, *entry)

        match_count = accumulation[0]
        if match_count > 1:
            raise RuntimeError("Key inserted multiple times")

        result = decode(accumulation[1:]) if match_count == 1 else None

        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        return result


db = KeyValueDatabase()

# Test: Insert/Query
db.insert(3, 4)
assert db.query(3) == 4

db.replace(3, 1)
assert db.query(3) == 1

# Test: Insert/Query
db.insert(25, 40)
assert db.query(25) == 40

# Test: Query Not Found
assert db.query(4) is None

# Test: Replace/Query
db.replace(3, 5)
assert db.query(3) == 5


# Define lower/upper bounds for the key
minimum_key = 0
maximum_key = 2 ** KEY_SIZE - 1
# Define lower/upper bounds for the value
minimum_value = 0
maximum_value = 2 ** VALUE_SIZE - 1


# Test: Insert/Replace/Query Bounds
# Insert (key: minimum_key , value: minimum_value) into the database
db.insert(minimum_key, minimum_value)

# Query the database for the key=minimum_key
# The value minimum_value should be returned
assert db.query(minimum_key) == minimum_value

# Insert (key: maximum_key , value: maximum_value) into the database
db.insert(maximum_key, maximum_value)

# Query the database for the key=maximum_key
# The value maximum_value should be returned
assert db.query(maximum_key) == maximum_value

# Replace the value of key=minimum_key with maximum_value
db.replace(minimum_key, maximum_value)

# Query the database for the key=minimum_key
# The value maximum_value should be returned
assert db.query(minimum_key) == maximum_value

# Replace the value of key=maximum_key with minimum_value
db.replace(maximum_key, minimum_value)

# Query the database for the key=maximum_key
# The value minimum_value should be returned
assert db.query(maximum_key) == minimum_value
