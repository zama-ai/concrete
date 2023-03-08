import time

import concrete.numpy as cnp
import numpy as np


NUMBER_OF_ENTRIES = 5
CHUNK_SIZE = 4

KEY_SIZE = 32
VALUE_SIZE = 32

assert KEY_SIZE % CHUNK_SIZE == 0
assert VALUE_SIZE % CHUNK_SIZE == 0

NUMBER_OF_KEY_CHUNKS = KEY_SIZE // CHUNK_SIZE
NUMBER_OF_VALUE_CHUNKS = VALUE_SIZE // CHUNK_SIZE

STATE_SHAPE = (NUMBER_OF_ENTRIES, 1 + NUMBER_OF_KEY_CHUNKS + NUMBER_OF_VALUE_CHUNKS)

FLAG = 0
KEY = slice(1, 1 + NUMBER_OF_KEY_CHUNKS)
VALUE = slice(1 + NUMBER_OF_KEY_CHUNKS, None)


def encode(number: int, width: int) -> np.array:
    binary_repr = np.binary_repr(number, width=width)
    blocks = [binary_repr[i:i+CHUNK_SIZE] for i in range(0, len(binary_repr), CHUNK_SIZE)]
    return np.array([int(block, 2) for block in blocks])

def encode_key(number: int) -> np.array:
    return encode(number, width=KEY_SIZE)

def encode_value(number: int) -> np.array:
    return encode(number, width=VALUE_SIZE)

def decode(encoded_number: np.array) -> int:
    result = 0
    for i in range(len(encoded_number)):
        result += 2**(CHUNK_SIZE*i) * encoded_number[(len(encoded_number) - i) - 1]
    return result


keep_selected_lut = cnp.LookupTable([0 for _ in range(16)] + [i for i in range(16)])

def _insert_impl(state, key, value):
    flags = state[:, FLAG]

    selection = cnp.zeros(NUMBER_OF_ENTRIES)

    found = cnp.zero()
    for i in range(NUMBER_OF_ENTRIES):
        packed_flag_and_already_found = (found * 2) + flags[i]
        is_selected = (packed_flag_and_already_found == 0)

        selection[i] = is_selected
        found += is_selected

    state_update = cnp.zeros(STATE_SHAPE)
    state_update[:, FLAG] = selection

    selection = selection.reshape((-1, 1))

    packed_selection_and_key = (selection * (2 ** CHUNK_SIZE)) + key
    key_update = keep_selected_lut[packed_selection_and_key]

    packed_selection_and_value = selection * (2 ** CHUNK_SIZE) + value
    value_update = keep_selected_lut[packed_selection_and_value]

    state_update[:, KEY] = key_update
    state_update[:, VALUE] = value_update

    new_state = state + state_update
    return new_state

def _replace_impl(state, key, value):
    flags = state[:, FLAG]
    keys = state[:, KEY]
    values = state[:, VALUE]

    equal_rows = (np.sum((keys - key) == 0, axis=1) == NUMBER_OF_KEY_CHUNKS)
    selection = (flags * 2 + equal_rows == 3).reshape((-1, 1))

    packed_selection_and_value = selection * (2 ** CHUNK_SIZE) + value
    set_value = keep_selected_lut[packed_selection_and_value]

    inverse_selection = 1 - selection
    packed_inverse_selection_and_values = inverse_selection * (2 ** CHUNK_SIZE) + values
    kept_values = keep_selected_lut[packed_inverse_selection_and_values]

    new_values = kept_values + set_value
    state[:, VALUE] = new_values

    return state

def _query_impl(state, key):
    keys = state[:, KEY]
    values = state[:, VALUE]

    selection = (np.sum((keys - key) == 0, axis=1) == NUMBER_OF_KEY_CHUNKS).reshape((-1, 1))
    found = np.sum(selection)

    packed_selection_and_values = selection * (2 ** CHUNK_SIZE) + values
    value_selection = keep_selected_lut[packed_selection_and_values]
    value = np.sum(value_selection, axis=0)

    return cnp.array([found, *value])


class KeyValueDatabase:

    _state: np.ndarray

    _insert_circuit: cnp.Circuit
    _replace_circuit: cnp.Circuit
    _query_circuit: cnp.Circuit

    def __init__(self):
        self._state = np.zeros(STATE_SHAPE, dtype=np.int64)

        inputset_binary = [
            (
                # state
                np.zeros(STATE_SHAPE, dtype=np.int64),
                # key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
            )
        ]
        inputset_ternary = [
            (
                # state
                np.zeros(STATE_SHAPE, dtype=np.int64),
                # key
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
                # value
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1),
            )
        ]

        configuration = cnp.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
        )

        insert_compiler = cnp.Compiler(
            _insert_impl,
            {"state": "encrypted", "key": "encrypted", "value": "encrypted"}
        )
        replace_compiler = cnp.Compiler(
            _replace_impl,
            {"state": "encrypted", "key": "encrypted", "value": "encrypted"}
        )
        query_compiler = cnp.Compiler(
            _query_impl,
            {"state": "encrypted", "key": "encrypted"}
        )

        print()

        print("Compiling insertion circuit...")
        start = time.time()
        self._insert_circuit = insert_compiler.compile(inputset_ternary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Compiling replacement circuit...")
        start = time.time()
        self._replace_circuit = replace_compiler.compile(inputset_ternary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Compiling query circuit...")
        start = time.time()
        self._query_circuit = query_compiler.compile(inputset_binary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Generating insertion keys...")
        start = time.time()
        self._insert_circuit.keygen()
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
        self._state = self._insert_circuit.encrypt_run_decrypt(
            self._state, encode_key(key), encode_value(value)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    def replace(self, key, value):
        print()
        print(f"Replacing...")
        start = time.time()
        self._state = self._replace_circuit.encrypt_run_decrypt(
            self._state, encode_key(key), encode_value(value)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    def query(self, key):
        print()
        print(f"Querying...")
        start = time.time()
        result = self._query_circuit.encrypt_run_decrypt(
            self._state, encode_key(key)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        if result[0] == 0:
            return None

        return decode(result[1:])


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
minimum_key = 1
maximum_key = 2 ** KEY_SIZE - 1
# Define lower/upper bounds for the value
minimum_value = 1
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
