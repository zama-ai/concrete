import time
from typing import Optional, Union

import numpy as np

from concrete import fhe


class StaticKeyValueDatabase:
    number_of_entries: int
    key_size: int
    value_size: int
    chunk_size: int

    _number_of_key_chunks: int
    _number_of_value_chunks: int
    _state_shape: tuple[int, ...]

    module: fhe.Module
    state: Optional[fhe.Value]

    def __init__(
        self,
        number_of_entries: int,
        key_size: int = 32,
        value_size: int = 32,
        chunk_size: int = 4,
        compiled: bool = True,
        configuration: Optional[fhe.Configuration] = None,
    ):
        self.number_of_entries = number_of_entries
        self.key_size = key_size
        self.value_size = value_size
        self.chunk_size = chunk_size

        self._number_of_key_chunks = key_size // chunk_size
        self._number_of_value_chunks = value_size // chunk_size
        self._state_shape = (
            number_of_entries,
            1 + self._number_of_key_chunks + self._number_of_value_chunks,
        )

        if compiled:
            if configuration is None:
                configuration = fhe.Configuration()

            self.module = self._module(
                configuration.fork(
                    multivariate_strategy_preference=fhe.MultivariateStrategy.PROMOTED,
                )
            )

        self.state = None

    def _encode(self, number: int, width: int) -> np.ndarray:
        binary_repr = np.binary_repr(number, width=width)
        blocks = [
            binary_repr[i : (i + self.chunk_size)]
            for i in range(0, len(binary_repr), self.chunk_size)
        ]
        return np.array([int(block, 2) for block in blocks])

    def _decode(self, encoded_number: np.ndarray) -> int:
        result = 0
        for i in range(len(encoded_number)):
            result += 2 ** (self.chunk_size * i) * encoded_number[(len(encoded_number) - i) - 1]
        return result

    def encode_key(self, key: int) -> np.ndarray:
        return self._encode(key, width=self.key_size)

    def decode_key(self, encoded_key: np.ndarray) -> int:
        return self._decode(encoded_key)

    def encode_value(self, value: int) -> np.ndarray:
        return self._encode(value, width=self.value_size)

    def decode_value(self, encoded_value: np.ndarray) -> int:
        return self._decode(encoded_value)

    def _module(self, configuration: fhe.Configuration) -> fhe.Module:
        flag_slice = 0
        key_slice = slice(1, 1 + self._number_of_key_chunks)
        value_slice = slice(1 + self._number_of_key_chunks, None)

        chunk_size = self.chunk_size
        number_of_entries = self.number_of_entries
        number_of_key_chunks = self._number_of_key_chunks
        state_shape = self._state_shape

        # pylint: disable=no-self-argument
        @fhe.module()
        class StaticKeyValueDatabaseModule:
            @fhe.function({"state": "clear"})
            def reset(state):
                return state + fhe.zero()

            @fhe.function({"state": "encrypted", "key": "encrypted", "value": "encrypted"})
            def insert(state, key, value):
                flags = state[:, flag_slice]

                selection = fhe.zeros(number_of_entries)

                found = fhe.zero()
                for i in range(number_of_entries):
                    is_selected = fhe.multivariate(
                        lambda found, flag: int(found == 0 and flag == 0)
                    )(found, flags[i])

                    selection[i] = is_selected
                    found += is_selected

                state_update = fhe.zeros(state_shape)
                state_update[:, flag_slice] = selection

                selection = selection.reshape((-1, 1))

                key_update = fhe.multivariate(lambda selection, key: selection * key)(
                    selection, key
                )
                value_update = fhe.multivariate(lambda selection, value: selection * value)(
                    selection, value
                )

                state_update[:, key_slice] = key_update
                state_update[:, value_slice] = value_update

                new_state = state + state_update
                return fhe.refresh(new_state)

            @fhe.function({"state": "encrypted", "key": "encrypted", "value": "encrypted"})
            def replace(state, key, value):
                flags = state[:, flag_slice]
                keys = state[:, key_slice]
                values = state[:, value_slice]

                number_of_matching_chunks = np.sum((keys - key) == 0, axis=1)
                fhe.hint(number_of_matching_chunks, can_store=number_of_key_chunks)

                equal_rows = number_of_matching_chunks == number_of_key_chunks
                selection = (flags * 2 + equal_rows == 3).reshape((-1, 1))

                set_value = fhe.multivariate(lambda selection, value: selection * value)(
                    selection, value
                )

                kept_values = fhe.multivariate(
                    lambda inverse_selection, values: inverse_selection * values
                )(1 - selection, values)

                new_values = kept_values + set_value
                state[:, value_slice] = new_values

                return fhe.refresh(state)

            @fhe.function({"state": "encrypted", "key": "encrypted"})
            def query(state, key):
                keys = state[:, key_slice]
                values = state[:, value_slice]

                number_of_matching_chunks = np.sum((keys - key) == 0, axis=1)
                fhe.hint(number_of_matching_chunks, can_store=number_of_key_chunks)

                selection = (number_of_matching_chunks == number_of_key_chunks).reshape((-1, 1))
                found = np.sum(selection)
                fhe.hint(found, can_store=number_of_entries)

                value_selection = fhe.multivariate(lambda selection, values: selection * values)(
                    selection, values
                )

                value = np.sum(value_selection, axis=0)
                fhe.hint(value, bit_width=chunk_size)

                return found, value

            @fhe.function({"state": "encrypted"})
            def inspect(state):
                return state

            # pylint: enable=no-self-argument

            composition = fhe.Wired(
                {
                    # from reset
                    fhe.Wire(fhe.Output(reset, 0), fhe.Input(insert, 0)),
                    fhe.Wire(fhe.Output(reset, 0), fhe.Input(replace, 0)),
                    fhe.Wire(fhe.Output(reset, 0), fhe.Input(query, 0)),
                    fhe.Wire(fhe.Output(reset, 0), fhe.Input(inspect, 0)),
                    # from insert
                    fhe.Wire(fhe.Output(insert, 0), fhe.Input(insert, 0)),
                    fhe.Wire(fhe.Output(insert, 0), fhe.Input(replace, 0)),
                    fhe.Wire(fhe.Output(insert, 0), fhe.Input(query, 0)),
                    fhe.Wire(fhe.Output(insert, 0), fhe.Input(inspect, 0)),
                    # from replace
                    fhe.Wire(fhe.Output(replace, 0), fhe.Input(insert, 0)),
                    fhe.Wire(fhe.Output(replace, 0), fhe.Input(replace, 0)),
                    fhe.Wire(fhe.Output(replace, 0), fhe.Input(query, 0)),
                    fhe.Wire(fhe.Output(replace, 0), fhe.Input(inspect, 0)),
                    # from inspect
                    fhe.Wire(fhe.Output(inspect, 0), fhe.Input(insert, 0)),
                    fhe.Wire(fhe.Output(inspect, 0), fhe.Input(replace, 0)),
                    fhe.Wire(fhe.Output(inspect, 0), fhe.Input(query, 0)),
                    fhe.Wire(fhe.Output(inspect, 0), fhe.Input(inspect, 0)),
                }
            )

        sample_state = np.array(
            [
                [i % 2] + self.encode_key(i).tolist() + self.encode_value(i).tolist()
                for i in range(self.number_of_entries)
            ]
        )

        insert_replace_inputset = [
            (
                # state
                sample_state,
                # key
                self.encode_key(i),
                # value
                self.encode_value(i),
            )
            for i in range(20)
        ]
        query_inputset = [
            (
                # state
                sample_state,
                # key
                self.encode_key(i),
            )
            for i in range(20)
        ]

        # pylint: disable=no-member
        return StaticKeyValueDatabaseModule.compile(  # type: ignore
            {
                "reset": [sample_state],
                "insert": insert_replace_inputset,
                "replace": insert_replace_inputset,
                "query": query_inputset,
                "inspect": [sample_state],
            },
            configuration,
        )
        # pylint: enable=no-member

    def keygen(self, force: bool = False):
        self.module.keygen(force=force)

    def initialize(self, initial_state: Optional[Union[list, np.ndarray]] = None):
        if initial_state is None:
            initial_state = np.zeros(self._state_shape, dtype=np.int64)

        if isinstance(initial_state, list):
            initial_state = np.array(initial_state)

        if initial_state.shape != self._state_shape:
            message = (
                f"Expected initial state to be of shape {self._state_shape} "
                f"but it's {initial_state.shape}"
            )
            raise ValueError(message)

        initial_state_clear = self.module.reset.encrypt(initial_state)
        initial_state_encrypted = self.module.reset.run(initial_state_clear)

        self.state = initial_state_encrypted  # type: ignore

    def decode_entry(self, entry: np.ndarray) -> Optional[tuple[int, int]]:
        if entry[0] == 0:
            return None

        encoded_key = entry[1 : (self._number_of_key_chunks + 1)]
        encoded_value = entry[(self._number_of_key_chunks + 1) :]

        return self.decode_key(encoded_key), self.decode_value(encoded_value)

    @property
    def insert(self) -> fhe.Function:
        return self.module.insert

    @property
    def replace(self) -> fhe.Function:
        return self.module.replace

    @property
    def query(self) -> fhe.Function:
        return self.module.query

    @property
    def inspect(self) -> fhe.Function:
        return self.module.inspect


def inspect(db: StaticKeyValueDatabase):
    encrypted_state = db.inspect.run(db.state)
    clear_state = db.inspect.decrypt(encrypted_state)
    print(clear_state)


def insert(db: StaticKeyValueDatabase, key: int, value: int):
    encoded_key, encoded_value = db.encode_key(key), db.encode_value(value)
    _, encrypted_key, encoded_value = db.insert.encrypt(  # type: ignore
        None,
        encoded_key,
        encoded_value,
    )

    print()

    print("Inserting...")
    start = time.time()
    db.state = db.insert.run(db.state, encrypted_key, encoded_value)  # type: ignore
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")


def replace(db: StaticKeyValueDatabase, key: int, value: int):
    encoded_key, encoded_value = db.encode_key(key), db.encode_value(value)
    _, encrypted_key, encoded_value = db.replace.encrypt(  # type: ignore
        None,
        encoded_key,
        encoded_value,
    )

    print()

    print("Replacing...")
    start = time.time()
    db.state = db.replace.run(db.state, encrypted_key, encoded_value)  # type: ignore
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")


def query(db: StaticKeyValueDatabase, key: int) -> Optional[int]:
    encoded_key = db.encode_key(key)
    _, encrypted_key = db.query.encrypt(None, encoded_key)  # type: ignore

    print()

    print("Querying...")
    start = time.time()
    encrypted_found, encrypted_value = db.query.run(db.state, encrypted_key)  # type: ignore
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")

    found, value = db.query.decrypt(encrypted_found, encrypted_value)  # type: ignore
    if not found:
        return None

    return db.decode_value(value)  # type: ignore


def main():
    print("Compiling...")
    start = time.time()
    db = StaticKeyValueDatabase(number_of_entries=10)
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")

    print()

    print("Generating keys...")
    start = time.time()
    db.keygen()
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")

    print()

    print("Initializing...")
    start = time.time()
    db.initialize()
    end = time.time()
    print(f"(took {end - start:.3f} seconds)")

    # Test: Insert/Query
    insert(db, 3, 4)
    assert query(db, 3) == 4

    replace(db, 3, 1)
    assert query(db, 3) == 1

    # Test: Insert/Query
    insert(db, 25, 40)
    assert query(db, 25) == 40

    # Test: Query Not Found
    assert query(db, 4) is None

    # Test: Replace/Query
    replace(db, 3, 5)
    assert query(db, 3) == 5

    # Define lower/upper bounds for the key
    minimum_key = 1
    maximum_key = 2**db.key_size - 1
    # Define lower/upper bounds for the value
    minimum_value = 1
    maximum_value = 2**db.value_size - 1

    # Test: Insert/Replace/Query Bounds
    # Insert (key: minimum_key , value: minimum_value) into the database
    insert(db, minimum_key, minimum_value)

    # Query the database for the key=minimum_key
    # The value minimum_value should be returned
    assert query(db, minimum_key) == minimum_value

    # Insert (key: maximum_key , value: maximum_value) into the database
    insert(db, maximum_key, maximum_value)

    # Query the database for the key=maximum_key
    # The value maximum_value should be returned
    assert query(db, maximum_key) == maximum_value

    # Replace the value of key=minimum_key with maximum_value
    replace(db, minimum_key, maximum_value)

    # Query the database for the key=minimum_key
    # The value maximum_value should be returned
    assert query(db, minimum_key) == maximum_value

    # Replace the value of key=maximum_key with minimum_value
    replace(db, maximum_key, minimum_value)

    # Query the database for the key=maximum_key
    # The value minimum_value should be returned
    assert query(db, maximum_key) == minimum_value


if __name__ == "__main__":
    main()
