"""
Tests of execution with Rust keygen.
"""

import os
import tempfile

from concrete import fhe


def _compute(x, y):
    return (x + y) % 2


def test_rust_keygen():
    """Test execution with Rust keygen"""
    compiler = fhe.Compiler(_compute, {"x": "encrypted", "y": "encrypted"})
    inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

    circuit = compiler.compile(inputset)

    ### Keygen with Rust ##########################################################
    _, keyinfo_path = tempfile.mkstemp()
    _, keyset_path = tempfile.mkstemp()

    # seriliaze key info
    keyset_info = circuit.client.specs.program_info.get_keyset_info()
    serialized_key_info = keyset_info.serialize()
    with open(keyinfo_path, "wb") as f:
        f.write(serialized_key_info)

    # run keygen in Rust
    bin_path = (
        f"{os.path.dirname(os.path.abspath(__file__))}"
        "/../../../concrete-rust/concrete-keygen/target/release/keygen"
    )
    assert os.system(f"{bin_path} {keyinfo_path} {keyset_path}") == 0  # noqa: S605

    # deserialize keyset
    with open(keyset_path, "rb") as f:
        serialized_keyset = f.read()

    os.remove(keyinfo_path)
    os.remove(keyset_path)

    circuit.client.keys.load_from_bytes(serialized_keyset)
    ###############################################################################

    encrypted_x, encrypted_y = circuit.encrypt(2, 6)
    encrypted_result = circuit.run(encrypted_x, encrypted_y)
    result = circuit.decrypt(encrypted_result)

    assert result == _compute(2, 6)
