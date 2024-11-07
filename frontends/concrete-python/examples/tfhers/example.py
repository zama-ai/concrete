import os
from functools import partial

import click
import numpy as np

from concrete import fhe
from concrete.fhe import tfhers

### Options ###########################
# These parameters were saved by running the tfhers_utils utility:
# tfhers_utils save-params tfhers_params.json
TFHERS_PARAMS_FILE = "tfhers_params.json"
FHEUINT_PRECISION = 8
IS_SIGNED = False
#######################################

tfhers_type = tfhers.get_type_from_params(
    TFHERS_PARAMS_FILE, IS_SIGNED, FHEUINT_PRECISION
)
tfhers_int = partial(tfhers.TFHERSInteger, tfhers_type)


def compute(tfhers_x, tfhers_y):
    ####### TFHE-rs to Concrete #########

    # x and y are supposed to be TFHE-rs values.
    # to_native will use type information from x and y to do
    # a correct conversion from TFHE-rs to Concrete
    concrete_x = tfhers.to_native(tfhers_x)
    concrete_y = tfhers.to_native(tfhers_y)
    ####### TFHE-rs to Concrete #########

    ####### Concrete Computation ########
    concrete_res = (concrete_x + concrete_y) % 213
    ####### Concrete Computation ########

    ####### Concrete to TFHE-rs #########
    tfhers_res = tfhers.from_native(
        concrete_res, tfhers_type
    )  # we have to specify the type we want to convert to
    ####### Concrete to TFHE-rs #########
    return tfhers_res


def ccompilee():
    compiler = fhe.Compiler(compute, {"tfhers_x": "encrypted", "tfhers_y": "encrypted"})

    inputset = [(tfhers_int(120), tfhers_int(120))]
    circuit = compiler.compile(inputset)

    tfhers_bridge = tfhers.new_bridge(circuit=circuit)
    return circuit, tfhers_bridge


@click.group()
def cli():
    pass


@cli.command()
@click.option("-s", "--secret-key", type=str, required=False)
@click.option("-o", "--output-secret-key", type=str, required=True)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
def keygen(output_secret_key: str, secret_key: str, concrete_keyset_path: str):
    """Concrete Key Generation"""

    circuit, tfhers_bridge = ccompilee()

    if os.path.exists(concrete_keyset_path):
        print(f"removing old keyset at '{concrete_keyset_path}'")
        os.remove(concrete_keyset_path)

    if secret_key:
        print(f"partial keygen from sk at '{secret_key}'")
        # load the initial secret key to use for keygen
        with open(
            secret_key,
            "rb",
        ) as f:
            buff = f.read()
        input_idx_to_key = {0: buff, 1: buff}
        tfhers_bridge.keygen_with_initial_keys(input_idx_to_key_buffer=input_idx_to_key)
    else:
        print("full keygen")
        circuit.keygen()

    print(f"saving Concrete keyset")
    circuit.client.keys.save(concrete_keyset_path)
    print(f"saved Concrete keyset to '{concrete_keyset_path}'")

    sk: bytes = tfhers_bridge.serialize_input_secret_key(input_idx=0)
    print(f"writing secret key of size {len(sk)} to '{output_secret_key}'")
    with open(output_secret_key, "wb") as f:
        f.write(sk)


@cli.command()
@click.option("-c1", "--rust-ct-1", type=str, required=True)
@click.option("-c2", "--rust-ct-2", type=str, required=True)
@click.option("-o", "--output-rust-ct", type=str, required=False)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
def run(rust_ct_1: str, rust_ct_2: str, output_rust_ct: str, concrete_keyset_path: str):
    """Run circuit"""
    circuit, tfhers_bridge = ccompilee()

    if not os.path.exists(concrete_keyset_path):
        raise RuntimeError("cannot find keys, you should run keygen before")
    print(f"loading keys from '{concrete_keyset_path}'")
    circuit.client.keys.load(concrete_keyset_path)

    # read tfhers int from file
    with open(rust_ct_1, "rb") as f:
        buff = f.read()
    # import fheuint8 and get its description
    tfhers_uint8_x = tfhers_bridge.import_value(buff, input_idx=0)

    # read tfhers int from file
    with open(rust_ct_2, "rb") as f:
        buff = f.read()
    # import fheuint8 and get its description
    tfhers_uint8_y = tfhers_bridge.import_value(buff, input_idx=1)

    encrypted_x, encrypted_y = tfhers_uint8_x, tfhers_uint8_y

    print(f"Homomorphic evaluation...")
    encrypted_result = circuit.run(encrypted_x, encrypted_y)

    if output_rust_ct:
        print("exporting Rust ciphertexts")
        # export fheuint8
        buff = tfhers_bridge.export_value(encrypted_result, output_idx=0)
        # write it to file
        with open(output_rust_ct, "wb") as f:
            f.write(buff)
    else:
        result = circuit.decrypt(encrypted_result)
        decoded = tfhers_type.decode(result)
        print(f"Concrete decryption result: raw({result}), decoded({decoded})")


if __name__ == "__main__":
    cli()
