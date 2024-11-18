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
IS_SIGNED = True
#######################################

tfhers_type = tfhers.get_type_from_params(
    TFHERS_PARAMS_FILE,
    is_signed=IS_SIGNED,
    precision=FHEUINT_PRECISION,
)
tfhers_int = partial(tfhers.TFHERSInteger, tfhers_type)

rounder = fhe.AutoRounder(target_msbs=8)  # We want to keep 8 MSBs

q_weights = np.array(
    [
        [-81],
        [-95],
        [-51],
        [-77],
        [-64],
        [-64],
        [-128],
        [127],
        [-122],
        [-81],
        [-96],
        [-93],
        [-63],
        [-50],
        [-104],
        [-99],
        [-112],
        [-46],
        [-106],
        [-42],
        [-56],
        [-46],
        [-67],
        [-116],
        [-107],
        [-75],
        [-105],
        [-109],
        [-88],
        [-80],
    ]
)
q_weights = np.ones_like(q_weights)
q_bias = np.array([[680]])
weight_quantizer_zero_point = -74


def ml_inference(q_X: np.ndarray) -> np.ndarray:
    # Quantizing weights and inputs makes an additional term appear in the inference function
    y_pred = q_X @ q_weights - weight_quantizer_zero_point * np.sum(q_X, axis=1, keepdims=True)
    y_pred += q_bias
    y_pred = fhe.round_bit_pattern(y_pred, rounder)
    y_pred = (y_pred >> rounder.lsbs_to_remove)
    return y_pred


def compute(tfhers_x):
    ####### TFHE-rs to Concrete #########

    # x and y are supposed to be TFHE-rs values.
    # to_native will use type information from x and y to do
    # a correct conversion from TFHE-rs to Concrete
    concrete_x = tfhers.to_native(tfhers_x)
    ####### TFHE-rs to Concrete #########

    ####### Concrete Computation ########
    concrete_res = ml_inference(concrete_x)
    ####### Concrete Computation ########

    ####### Concrete to TFHE-rs #########
    tfhers_res = tfhers.from_native(
        concrete_res, tfhers_type
    )  # we have to specify the type we want to convert to
    ####### Concrete to TFHE-rs #########
    return tfhers_res


def ccompilee():
    compiler = fhe.Compiler(
        compute,
        {
            "tfhers_x": "encrypted",
        },
    )

    inputset = [
        (
            tfhers_int(
                np.array(
                    [
                        1,
                    ]
                    * 30
                ).reshape((1, 30))
            ),
        )
    ]

    # Add the auto-adjustment before compilation
    fhe.AutoRounder.adjust(compute, inputset)

    circuit = compiler.compile(inputset, show_graph=True, show_mlir=True)

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
        input_idx_to_key = {0: buff}
        tfhers_bridge.keygen_with_initial_keys(input_idx_to_key_buffer=input_idx_to_key)
    else:
        print("full keygen")
        circuit.keygen()

    print("saving Concrete keyset")
    circuit.client.keys.save(concrete_keyset_path)
    print(f"saved Concrete keyset to '{concrete_keyset_path}'")

    sk: bytes = tfhers_bridge.serialize_input_secret_key(input_idx=0)
    print(f"writing secret key of size {len(sk)} to '{output_secret_key}'")
    with open(output_secret_key, "wb") as f:
        f.write(sk)


@cli.command()
@click.option("-c1", "--rust-ct-1", type=str, required=True)
@click.option("-o", "--output-rust-ct", type=str, required=False)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
def run(rust_ct_1: str, output_rust_ct: str, concrete_keyset_path: str):
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

    print("Homomorphic evaluation...")
    encrypted_result = circuit.run(tfhers_uint8_x)

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
