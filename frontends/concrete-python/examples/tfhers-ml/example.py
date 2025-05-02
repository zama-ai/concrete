import os
import typing
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

#### Model Parameters ##################
q_weights = np.array([[-25, 21, -10], [42, -20, -37], [-128, -15, 127], [-58, -51, 94]])
q_bias = np.array([[35167, 9417, -44584]])
weight_quantizer_zero_point = -5
########################################

rounder = fhe.AutoRounder(target_msbs=8)  # We want to keep 8 MSBs


@typing.no_type_check
def ml_inference(input_x: np.ndarray) -> np.ndarray:
    y_pred = input_x @ q_weights - weight_quantizer_zero_point * np.sum(
        input_x, axis=1, keepdims=True
    )
    y_pred += q_bias
    y_pred = fhe.round_bit_pattern(y_pred, rounder)
    y_pred = y_pred >> rounder.lsbs_to_remove
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
                        [36, -17, -85, -124],
                        [29, -33, -85, -124],
                        [23, -26, -88, -124],
                        [19, -30, -82, -124],
                        [32, -13, -85, -124],
                    ]
                )
            ),
        )
    ]

    # Add the auto-adjustment before compilation
    fhe.AutoRounder.adjust(compute, inputset)

    # Print the number of bits rounded
    print(f"lsbs_to_remove: {rounder.lsbs_to_remove}")

    circuit = compiler.compile(inputset)

    tfhers_bridge = tfhers.new_bridge(circuit)
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

    print("saving Concrete Evaluation Keys")
    with open(concrete_keyset_path, "wb") as f:
        f.write(circuit.client.evaluation_keys.serialize())
    print(f"saved Concrete Evaluation Keys to '{concrete_keyset_path}'")

    sk: bytes = tfhers_bridge.serialize_input_secret_key(input_idx=0)
    print(f"writing secret key of size {len(sk)} to '{output_secret_key}'")
    with open(output_secret_key, "wb") as f:
        f.write(sk)


@cli.command()
@click.option("-c", "--rust-ct", type=str, required=True)
@click.option("-o", "--output-rust-ct", type=str, required=True)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
def run(rust_ct: str, output_rust_ct: str, concrete_keyset_path: str):
    """Run circuit"""
    circuit, tfhers_bridge = ccompilee()

    if not os.path.exists(concrete_keyset_path):
        msg = "cannot find keys, you should run keygen before"
        raise RuntimeError(msg)
    print(f"loading keys from '{concrete_keyset_path}'")
    with open(concrete_keyset_path, "rb") as f:
        eval_keys = fhe.EvaluationKeys.deserialize(f.read())

    # read tfhers int from file
    with open(rust_ct, "rb") as f:
        buff = f.read()
    # import fheuint8 and get its description
    tfhers_uint8_x = tfhers_bridge.import_value(buff, input_idx=0)

    print("Homomorphic evaluation...")
    encrypted_result = circuit.server.run(tfhers_uint8_x, evaluation_keys=eval_keys)

    print("exporting Rust ciphertexts")
    # export fheuint8
    buff = tfhers_bridge.export_value(encrypted_result, output_idx=0)
    # write it to file
    with open(output_rust_ct, "wb") as f:
        f.write(buff)


if __name__ == "__main__":
    cli()
