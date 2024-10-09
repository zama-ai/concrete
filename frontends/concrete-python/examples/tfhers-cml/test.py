import os
from functools import partial

import click
import numpy as np

from concrete import fhe
from concrete.fhe import tfhers

from numpy.random import randint

# FIXME: should we move this to Concrete library directly, hidden to the user
def get_tfhers_params_and_type_and_int(precision):
    tfhers_params = tfhers.CryptoParams(
        lwe_dimension=LWE_DIM,
        glwe_dimension=GLWE_DIM,
        polynomial_size=POLY_SIZE,
        pbs_base_log=PBS_BASE_LOG,
        pbs_level=PBS_LEVEL,
        lwe_noise_distribution=LWE_NOISE_DISTR,
        glwe_noise_distribution=GLWE_NOISE_DISTR,
        encryption_key_choice=ENCRYPTION_KEY_CHOICE,
    )
    tfhers_type = tfhers.TFHERSIntegerType(
        is_signed=False,
        bit_width=precision,
        carry_width=CARRY_WIDTH,
        msg_width=MSG_WIDTH,
        params=tfhers_params,
    )
    tfhers_int = partial(tfhers.TFHERSInteger, tfhers_type)

    return tfhers_params, tfhers_type, tfhers_int

# FIXME Params: users shouldn't change them, should we hide it
LWE_DIM = 909
GLWE_DIM = 1
POLY_SIZE = 4096
PBS_BASE_LOG = 15
PBS_LEVEL = 2
MSG_WIDTH = 2
CARRY_WIDTH = 3
ENCRYPTION_KEY_CHOICE = tfhers.EncryptionKeyChoice.BIG
LWE_NOISE_DISTR = 0
GLWE_NOISE_DISTR = 2.168404344971009e-19

assert GLWE_DIM == 1, "glwe dim must be 1"

# Options: the user can change the following
# FIXME: explain FHEUINT_PRECISION, ie can it be changed
FHEUINT_PRECISION = 8
tfhers_params, tfhers_type, tfhers_int = get_tfhers_params_and_type_and_int(FHEUINT_PRECISION)

# Describe the function you want to apply, on Concrete ciphertexts
def server_side_function_in_concrete(concrete_x, concrete_y, concrete_z):
    return (((concrete_x + concrete_y) % 47) + (2 * concrete_z) % 47) % 47

# The user must specify the range of the TFHE-rs inputs
# FIXME: why can't we set the limit at 256? It's needed for FHEUint8
inputset_of_tfhe_rs_inputs = [(tfhers_int(randint(128)),
                               tfhers_int(randint(128)),
                               tfhers_int(randint(128))) for _ in range(100)]

# End of options

# This is the compiled function: user doesn't have to change this, except to
# add more inputs (ie, tfhers_z etc)
def function_to_run_in_concrete(tfhers_x, tfhers_y, tfhers_z):

    # Here, tfhers_x and tfhers_y are in TFHE-rs format

    concrete_x = tfhers.to_native(tfhers_x)
    concrete_y = tfhers.to_native(tfhers_y)
    concrete_z = tfhers.to_native(tfhers_z)

    # Here, concrete_'s variables are in Concrete format

    # Here we can apply whatever function we want in Concrete
    concrete_res = server_side_function_in_concrete(concrete_x, concrete_y, concrete_z)

    # Here, concrete_res is in Concrete format

    tfhers_res = tfhers.from_native(
        concrete_res, tfhers_type
    )  # we have to specify the type we want to convert to


    # Here, tfhers_res is in TFHE-rs format

    return tfhers_res

# This is where we compile the function with Concrete: user doesn't have to
# change this, except to add more inputs (ie, tfhers_z etc)
def compile_concrete_function():
    compiler = fhe.Compiler(function_to_run_in_concrete,
        {"tfhers_x": "encrypted",
         "tfhers_y": "encrypted",
         "tfhers_z": "encrypted"})

    circuit = compiler.compile(inputset_of_tfhe_rs_inputs)

    tfhers_bridge = tfhers.new_bridge(circuit=circuit)
    return circuit, tfhers_bridge


@click.group()
def cli():
    pass

def read_var_from_file(tfhers_bridge, filename, input_idx):
    with open(filename, "rb") as f:
        buff = f.read()
    return tfhers_bridge.import_value(buff, input_idx=input_idx)


@cli.command()
@click.option("-s", "--secret-key", type=str, required=True)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
# This is where we generate the evaluation key at the Concrete format, from the
# secret key coming from TFHE-rs, on the client side
def keygen(secret_key: str, concrete_keyset_path: str):
    """Concrete Key Generation"""

    # Compile the Concrete function
    circuit, tfhers_bridge = compile_concrete_function()

    if os.path.exists(concrete_keyset_path):
        os.remove(concrete_keyset_path)

    # Load the initial secret key to use for keygen
    with open(
        secret_key,
        "rb",
    ) as f:
        buff = f.read()

    input_idx_to_key = {0: buff, 1: buff}
    tfhers_bridge.keygen_with_initial_keys(input_idx_to_key_buffer=input_idx_to_key)

    # FIXME: remove the secret key before saving. The secret key can be used for
    # debugging but should really be removed in production
    circuit.client.keys.save(concrete_keyset_path)


@cli.command()
@click.option("-c1", "--rust-ct-1", type=str, required=True)
@click.option("-c2", "--rust-ct-2", type=str, required=True)
@click.option("-c3", "--rust-ct-3", type=str, required=True)
@click.option("-o", "--output-rust-ct", type=str, required=True)
@click.option("-k", "--concrete-keyset-path", type=str, required=True)
# This is the actual FHE computation, on the server side
def run(rust_ct_1: str, rust_ct_2: str, rust_ct_3: str, output_rust_ct: str, concrete_keyset_path: str):
    """Run circuit"""
    circuit, tfhers_bridge = compile_concrete_function()

    if not os.path.exists(concrete_keyset_path):
        raise RuntimeError("cannot find keys, you should run keygen before")

    circuit.client.keys.load(concrete_keyset_path)

    tfhers_uint8_x = read_var_from_file(tfhers_bridge, rust_ct_1, input_idx=0)
    tfhers_uint8_y = read_var_from_file(tfhers_bridge, rust_ct_2, input_idx=1)
    tfhers_uint8_z = read_var_from_file(tfhers_bridge, rust_ct_3, input_idx=2)

    encrypted_result = circuit.run(tfhers_uint8_x, tfhers_uint8_y, tfhers_uint8_z)

    # export fheuint8
    buff = tfhers_bridge.export_value(encrypted_result, output_idx=0)
    # write it to file
    with open(output_rust_ct, "wb") as f:
        f.write(buff)

    # BCM BEGIN: to debug computations
    # FIXME: how does it decrypt? we are on the server side, we shouldn't have
    # the secret key. I think it's because the secret key is saved in concrete_keyset_path

    # x = circuit.decrypt(tfhers_uint8_x)
    # decoded = tfhers_type.decode(x)
    # print(f"Concrete decryption result: raw({x}), decoded({decoded})")

    # y = circuit.decrypt(tfhers_uint8_y)
    # decoded = tfhers_type.decode(y)
    # print(f"Concrete decryption result: raw({y}), decoded({decoded})")

    # result = circuit.decrypt(encrypted_result)
    # decoded = tfhers_type.decode(result)
    # print(f"Concrete decryption result: raw({result}), decoded({decoded})")
    # BCM END


if __name__ == "__main__":
    cli()