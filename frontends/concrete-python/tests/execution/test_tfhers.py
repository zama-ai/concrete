"""
Tests execution of tfhers conversion operations.
"""

import json
import os
import tempfile
from typing import List

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe import tfhers


def binary_tfhers(x, y, binary_op, tfhers_type):
    """wrap binary op in tfhers conversion (2 tfhers inputs)"""
    x = tfhers.to_native(x)
    y = tfhers.to_native(y)
    return tfhers.from_native(binary_op(x, y), tfhers_type)


def one_tfhers_one_native(x, y, binary_op, tfhers_type):
    """wrap binary op in tfhers conversion (1 tfhers, 1 native input)"""
    x = tfhers.to_native(x)
    return tfhers.from_native(binary_op(x, y), tfhers_type)


def parameterize_partial_dtype(partial_dtype) -> tfhers.TFHERSIntegerType:
    """Create a tfhers type from a partial func missing tfhers params.

    Args:
        partial_dtype (callable): partial function to create dtype (missing params)

    Returns:
        tfhers.TFHERSIntegerType: tfhers type
    """
    tfhers_params = tfhers.CryptoParams(
        909,
        1,
        4096,
        15,
        2,
        0,
        2.168404344971009e-19,
        tfhers.EncryptionKeyChoice.BIG,
    )
    return partial_dtype(tfhers_params)


def is_input_and_output_tfhers(
    circuit: fhe.Circuit,
    lwe_dim: int,
    tfhers_ins: List[int],
    tfhers_outs: List[int],
) -> bool:
    """Check if inputs and outputs description match tfhers parameters"""
    params = json.loads(circuit.client.specs.serialize())
    main_circuit = params["circuits"][0]
    # check all encrypted input/output have the correct lwe_dim
    ins = main_circuit["inputs"]
    outs = main_circuit["outputs"]
    for indices, param in [(tfhers_ins, ins), (tfhers_outs, outs)]:
        for i in indices:
            if param[i]["rawInfo"]["shape"]["dimensions"][-1] != lwe_dim + 1:
                return False
    return True


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            tfhers.uint8_2_2,
            id="x * y",
        ),
    ],
)
def test_tfhers_conversion_binary_encrypted(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (2 tfhers inputs).
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [0, 1],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)
    encoded_sample = (dtype.encode(v) for v in sample)
    encoded_result = circuit.encrypt_run_decrypt(*encoded_sample)

    assert (dtype.decode(encoded_result) == function(*sample)).all()


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x + clear(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x + clear(y) big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x - clear(y)",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            tfhers.uint8_2_2,
            id="x * y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "clear"},
            },
            tfhers.uint8_2_2,
            id="x * clear(y)",
        ),
    ],
)
def test_tfhers_conversion_one_tfhers_one_native(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (1 tfhers, 1 native input).
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: one_tfhers_one_native(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        (tfhers.TFHERSInteger(dtype, inpt[0]), inpt[1])
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [
            0,
        ],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)
    encoded_sample = (dtype.encode(sample[0]), sample[1])
    encoded_result = circuit.encrypt_run_decrypt(*encoded_sample)

    assert (dtype.decode(encoded_result) == function(*sample)).all()


def lut_add_lut(x, y):
    """lut add lut compute"""
    lut = fhe.LookupTable(list(range(256)))
    x = lut[x]
    y = lut[y]
    return lut[x + y]


TFHERS_UINT_8_3_2_4096 = tfhers.TFHERSIntegerType(
    False,
    bit_width=8,
    carry_width=3,
    msg_width=2,
    params=tfhers.CryptoParams(
        lwe_dimension=909,
        glwe_dimension=1,
        polynomial_size=4096,
        pbs_base_log=15,
        pbs_level=2,
        lwe_noise_distribution=0,
        glwe_noise_distribution=2.168404344971009e-19,
        encryption_key_choice=tfhers.EncryptionKeyChoice.BIG,
    ),
)

TFHERS_INT_8_3_2_4096 = tfhers.TFHERSIntegerType(
    True,
    bit_width=8,
    carry_width=3,
    msg_width=2,
    params=tfhers.CryptoParams(
        lwe_dimension=909,
        glwe_dimension=1,
        polynomial_size=4096,
        pbs_base_log=15,
        pbs_level=2,
        lwe_noise_distribution=0,
        glwe_noise_distribution=2.168404344971009e-19,
        encryption_key_choice=tfhers.EncryptionKeyChoice.BIG,
    ),
)


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**7 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**7 - 1], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [-(2**6), -2], "status": "encrypted"},
                "y": {"range": [0, 2**6 - 1], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) + signed(y)",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**7 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**4 - 1], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [-(2**3), -2], "status": "encrypted"},
                "y": {"range": [-(2**3), -2], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) - signed(y)",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**3 - 1], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x * y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [-(2**3), 2**2], "status": "encrypted"},
                "y": {"range": [-(2**2), 2**3], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) * signed(y)",
        ),
        pytest.param(
            lut_add_lut,
            {
                "x": {"range": [0, 2**7 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**7 - 1], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="lut_add_lut",
        ),
    ],
)
def test_tfhers_binary_encrypted_complete_circuit_concrete_keygen(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (2 tfhers inputs).

    Encryption/decryption are done in Rust using TFHErs, while Keygen is done in Concrete.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [0, 1],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)

    ###### Full Concrete Execution ################################################
    concrete_encoded_sample = (dtype.encode(v) for v in sample)
    concrete_encoded_result = circuit.encrypt_run_decrypt(*concrete_encoded_sample)
    assert (dtype.decode(concrete_encoded_result) == function(*sample)).all()

    ###### TFHErs Encryption & Computation ########################################
    tfhers_bridge = tfhers.new_bridge(circuit)

    # serialize key
    _, key_path = tempfile.mkstemp()
    serialized_key = tfhers_bridge.serialize_input_secret_key(input_idx=0)
    with open(key_path, "wb") as f:
        f.write(serialized_key)

    ct1, ct2 = sample
    _, ct1_path = tempfile.mkstemp()
    _, ct2_path = tempfile.mkstemp()
    _, ct_one_path = tempfile.mkstemp()
    _, client_key_path = tempfile.mkstemp()
    _, server_key_path = tempfile.mkstemp()

    tfhers_utils = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../tfhers-utils/target/release/tfhers_utils"
    )

    # keygen starting from Concrete's secret key
    assert (
        os.system(
            f"{tfhers_utils} keygen --lwe-sk {key_path} -c {client_key_path} -s {server_key_path}"
        )
        == 0
    )

    # encrypt inputs and incremnt them by one in TFHErs
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key {'--signed' if dtype.is_signed else ''} --value=1 -c {ct_one_path} --client-key {client_key_path}"
        )
        == 0
    )
    sample = [s + 1 for s in sample]
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key {'--signed' if dtype.is_signed else ''} --value={ct1} -c {ct1_path} --client-key {client_key_path}"
        )
        == 0
    )
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key {'--signed' if dtype.is_signed else ''} --value={ct2} -c {ct2_path} --client-key {client_key_path}"
        )
        == 0
    )
    assert (
        os.system(
            f"{tfhers_utils} add {'--signed' if dtype.is_signed else ''} -c {ct1_path} {ct_one_path} -s {server_key_path} -o {ct1_path}"
        )
        == 0
    )
    assert (
        os.system(
            f"{tfhers_utils} add {'--signed' if dtype.is_signed else ''} -c {ct2_path} {ct_one_path} -s {server_key_path} -o {ct2_path}"
        )
        == 0
    )

    # import ciphertexts and run
    cts = []
    with open(ct1_path, "rb") as f:
        buff = f.read()
        cts.append(tfhers_bridge.import_value(buff, 0))
    with open(ct2_path, "rb") as f:
        buff = f.read()
        cts.append(tfhers_bridge.import_value(buff, 1))
    os.remove(ct1_path)
    os.remove(ct2_path)

    tfhers_encrypted_result = circuit.run(*cts)

    # concrete decryption should work
    decrypted = circuit.decrypt(tfhers_encrypted_result)
    assert (dtype.decode(decrypted) == function(*sample)).all()  # type: ignore

    # tfhers decryption
    buff = tfhers_bridge.export_value(tfhers_encrypted_result, output_idx=0)  # type: ignore
    _, ct_out_path = tempfile.mkstemp()
    _, pt_path = tempfile.mkstemp()
    with open(ct_out_path, "wb") as f:
        f.write(buff)

    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key"
            f"{' --signed ' if dtype.is_signed else ''}"
            f" -c {ct_out_path} --lwe-sk {key_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        result = int(f.read())

    # close remaining tempfiles
    os.remove(key_path)
    os.remove(ct_out_path)
    os.remove(pt_path)
    os.remove(ct_one_path)
    os.remove(client_key_path)
    os.remove(server_key_path)

    assert result == function(*sample)


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**7], "status": "encrypted"},
                "y": {"range": [0, 2**7], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [-(2**6), -2], "status": "encrypted"},
                "y": {"range": [0, 2**6 - 1], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(-x) + signed(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**6 - 1], "status": "encrypted"},
                "y": {"range": [-(2**6), -2], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) + signed(-y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**7], "status": "encrypted"},
                "y": {"range": [0, 2**7], "status": "clear"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x + clear(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [-(2**6), -2], "status": "encrypted"},
                "y": {"range": [0, 2**6 - 1], "status": "clear"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(-x) + clear(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**6 - 1], "status": "encrypted"},
                "y": {"range": [-(2**6), -2], "status": "clear"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) + clear(-y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**6, 2**7 - 1], "status": "encrypted"},
                "y": {"range": [2**6, 2**7 - 1], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**6, 2**7 - 1], "status": "encrypted"},
                "y": {"range": [2**6, 2**7 - 1], "status": "clear"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x + clear(y) big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**8 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**4], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [-(2**3), -2], "status": "encrypted"},
                "y": {"range": [-(2**3), -2], "status": "encrypted"},
            },
            TFHERS_INT_8_3_2_4096,
            id="signed(x) - signed(y)",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**8 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**4], "status": "clear"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x - clear(y)",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x * y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "clear"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="x * clear(y)",
        ),
        pytest.param(
            lut_add_lut,
            {
                "x": {"range": [0, 2**7], "status": "encrypted"},
                "y": {"range": [0, 2**7], "status": "encrypted"},
            },
            TFHERS_UINT_8_3_2_4096,
            id="lut_add_lut(x , y)",
        ),
    ],
)
def test_tfhers_one_tfhers_one_native_complete_circuit_concrete_keygen(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (1 tfhers, 1 native).

    Encryption/decryption are done in Rust using TFHErs, while Keygen is done in Concrete.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    compiler = fhe.Compiler(
        lambda x, y: one_tfhers_one_native(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        (tfhers.TFHERSInteger(dtype, inpt[0]), inpt[1])
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [
            0,
        ],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)

    ###### Full Concrete Execution ################################################
    concrete_encoded_result = circuit.encrypt_run_decrypt(dtype.encode(sample[0]), sample[1])
    assert (dtype.decode(concrete_encoded_result) == function(*sample)).all()

    ###### TFHErs Encryption ######################################################
    tfhers_bridge = tfhers.new_bridge(circuit)

    # serialize key
    _, key_path = tempfile.mkstemp()
    serialized_key = tfhers_bridge.serialize_input_secret_key(input_idx=0)
    with open(key_path, "wb") as f:
        f.write(serialized_key)

    # encrypt first input
    ct1, _ = sample
    _, ct1_path = tempfile.mkstemp()

    tfhers_utils = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../tfhers-utils/target/release/tfhers_utils"
    )
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key {'--signed' if dtype.is_signed else ''} --value={ct1} -c {ct1_path} --lwe-sk {key_path}"
        )
        == 0
    )

    # import first ciphertexts and encrypt second with concrete
    with open(ct1_path, "rb") as f:
        buff = f.read()
        tfhers_ct = tfhers_bridge.import_value(buff, 0)
    os.remove(ct1_path)

    _, native_ct = circuit.encrypt(None, sample[1])  # type: ignore

    tfhers_encrypted_result = circuit.run(tfhers_ct, native_ct)

    # concrete decryption should work
    decrypted = circuit.decrypt(tfhers_encrypted_result)
    assert (dtype.decode(decrypted) == function(*sample)).all()  # type: ignore

    # tfhers decryption
    buff = tfhers_bridge.export_value(tfhers_encrypted_result, output_idx=0)  # type: ignore
    _, ct_out_path = tempfile.mkstemp()
    _, pt_path = tempfile.mkstemp()
    with open(ct_out_path, "wb") as f:
        f.write(buff)

    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key"
            f"{' --signed ' if dtype.is_signed else ''}"
            f" -c {ct_out_path} --lwe-sk {key_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        result = int(f.read())

    # close remaining tempfiles
    os.remove(key_path)
    os.remove(ct_out_path)
    os.remove(pt_path)

    assert result == function(*sample)


@pytest.mark.parametrize(
    "function, parameters, tfhers_value_range, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**6], "status": "encrypted"},
                "y": {"range": [0, 2**6], "status": "encrypted"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**7], "status": "encrypted"},
                "y": {"range": [0, 2**4], "status": "encrypted"},
            },
            [0, 2**3],
            TFHERS_UINT_8_3_2_4096,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            [0, 2**2],
            TFHERS_UINT_8_3_2_4096,
            id="x * y",
        ),
        pytest.param(
            lut_add_lut,
            {
                "x": {"range": [0, 2**6], "status": "encrypted"},
                "y": {"range": [0, 2**6], "status": "encrypted"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="lut_add_lut",
        ),
    ],
)
def test_tfhers_binary_encrypted_complete_circuit_tfhers_keygen(
    function, parameters, tfhers_value_range, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (2 tfhers inputs).

    Encryption/decryption are done in Rust using TFHErs, while Keygen is done in Concrete.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # there is no point of using the cache here as new keys will be generated everytime
    config = helpers.configuration().fork(
        use_insecure_key_cache=False, insecure_key_cache_location=None
    )

    # Only valid when running in multi
    if config.parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, config)

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [0, 1],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)

    ###### TFHErs Keygen ##########################################################
    _, client_key_path = tempfile.mkstemp()
    _, server_key_path = tempfile.mkstemp()
    _, sk_path = tempfile.mkstemp()

    tfhers_utils = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../tfhers-utils/target/release/tfhers_utils"
    )

    assert (
        os.system(
            f"{tfhers_utils} keygen -s {server_key_path} -c {client_key_path} --output-lwe-sk {sk_path}"
        )
        == 0
    )

    ###### Concrete Keygen ########################################################
    tfhers_bridge = tfhers.new_bridge(circuit)

    with open(sk_path, "rb") as f:
        sk_buff = f.read()

    # set sk for input 0 and generate the remaining keys
    tfhers_bridge.keygen_with_initial_keys({0: sk_buff}, force=True)

    ###### Full Concrete Execution ################################################
    concrete_encoded_sample = (dtype.encode(v) for v in sample)
    concrete_encoded_result = circuit.encrypt_run_decrypt(*concrete_encoded_sample)
    assert (dtype.decode(concrete_encoded_result) == function(*sample)).all()

    ###### TFHErs Encryption ######################################################

    # encrypt inputs
    ct1, ct2 = sample
    _, ct1_path = tempfile.mkstemp()
    _, ct2_path = tempfile.mkstemp()

    tfhers_utils = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../tfhers-utils/target/release/tfhers_utils"
    )
    assert (
        os.system(f"{tfhers_utils} encrypt-with-key --value={ct1} -c {ct1_path} --lwe-sk {sk_path}")
        == 0
    )
    assert (
        os.system(f"{tfhers_utils} encrypt-with-key --value={ct2} -c {ct2_path} --lwe-sk {sk_path}")
        == 0
    )

    # import ciphertexts and run
    cts = []
    with open(ct1_path, "rb") as f:
        buff = f.read()
        cts.append(tfhers_bridge.import_value(buff, 0))
    with open(ct2_path, "rb") as f:
        buff = f.read()
        cts.append(tfhers_bridge.import_value(buff, 1))
    os.remove(ct1_path)
    os.remove(ct2_path)

    tfhers_encrypted_result = circuit.run(*cts)

    # concrete decryption should work
    decrypted = circuit.decrypt(tfhers_encrypted_result)
    assert (dtype.decode(decrypted) == function(*sample)).all()  # type: ignore

    # tfhers decryption
    buff = tfhers_bridge.export_value(tfhers_encrypted_result, output_idx=0)  # type: ignore
    _, ct_out_path = tempfile.mkstemp()
    _, pt_path = tempfile.mkstemp()
    with open(ct_out_path, "wb") as f:
        f.write(buff)

    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key" f" -c {ct_out_path} --lwe-sk {sk_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        result = int(f.read())
    assert result == function(*sample)

    ###### Compute with TFHErs ####################################################
    _, random_ct_path = tempfile.mkstemp()
    _, sum_ct_path = tempfile.mkstemp()

    # encrypt random value
    random_value = np.random.randint(*tfhers_value_range)
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key --value={random_value} -c {random_ct_path} --client-key {client_key_path}"
        )
        == 0
    )

    # add random value to the result ct
    assert (
        os.system(
            f"{tfhers_utils} add -c {ct_out_path} {random_ct_path} -s {server_key_path} -o {sum_ct_path}"
        )
        == 0
    )

    # decrypt result
    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key -c {sum_ct_path} --lwe-sk {sk_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        tfhers_result = int(f.read())
    assert result + random_value == tfhers_result

    # close remaining tempfiles
    os.remove(client_key_path)
    os.remove(server_key_path)
    os.remove(sk_path)
    os.remove(ct_out_path)
    os.remove(pt_path)
    os.remove(random_ct_path)
    os.remove(sum_ct_path)


@pytest.mark.parametrize(
    "function, parameters, tfhers_value_range, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**6], "status": "encrypted"},
                "y": {"range": [0, 2**6], "status": "encrypted"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**6], "status": "encrypted"},
                "y": {"range": [0, 2**6], "status": "clear"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="x + clear(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**5, 2**6], "status": "encrypted"},
                "y": {"range": [2**5, 2**6], "status": "encrypted"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**5, 2**6], "status": "encrypted"},
                "y": {"range": [2**5, 2**6], "status": "clear"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="x + clear(y) big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**8 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**4], "status": "encrypted"},
            },
            [0, 2**3],
            TFHERS_UINT_8_3_2_4096,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**4, 2**8 - 1], "status": "encrypted"},
                "y": {"range": [0, 2**4], "status": "clear"},
            },
            [0, 2**3],
            TFHERS_UINT_8_3_2_4096,
            id="x - clear(y)",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            [0, 2**2],
            TFHERS_UINT_8_3_2_4096,
            id="x * y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "clear"},
            },
            [0, 2**2],
            TFHERS_UINT_8_3_2_4096,
            id="x * clear(y)",
        ),
        pytest.param(
            lut_add_lut,
            {
                "x": {"range": [0, 2**6], "status": "encrypted"},
                "y": {"range": [0, 2**6], "status": "encrypted"},
            },
            [0, 2**6],
            TFHERS_UINT_8_3_2_4096,
            id="lut_add_lut(x , y)",
        ),
    ],
)
def test_tfhers_one_tfhers_one_native_complete_circuit_tfhers_keygen(
    function, parameters, tfhers_value_range, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (1 tfhers, 1 native).

    Keygen, Encryption/decryption are done in Rust using TFHErs.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # there is no point of using the cache here as new keys will be generated everytime
    config = helpers.configuration().fork(
        use_insecure_key_cache=False, insecure_key_cache_location=None
    )

    # Only valid when running in multi
    if config.parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    compiler = fhe.Compiler(
        lambda x, y: one_tfhers_one_native(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        (tfhers.TFHERSInteger(dtype, inpt[0]), inpt[1])
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, config)

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [
            0,
        ],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)

    ###### TFHErs Keygen ##########################################################
    _, client_key_path = tempfile.mkstemp()
    _, server_key_path = tempfile.mkstemp()
    _, sk_path = tempfile.mkstemp()

    tfhers_utils = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../tfhers-utils/target/release/tfhers_utils"
    )

    assert (
        os.system(
            f"{tfhers_utils} keygen -s {server_key_path} -c {client_key_path} --output-lwe-sk {sk_path}"
        )
        == 0
    )

    ###### Concrete Keygen ########################################################
    tfhers_bridge = tfhers.new_bridge(circuit)

    with open(sk_path, "rb") as f:
        sk_buff = f.read()

    # set sk for input 0 and generate the remaining keys
    tfhers_bridge.keygen_with_initial_keys({0: sk_buff}, force=True)

    ###### Full Concrete Execution ################################################
    concrete_encoded_result = circuit.encrypt_run_decrypt(dtype.encode(sample[0]), sample[1])
    assert (dtype.decode(concrete_encoded_result) == function(*sample)).all()

    ###### TFHErs Encryption ######################################################

    # encrypt first input
    pt1, _ = sample
    _, ct1_path = tempfile.mkstemp()

    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key --value={pt1} -c {ct1_path} --client-key {client_key_path}"
        )
        == 0
    )

    # import first ciphertexts and encrypt second with concrete
    with open(ct1_path, "rb") as f:
        buff = f.read()
        tfhers_ct = tfhers_bridge.import_value(buff, 0)
    os.remove(ct1_path)

    _, native_ct = circuit.encrypt(None, sample[1])  # type: ignore

    tfhers_encrypted_result = circuit.run(tfhers_ct, native_ct)

    # concrete decryption should work
    decrypted = circuit.decrypt(tfhers_encrypted_result)
    assert (dtype.decode(decrypted) == function(*sample)).all()  # type: ignore

    # tfhers decryption
    buff = tfhers_bridge.export_value(tfhers_encrypted_result, output_idx=0)  # type: ignore
    _, ct_out_path = tempfile.mkstemp()
    _, pt_path = tempfile.mkstemp()
    with open(ct_out_path, "wb") as f:
        f.write(buff)

    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key -c {ct_out_path} --lwe-sk {sk_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        result = int(f.read())
    assert result == function(*sample)

    ###### Compute with TFHErs ####################################################
    _, random_ct_path = tempfile.mkstemp()
    _, sum_ct_path = tempfile.mkstemp()

    # encrypt random value
    random_value = np.random.randint(*tfhers_value_range)
    assert (
        os.system(
            f"{tfhers_utils} encrypt-with-key --value={random_value} -c {random_ct_path} --client-key {client_key_path}"
        )
        == 0
    )

    # add random value to the result ct
    assert (
        os.system(
            f"{tfhers_utils} add -c {ct_out_path} {random_ct_path} -s {server_key_path} -o {sum_ct_path}"
        )
        == 0
    )

    # decrypt result
    assert (
        os.system(
            f"{tfhers_utils} decrypt-with-key -c {sum_ct_path} --lwe-sk {sk_path} -p {pt_path}"
        )
        == 0
    )

    with open(pt_path, "r", encoding="utf-8") as f:
        tfhers_result = int(f.read())
    assert result + random_value == tfhers_result

    # close remaining tempfiles
    os.remove(client_key_path)
    os.remove(server_key_path)
    os.remove(sk_path)
    os.remove(ct_out_path)
    os.remove(pt_path)
    os.remove(random_ct_path)
    os.remove(sum_ct_path)


@pytest.mark.parametrize(
    "lhs,rhs,is_equal",
    [
        pytest.param(
            tfhers.uint16_2_2,
            tfhers.uint16_2_2,
            True,
        ),
        pytest.param(
            tfhers.uint16_2_2,
            tfhers.uint8_2_2,
            False,
        ),
    ],
)
def test_tfhers_integer_eq(lhs, rhs, is_equal):
    """
    Test TFHERSIntegerType equality.
    """
    assert is_equal == (parameterize_partial_dtype(lhs) == parameterize_partial_dtype(rhs))


@pytest.mark.parametrize(
    "dtype,value,encoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [10, 20, 30],
            [
                [2, 2, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [2, 3, 1, 0, 0, 0, 0, 0],
            ],
        ),
        pytest.param(
            tfhers.int8_2_2,
            [-128, 0, 127],
            [
                [0, 0, 0, 2],
                [0, 0, 0, 0],
                [3, 3, 3, 1],
            ],
        ),
    ],
)
def test_tfhers_integer_encode(dtype, value, encoded):
    """
    Test TFHERSIntegerType encode.
    """
    dtype = parameterize_partial_dtype(dtype)
    assert np.array_equal(dtype.encode(value), encoded)


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "can only encode int, np.integer, list or ndarray, but got <class 'str'>",
        ),
    ],
)
def test_tfhers_integer_bad_encode(dtype, value, expected_error, expected_message):
    """
    Test TFHERSIntegerType encode.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        dtype.encode(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,encoded,decoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [
                [2, 2, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [2, 3, 1, 0, 0, 0, 0, 0],
            ],
            [10, 20, 30],
        ),
        pytest.param(
            tfhers.int8_2_2,
            [
                [2, 1, 0, 2],
                [0, 3, 1, 0],
                [2, 1, 0, 1],
            ],
            [-122, 28, 70],
        ),
    ],
)
def test_tfhers_integer_decode(dtype, encoded, decoded):
    """
    Test TFHERSIntegerType decode.
    """

    dtype = parameterize_partial_dtype(dtype)
    assert np.array_equal(dtype.decode(encoded), decoded)


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "can only decode list of integers or ndarray of integers, but got <class 'str'>",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [1, 2, 3],
            ValueError,
            "expected the last dimension of encoded value to be 8 but it's 3",
        ),
    ],
)
def test_tfhers_integer_bad_decode(dtype, value, expected_error, expected_message):
    """
    Test TFHERSIntegerType decode.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        dtype.decode(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,value,encoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            10,
            10,
        ),
        pytest.param(
            tfhers.uint16_2_2,
            20,
            20,
        ),
    ],
)
def test_tfhers_from_native_outside_tracing(dtype, value, encoded):
    """
    Test tfhers.from_native outside tracing.
    """

    assert np.array_equal(tfhers.from_native(value, dtype), encoded)


@pytest.mark.parametrize(
    "value,decoded",
    [
        pytest.param(
            tfhers.TFHERSInteger(parameterize_partial_dtype(tfhers.uint8_2_2), [0, 0, 2, 2]),
            [0, 0, 2, 2],
        ),
    ],
)
def test_tfhers_to_native_outside_tracing(value, decoded):
    """
    Test tfhers.to_native outside tracing.
    """

    print(tfhers.to_native(value))
    assert np.array_equal(tfhers.to_native(value), decoded)


@pytest.mark.parametrize(
    "value,expected_error,expected_message",
    [
        pytest.param(
            "foo",
            ValueError,
            "tfhers.to_native should be called with a TFHERSInteger",
        ),
    ],
)
def test_tfhers_bad_to_native(value, expected_error, expected_message):
    """
    Test tfhers.to_native with bad arguments.
    """

    with pytest.raises(expected_error) as excinfo:
        tfhers.to_native(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [[1], [2, 3, 4], [5, 6]],
            ValueError,
            "got error while trying to convert list value into a numpy array: "
            "setting an array element with a sequence. The requested array has "
            "an inhomogeneous shape after 1 dimensions. The detected shape was "
            "(3,) + inhomogeneous part.",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [100_000],
            ValueError,
            "ndarray value has bigger elements than what the dtype can support",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [-100_000],
            ValueError,
            "ndarray value has smaller elements than what the dtype can support",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "value can either be an int or ndarray, not a <class 'str'>",
        ),
    ],
)
def test_tfhers_integer_bad_init(dtype, value, expected_error, expected_message):
    """
    Test __init__ of TFHERSInteger with bad arguments.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        tfhers.TFHERSInteger(dtype, value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "function, parameters, parameter_strategy",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            fhe.ParameterSelectionStrategy.MONO,
            id="mono",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            fhe.ParameterSelectionStrategy.V0,
            id="v0",
        ),
    ],
)
def test_tfhers_conversion_without_multi(function, parameters, parameter_strategy, helpers):
    """
    Test that circuits using tfhers integers need to use multi parameters.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    conf = helpers.configuration()
    conf.parameter_selection_strategy = parameter_strategy

    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    with pytest.raises(RuntimeError, match=f"Can't use tfhers integers with {parameter_strategy}"):
        compiler.compile(inputset, conf)


def test_tfhers_circuit_eval():
    """Test evaluation of tfhers function."""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    x = tfhers.TFHERSInteger(dtype, 1)
    y = tfhers.TFHERSInteger(dtype, 2)
    result = binary_tfhers(x, y, lambda x, y: x + y, dtype)
    assert result == 3
