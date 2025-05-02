"""
tfhers module to represent, and compute on tfhers integer values.
"""

import json
from math import log2

from .bridge import new_bridge
from .dtypes import (
    CryptoParams,
    EncryptionKeyChoice,
    TFHERSIntegerType,
    int8,
    int8_2_2,
    int16,
    int16_2_2,
    uint8,
    uint8_2_2,
    uint16,
    uint16_2_2,
)
from .specs import TFHERSClientSpecs
from .tracing import from_native, to_native
from .values import TFHERSInteger


def get_type_from_params(
    path_to_params_json: str, is_signed: bool, precision: int
) -> TFHERSIntegerType:
    """Get a TFHE-rs integer type from TFHE-rs parameters in JSON format.

    Args:
        path_to_params_json (str): path to the TFHE-rs parameters (JSON format)
        is_signed (bool): sign of the result type
        precision (int): precision of the result type

    Returns:
        TFHERSIntegerType: constructed type from the loaded parameters
    """

    # Read crypto parameters from TFHE-rs in the json file
    with open(path_to_params_json, "r", encoding="utf-8") as f:
        crypto_param_dict = json.load(f)

    return get_type_from_params_dict(crypto_param_dict, is_signed, precision)


def get_type_from_params_dict(
    crypto_param_dict: dict, is_signed: bool, precision: int
) -> TFHERSIntegerType:
    """Get a TFHE-rs integer type from TFHE-rs parameters in JSON format.

    Args:
        crypto_param_dict (Dict): dictionary of TFHE-rs parameters
        is_signed (bool): sign of the result type
        precision (int): precision of the result type

    Returns:
        TFHERSIntegerType: constructed type from the loaded parameters
    """

    lwe_dim = crypto_param_dict["lwe_dimension"]
    glwe_dim = crypto_param_dict["glwe_dimension"]
    poly_size = crypto_param_dict["polynomial_size"]
    pbs_base_log = crypto_param_dict["pbs_base_log"]
    pbs_level = crypto_param_dict["pbs_level"]
    msg_width = int(log2(crypto_param_dict["message_modulus"]))
    carry_width = int(log2(crypto_param_dict["carry_modulus"]))
    lwe_noise_distr = crypto_param_dict["lwe_noise_distribution"]["Gaussian"]["std"]
    glwe_noise_distr = crypto_param_dict["glwe_noise_distribution"]["Gaussian"]["std"]
    encryption_key_choice = (
        EncryptionKeyChoice.BIG
        if crypto_param_dict["encryption_key_choice"] == "Big"
        else EncryptionKeyChoice.SMALL
    )

    assert glwe_dim == 1, "glwe dim must be 1"
    assert encryption_key_choice == EncryptionKeyChoice.BIG, "encryption_key_choice must be BIG"

    tfhers_params = CryptoParams(
        lwe_dimension=lwe_dim,
        glwe_dimension=glwe_dim,
        polynomial_size=poly_size,
        pbs_base_log=pbs_base_log,
        pbs_level=pbs_level,
        lwe_noise_distribution=lwe_noise_distr,
        glwe_noise_distribution=glwe_noise_distr,
        encryption_key_choice=encryption_key_choice,
    )
    return TFHERSIntegerType(
        is_signed=is_signed,
        bit_width=precision,
        carry_width=carry_width,
        msg_width=msg_width,
        params=tfhers_params,
    )
