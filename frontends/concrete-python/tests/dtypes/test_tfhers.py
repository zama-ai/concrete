"""
Tests of `TFHERSIntegerType` data type.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from concrete.fhe import tfhers

DEFAULT_TFHERS_PARAM = tfhers.CryptoParams(
    909,
    1,
    4096,
    15,
    2,
    0,
    2.168404344971009e-19,
    tfhers.EncryptionKeyChoice.BIG,
)

DEFAULT_TFHERS_PARAMS_DICT = {
    "lwe_dimension": 902,
    "glwe_dimension": 1,
    "polynomial_size": 4096,
    "lwe_noise_distribution": {"Gaussian": {"std": 1.0994794733558207e-6, "mean": 0.0}},
    "glwe_noise_distribution": {"Gaussian": {"std": 2.168404344971009e-19, "mean": 0.0}},
    "pbs_base_log": 15,
    "pbs_level": 2,
    "ks_base_log": 3,
    "ks_level": 6,
    "message_modulus": 4,
    "carry_modulus": 8,
    "max_noise_level": 10,
    "log2_p_fail": -64.084,
    "ciphertext_modulus": {"modulus": 0, "scalar_bits": 64},
    "encryption_key_choice": "Big",
}


def parameterize_partial_dtype(partial_dtype) -> tfhers.TFHERSIntegerType:
    """Create a tfhers type from a partial func missing tfhers params.

    Args:
        partial_dtype (callable): partial function to create dtype (missing params)

    Returns:
        tfhers.TFHERSIntegerType: tfhers type
    """

    return partial_dtype(DEFAULT_TFHERS_PARAM)


def test_tfhers_encode_bad_type():
    """Test encoding of unsupported type"""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    with pytest.raises(
        TypeError,
        match=r"can only encode int, np.integer, list or ndarray, but got <class 'str'>",
    ):
        dtype.encode("bad type")


def test_tfhers_encode_ndarray():
    """Test ndarray encoding"""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    shape = (4, 5)
    value = np.random.randint(0, 2**10, size=shape)
    encoded = dtype.encode(value)
    decoded = dtype.decode(encoded)
    assert (decoded == value).all()
    assert encoded.shape == shape + (8,)


def test_tfhers_bad_decode():
    """Test decoding of bad values"""
    dtype = parameterize_partial_dtype(tfhers.uint8_2_2)
    shape = (2, 10)
    bad_value = np.random.randint(0, 2**10, size=shape)
    with pytest.raises(
        ValueError,
        match=r"expected the last dimension of encoded value to be 4 but it's 10",
    ):
        dtype.decode(bad_value)


def test_tfhers_integer_bad_values():
    """Test new integer with bad values"""
    dtype = parameterize_partial_dtype(tfhers.uint8_2_2)
    with pytest.raises(
        ValueError,
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, 2],
                [
                    2,
                ],
            ],
        )

    with pytest.raises(
        ValueError,
        match=r"ndarray value has bigger elements than what the dtype can support",
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, 2],
                [2, 2**10],
            ],
        )

    with pytest.raises(
        ValueError,
        match=r"ndarray value has smaller elements than what the dtype can support",
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, -2],
                [2, 2],
            ],
        )

    with pytest.raises(
        TypeError,
        match=r"value can either be an int or ndarray, not a <class 'str'>",
    ):
        tfhers.TFHERSInteger(dtype, "bad value")


@pytest.mark.parametrize(
    "crypto_params",
    [
        pytest.param(
            tfhers.CryptoParams(
                909,
                1,
                4096,
                15,
                2,
                0,
                2.168404344971009e-19,
                tfhers.EncryptionKeyChoice.BIG,
            ),
            id="big",
        ),
        pytest.param(
            tfhers.CryptoParams(
                909,
                1,
                4096,
                15,
                2,
                2.13278129e-14,
                0,
                tfhers.EncryptionKeyChoice.SMALL,
            ),
            id="small",
        ),
    ],
)
def test_tfhers_encryption_variance(crypto_params: tfhers.CryptoParams):
    """Test encryption variance computation"""
    if crypto_params.encryption_key_choice == tfhers.EncryptionKeyChoice.BIG:
        assert crypto_params.encryption_variance() == crypto_params.glwe_noise_distribution**2
        return
    assert crypto_params.encryption_key_choice == tfhers.EncryptionKeyChoice.SMALL
    assert crypto_params.encryption_variance() == crypto_params.lwe_noise_distribution**2


@pytest.mark.parametrize("params_dict", (DEFAULT_TFHERS_PARAMS_DICT,))
@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("n_bits", [5, 8, 13, 16])
def test_load_tfhers_params_dict(params_dict, is_signed, n_bits):
    """Test loading tfhers params dict"""
    dtype = tfhers.get_type_from_params_dict(params_dict, is_signed, n_bits)
    assert dtype.bit_width == n_bits
    assert dtype.is_signed == is_signed

    test_keys = ["lwe_dimension", "glwe_dimension", "polynomial_size", "pbs_base_log", "pbs_level"]

    for k in test_keys:
        assert getattr(dtype.params, k) == params_dict[k]


@pytest.mark.parametrize("params_dict", (DEFAULT_TFHERS_PARAMS_DICT,))
def test_load_tfhers_params_file(params_dict):
    """Test loading tfhers params file"""
    with tempfile.NamedTemporaryFile(delete=False) as ftemp:
        fpath = ftemp.name
        ftemp.write(bytes(json.dumps(params_dict), "utf8"))

    tfhers.get_type_from_params(fpath, True, 8)
    os.unlink(fpath)
