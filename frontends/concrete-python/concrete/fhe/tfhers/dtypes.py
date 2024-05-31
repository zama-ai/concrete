"""
Declaration of `TFHERSIntegerType` class.
"""

from functools import partial
from typing import Any, Optional, Union

import numpy as np

from ..dtypes import Integer


class TFHERSParams:
    """Crypto parameters used for a tfhers integer."""

    lwe_dimension: int
    glwe_dimension: int
    polynomial_size: int
    lwe_noise_distribution_std_dev: float
    glwe_noise_distribution_std_dev: float
    pbs_base_log: int
    pbs_level: int
    ks_base_log: int
    ks_level: int
    message_modulus: int
    carry_modulus: int
    max_noise_level: int
    log2_p_fail: float
    ciphertext_modulus: Optional[int]
    big_encryption_key: bool

    def __init__(
        self,
        lwe_dimension: int,
        glwe_dimension: int,
        polynomial_size: int,
        lwe_noise_distribution_std_dev: float,
        glwe_noise_distribution_std_dev: float,
        pbs_base_log: int,
        pbs_level: int,
        ks_base_log: int,
        ks_level: int,
        message_modulus: int,
        carry_modulus: int,
        max_noise_level: int,
        log2_p_fail: float,
        ciphertext_modulus: Optional[int],
        big_encryption_key: bool,
    ):
        self.lwe_dimension = lwe_dimension
        self.glwe_dimension = glwe_dimension
        self.polynomial_size = polynomial_size
        self.lwe_noise_distribution_std_dev = lwe_noise_distribution_std_dev
        self.glwe_noise_distribution_std_dev = glwe_noise_distribution_std_dev
        self.pbs_base_log = pbs_base_log
        self.pbs_level = pbs_level
        self.ks_base_log = ks_base_log
        self.ks_level = ks_level
        self.message_modulus = message_modulus
        self.carry_modulus = carry_modulus
        self.max_noise_level = max_noise_level
        self.log2_p_fail = log2_p_fail
        self.ciphertext_modulus = ciphertext_modulus
        self.big_encryption_key = big_encryption_key

    def __str__(self) -> str:
        return (
            f"tfhers_params<{self.lwe_dimension}, {self.glwe_dimension}, {self.polynomial_size}, "
            f"{self.lwe_noise_distribution_std_dev}, {self.glwe_noise_distribution_std_dev}, "
            f"{self.pbs_base_log}, {self.pbs_level}, {self.ks_base_log}, {self.ks_level}, "
            f"{self.message_modulus}, {self.carry_modulus}, {self.max_noise_level}, "
            f"{self.log2_p_fail}, {self.ciphertext_modulus}, {self.big_encryption_key}>"
        )

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.lwe_dimension == other.lwe_dimension
            and self.glwe_dimension == other.glwe_dimension
            and self.polynomial_size == other.polynomial_size
            and self.lwe_noise_distribution_std_dev == other.lwe_noise_distribution_std_dev
            and self.glwe_noise_distribution_std_dev == other.glwe_noise_distribution_std_dev
            and self.pbs_base_log == other.pbs_base_log
            and self.pbs_level == other.pbs_level
            and self.ks_base_log == other.ks_base_log
            and self.ks_level == other.ks_level
            and self.message_modulus == other.message_modulus
            and self.carry_modulus == other.carry_modulus
            and self.max_noise_level == other.max_noise_level
            and self.log2_p_fail == other.log2_p_fail
            and self.ciphertext_modulus == other.ciphertext_modulus
            and self.big_encryption_key == other.big_encryption_key
        )


class TFHERSIntegerType(Integer):
    """
    TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types.
    """

    carry_width: int
    msg_width: int
    params: TFHERSParams

    def __init__(
        self,
        is_signed: bool,
        bit_width: int,
        carry_width: int,
        msg_width: int,
        params: TFHERSParams,
    ):
        super().__init__(is_signed, bit_width)
        self.carry_width = carry_width
        self.msg_width = msg_width
        self.params = params

        if 2**msg_width != params.message_modulus:
            msg = (
                f"inconsistency between msg_modulus({params.message_modulus}), "
                f"and msg_width({msg_width}). msg_modulus should be 2**msg_width"
            )
            raise ValueError(msg)

        if 2**carry_width != params.carry_modulus:
            msg = (
                f"inconsistency between carry_modulus({params.carry_modulus}), "
                f"and carry_width({carry_width}). carry_modulus should be 2**carry_width"
            )
            raise ValueError(msg)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and super().__eq__(other)
            and self.carry_width == other.carry_width
            and self.msg_width == other.msg_width
            and self.params == other.params
        )

    def __str__(self) -> str:
        return (
            f"tfhers<{('int' if self.is_signed else 'uint')}"
            f"{self.bit_width}, {self.carry_width}, {self.msg_width}>"
        )

    def encode(self, value: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Encode a scalar or tensor to tfhers integers.

        Args:
            value (Union[int, np.ndarray]): scalar or tensor of integer to encode

        Raises:
            TypeError: wrong value type

        Returns:
            np.ndarray: encoded scalar or tensor
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        if isinstance(value, (int, np.integer)):
            value_bin = bin(value)[2:].zfill(bit_width)
            # msb first
            return np.array(
                [int(value_bin[i : i + msg_width], 2) for i in range(0, bit_width, msg_width)]
            )
        if isinstance(value, np.ndarray):
            return np.array([self.encode(int(v)) for v in value.flatten()]).reshape(
                value.shape + (bit_width // msg_width,)
            )
        msg = f"can only encode int or ndarray, but got {type(value)}"
        raise TypeError(msg)

    def decode(self, value: np.ndarray) -> Union[int, np.ndarray]:
        """Decode a tfhers-encoded integer (scalar or tensor).

        Args:
            value (np.ndarray): encoded value

        Raises:
            ValueError: bad encoding

        Returns:
            Union[int, np.ndarray]: decoded value
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        expected_ct_shape = bit_width // msg_width
        if value.shape[-1] != expected_ct_shape:
            msg = (
                f"bad encoding: expected value with last shape being {expected_ct_shape} "
                f"but got {value.shape[-1]}"
            )
            raise ValueError(msg)
        if len(value.shape) == 1:
            # reversed because it's msb first and we are computing powers lsb first
            return sum(v << i * msg_width for i, v in enumerate(reversed(value)))
        cts = value.reshape((-1, expected_ct_shape))
        return np.array([self.decode(ct) for ct in cts]).reshape(value.shape[:-1])


int8 = partial(TFHERSIntegerType, True, 8)
uint8 = partial(TFHERSIntegerType, False, 8)
int16 = partial(TFHERSIntegerType, True, 16)
uint16 = partial(TFHERSIntegerType, False, 16)

int8_2_2 = partial(TFHERSIntegerType, True, 8, 2, 2)
uint8_2_2 = partial(TFHERSIntegerType, False, 8, 2, 2)
int16_2_2 = partial(TFHERSIntegerType, True, 16, 2, 2)
uint16_2_2 = partial(TFHERSIntegerType, False, 16, 2, 2)
