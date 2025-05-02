"""
Declaration of `TFHERSIntegerType` class.
"""

from enum import Enum
from functools import partial
from typing import Any, Union

import numpy as np

from ..dtypes import Integer


class EncryptionKeyChoice(Enum):
    """TFHErs key choice: big or small."""

    BIG = 0
    SMALL = 1


class CryptoParams:
    """Crypto parameters used for a tfhers integer."""

    lwe_dimension: int
    glwe_dimension: int
    polynomial_size: int
    pbs_base_log: int
    pbs_level: int
    lwe_noise_distribution: float
    glwe_noise_distribution: float
    encryption_key_choice: EncryptionKeyChoice

    def __init__(
        self,
        lwe_dimension: int,
        glwe_dimension: int,
        polynomial_size: int,
        pbs_base_log: int,
        pbs_level: int,
        lwe_noise_distribution: float,
        glwe_noise_distribution: float,
        encryption_key_choice: EncryptionKeyChoice,
    ):
        self.lwe_dimension = lwe_dimension
        self.glwe_dimension = glwe_dimension
        self.polynomial_size = polynomial_size
        self.pbs_base_log = pbs_base_log
        self.pbs_level = pbs_level
        self.lwe_noise_distribution = lwe_noise_distribution
        self.glwe_noise_distribution = glwe_noise_distribution
        self.encryption_key_choice = encryption_key_choice

    def to_dict(self) -> dict[str, Any]:
        """Convert the CryptoParams object to a dictionary representation.

        Returns:
            Dict[str, Any]: dictionary representation
        """

        return {
            "lwe_dimension": self.lwe_dimension,
            "glwe_dimension": self.glwe_dimension,
            "polynomial_size": self.polynomial_size,
            "pbs_base_log": self.pbs_base_log,
            "pbs_level": self.pbs_level,
            "lwe_noise_distribution": self.lwe_noise_distribution,
            "glwe_noise_distribution": self.glwe_noise_distribution,
            "encryption_key_choice": self.encryption_key_choice.value,
        }

    @staticmethod
    def from_dict(dict_obj: dict[str, Any]) -> "CryptoParams":
        """Create a CryptoParams instance from a dictionary.

        Args:
            dict_obj (dict): A dictionary containing the parameters.

        Returns:
            CryptoParams:
                An instance of CryptoParams initialized with the values from the dictionary.
        """

        return CryptoParams(
            dict_obj["lwe_dimension"],
            dict_obj["glwe_dimension"],
            dict_obj["polynomial_size"],
            dict_obj["pbs_base_log"],
            dict_obj["pbs_level"],
            dict_obj["lwe_noise_distribution"],
            dict_obj["glwe_noise_distribution"],
            EncryptionKeyChoice(dict_obj["encryption_key_choice"]),
        )

    def encryption_variance(self) -> float:
        """Get encryption variance based on parameters.

        This will return different values depending on the encryption key choice.

        Returns:
            float: encryption variance
        """
        if self.encryption_key_choice == EncryptionKeyChoice.BIG:
            return self.glwe_noise_distribution**2
        assert self.encryption_key_choice == EncryptionKeyChoice.SMALL
        return self.lwe_noise_distribution**2

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"crypto_params<lwe_dim={self.lwe_dimension}, glwe_dim={self.glwe_dimension}, "
            f"poly_size={self.polynomial_size}, pbs_base_log={self.pbs_base_log}, "
            f"pbs_level={self.pbs_level}, lwe_noise_distribution={self.lwe_noise_distribution}, "
            f"glwe_noise_distribution={self.glwe_noise_distribution}, encryption_key_choice="
            f"{self.encryption_key_choice.name}>"
        )

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        return (
            isinstance(other, self.__class__)
            and self.lwe_dimension == other.lwe_dimension
            and self.glwe_dimension == other.glwe_dimension
            and self.polynomial_size == other.polynomial_size
            and self.pbs_base_log == other.pbs_base_log
            and self.pbs_level == other.pbs_level
            and self.lwe_noise_distribution == other.lwe_noise_distribution
            and self.glwe_noise_distribution == other.glwe_noise_distribution
            and self.encryption_key_choice == other.encryption_key_choice
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.lwe_dimension,
                self.glwe_dimension,
                self.polynomial_size,
                self.pbs_base_log,
                self.pbs_level,
            )
        )


class TFHERSIntegerType(Integer):
    """
    TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types.
    """

    carry_width: int
    msg_width: int
    params: CryptoParams

    def __init__(
        self,
        is_signed: bool,
        bit_width: int,
        carry_width: int,
        msg_width: int,
        params: CryptoParams,
    ):
        super().__init__(is_signed, bit_width)
        self.carry_width = carry_width
        self.msg_width = msg_width
        self.params = params

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the object's attributes
        """

        return {
            "is_signed": self.is_signed,
            "bit_width": self.bit_width,
            "carry_width": self.carry_width,
            "msg_width": self.msg_width,
            "params": self.params.to_dict(),
        }

    @staticmethod
    def from_dict(dict_obj) -> "TFHERSIntegerType":
        """Create a TFHERSIntegerType instance from a dictionary.

        Args:
            dict_obj (dict): A dictionary representation of the object.

        Returns:
            TFHERSIntegerType: An instance of TFHERSIntegerType created from the dictionary.
        """

        return TFHERSIntegerType(
            dict_obj["is_signed"],
            dict_obj["bit_width"],
            dict_obj["carry_width"],
            dict_obj["msg_width"],
            CryptoParams.from_dict(dict_obj["params"]),
        )

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
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
            f"{self.bit_width}, {self.carry_width}, {self.msg_width}, params={self.params}>"
        )

    def encode(self, value: Union[int, np.integer, list, np.ndarray]) -> np.ndarray:
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
            if self.is_signed and value < 0:
                value_bin = bin(2**bit_width + value)[2:].zfill(bit_width)
            else:
                value_bin = bin(value)[2:].zfill(bit_width)
            # lsb first
            return np.array(
                [int(value_bin[i : i + msg_width], 2) for i in range(0, bit_width, msg_width)][::-1]
            )

        if isinstance(value, list):  # pragma: no cover
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                pass  # pragma: no cover

        if isinstance(value, np.ndarray):
            return np.array([self.encode(int(v)) for v in value.flatten()]).reshape(
                value.shape + (bit_width // msg_width,)
            )

        msg = f"can only encode int, np.integer, list or ndarray, but got {type(value)}"
        raise TypeError(msg)

    def decode(self, value: Union[list, np.ndarray]) -> Union[int, np.ndarray]:
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

        if isinstance(value, list):  # pragma: no cover
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                pass  # pragma: no cover

        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.integer):
            msg = f"can only decode list of integers or ndarray of integers, but got {type(value)}"
            raise TypeError(msg)

        if value.shape[-1] != expected_ct_shape:
            msg = (
                f"expected the last dimension of encoded value "
                f"to be {expected_ct_shape} but it's {value.shape[-1]}"
            )
            raise ValueError(msg)

        if len(value.shape) == 1:
            # lsb first
            decoded = sum(int(v) << (i * msg_width) for i, v in enumerate(value))
            if self.is_signed and decoded >= 2 ** (bit_width - 1):
                decoded = decoded - 2**bit_width
            return decoded

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
