"""Import and export TFHErs integers into Concrete."""

# pylint: disable=no-name-in-module,import-error

from mlir._mlir_libs._concretelang._compiler import (
    import_tfhers_fheuint8 as _import_tfhers_fheuint8,
    export_tfhers_fheuint8 as _export_tfhers_fheuint8,
    get_tfhers_fheuint8_description as _get_tfhers_fheuint8_description,
    TfhersFheIntDescription as _TfhersFheIntDescription,
)
from .value import Value
from .wrapper import WrapperCpp

# pylint: enable=no-name-in-module,import-error


class TfhersFheIntDescription(WrapperCpp):
    """A helper class to create `TfhersFheIntDescription`s."""

    def __init__(self, desc: _TfhersFheIntDescription):
        """
        Wrap the native C++ object.

        Args:
            desc (_TfhersFheIntDescription):
                object to wrap

        Raises:
            TypeError:
                if `desc` is not of type `_TfhersFheIntDescription`
        """

        if not isinstance(desc, _TfhersFheIntDescription):
            raise TypeError(
                f"desc must be of type _TfhersFheIntDescription, not {type(desc)}"
            )

        super().__init__(desc)

    @staticmethod
    # pylint: disable=arguments-differ
    def new(
        width: int,
        is_signed: bool,
        message_modulus: int,
        carry_modulus: int,
        degree: int,
        lwe_size: int,
        n_cts: int,
        noise_level: int,
        ks_first: bool,
    ) -> "TfhersFheIntDescription":
        """Create a TfhersFheIntDescription.

        Args:
            width (int): integer width
            is_signed (bool): signed or unsigned
            message_modulus (int): message modulus (not its log2)
            carry_modulus (int): carry modulus (not its log2)
            degree (int): degree
            lwe_size (int): LWE size
            n_cts (int): number of ciphertexts
            noise_level (int): noise level
            ks_first (bool): PBS order (keyswitch first, or bootstrap first)

        Returns:
            TfhersFheIntDescription: TFHErs integer description
        """
        return TfhersFheIntDescription(
            _TfhersFheIntDescription(
                width,
                is_signed,
                message_modulus,
                carry_modulus,
                degree,
                lwe_size,
                n_cts,
                noise_level,
                ks_first,
            )
        )

    @property
    def width(self) -> int:
        """Total integer bitwidth"""
        return self.cpp().width

    @width.setter
    def width(self, width: int):
        self.cpp().width = width

    @property
    def is_signed(self) -> bool:
        """Is the integer signed"""
        return self.cpp().is_signed

    @is_signed.setter
    def is_signed(self, is_signed: bool):
        self.cpp().is_signed = is_signed

    @property
    def message_modulus(self) -> int:
        """Modulus of the message part in each ciphertext"""
        return self.cpp().message_modulus

    @message_modulus.setter
    def message_modulus(self, message_modulus: int):
        self.cpp().message_modulus = message_modulus

    @property
    def carry_modulus(self) -> int:
        """Modulus of the carry part in each ciphertext"""
        return self.cpp().carry_modulus

    @carry_modulus.setter
    def carry_modulus(self, carry_modulus: int):
        self.cpp().carry_modulus = carry_modulus

    @property
    def degree(self) -> int:
        """Tracks the number of operations that have been done"""
        return self.cpp().degree

    @degree.setter
    def degree(self, degree: int):
        self.cpp().degree = degree

    @property
    def lwe_size(self) -> int:
        """LWE size"""
        return self.cpp().lwe_size

    @lwe_size.setter
    def lwe_size(self, lwe_size: int):
        self.cpp().lwe_size = lwe_size

    @property
    def n_cts(self) -> int:
        """Number of ciphertexts"""
        return self.cpp().n_cts

    @n_cts.setter
    def n_cts(self, n_cts: int):
        self.cpp().n_cts = n_cts

    @property
    def noise_level(self) -> int:
        """Noise level"""
        return self.cpp().noise_level

    @noise_level.setter
    def noise_level(self, noise_level: int):
        self.cpp().noise_level = noise_level

    @staticmethod
    def get_unknown_noise_level() -> int:
        """Get unknow noise level value.

        Returns:
            int: unknown noise level value
        """
        return _TfhersFheIntDescription.UNKNOWN_NOISE_LEVEL()

    @property
    def ks_first(self) -> bool:
        """Keyswitch placement relative to the bootsrap in a PBS"""
        return self.cpp().ks_first

    @ks_first.setter
    def ks_first(self, ks_first: bool):
        self.cpp().ks_first = ks_first

    def __str__(self) -> str:
        return (
            f"tfhers_int_description<width={self.width}, signed={self.is_signed}, msg_mod="
            f"{self.message_modulus}, carry_mod={self.carry_modulus}, degree={self.degree}, "
            f"lwe_size={self.lwe_size}, n_cts={self.n_cts}, noise_level={self.noise_level} "
            f"ks_first={self.ks_first}>"
        )

    @staticmethod
    def from_serialized_fheuint8(buffer: bytes) -> "TfhersFheIntDescription":
        """Get the description of a serialized TFHErs fheuint8.

        Args:
            buffer (bytes): serialized fheuint8

        Raises:
            TypeError: buffer is not of type bytes

        Returns:
            TfhersFheIntDescription: description of the serialized fheuint8
        """
        if not isinstance(buffer, bytes):
            raise TypeError(f"buffer must be of type bytes, not {type(buffer)}")
        return TfhersFheIntDescription.wrap(_get_tfhers_fheuint8_description(buffer))


class TfhersExporter:
    """A helper class to import and export TFHErs big integers."""

    @staticmethod
    def export_fheuint8(value: Value, info: TfhersFheIntDescription) -> bytes:
        """Convert Concrete value to TFHErs and serialize it.

        Args:
            value (Value): value to export
            info (TfhersFheIntDescription): description of the TFHErs integer to export to

        Raises:
            TypeError: if wrong input types

        Returns:
            bytes: converted and serialized fheuint8
        """
        if not isinstance(value, Value):
            raise TypeError(f"value must be of type Value, not {type(value)}")
        if not isinstance(info, TfhersFheIntDescription):
            raise TypeError(
                f"info must be of type TfhersFheIntDescription, not {type(info)}"
            )
        return bytes(_export_tfhers_fheuint8(value.cpp(), info.cpp()))

    @staticmethod
    def import_fheuint8(
        buffer: bytes, info: TfhersFheIntDescription, keyid: int, variance: float
    ) -> Value:
        """Unserialize and convert from TFHErs to Concrete value.

        Args:
            buffer (bytes): serialized fheuint8
            info (TfhersFheIntDescription): description of the TFHErs integer to import
            keyid (int): id of the key used for encryption
            variance (float): variance used for encryption

        Raises:
            TypeError: if wrong input types

        Returns:
            Value: unserialized and converted value
        """
        if not isinstance(buffer, bytes):
            raise TypeError(f"buffer must be of type bytes, not {type(buffer)}")
        if not isinstance(info, TfhersFheIntDescription):
            raise TypeError(
                f"info must be of type TfhersFheIntDescription, not {type(info)}"
            )
        if not isinstance(keyid, int):
            raise TypeError(f"keyid must be of type int, not {type(keyid)}")
        if not isinstance(variance, float):
            raise TypeError(f"variance must be of type float, not {type(variance)}")
        return Value.wrap(_import_tfhers_fheuint8(buffer, info.cpp(), keyid, variance))
