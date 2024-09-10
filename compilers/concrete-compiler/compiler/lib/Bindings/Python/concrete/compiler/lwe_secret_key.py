#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.


"""LweSecretKey."""

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LweSecretKey as _LweSecretKey,
    LweSecretKeyParam as _LweSecretKeyParam,
)

# pylint: enable=no-name-in-module,import-error
from .wrapper import WrapperCpp


class LweSecretKeyParam(WrapperCpp):
    """LWE Secret Key Parameters"""

    def __init__(self, lwe_secret_key_param: _LweSecretKeyParam):
        """Wrap the native Cpp object.

        Args:
            lwe_secret_key_param (_LweSecretKeyParam): object to wrap

        Raises:
            TypeError: if lwe_secret_key_param is not of type _LweSecretKeyParam
        """
        if not isinstance(lwe_secret_key_param, _LweSecretKeyParam):
            raise TypeError(
                "lwe_secret_key_param must be of type _LweSecretKeyParam, "
                f"not {type(lwe_secret_key_param)}"
            )
        super().__init__(lwe_secret_key_param)

    @property
    def dimension(self) -> int:
        """LWE dimension"""
        return self.cpp().dimension


class LweSecretKey(WrapperCpp):
    """An LweSecretKey."""

    def __init__(self, lwe_secret_key: _LweSecretKey):
        """Wrap the native Cpp object.

        Args:
            lwe_secret_key (_LweSecretKey): object to wrap

        Raises:
            TypeError: if lwe_secret_key is not of type _LweSecretKey
        """
        if not isinstance(lwe_secret_key, _LweSecretKey):
            raise TypeError(
                f"lwe_secret_key must be of type _LweSecretKey, not {type(lwe_secret_key)}"
            )
        super().__init__(lwe_secret_key)

    def serialize(self) -> bytes:
        """Serialize key.

        Returns:
            bytes: serialized key
        """

        return self.cpp().serialize()

    @staticmethod
    def deserialize(serialized_key: bytes, param: LweSecretKeyParam) -> "LweSecretKey":
        """Deserialize LweSecretKey from bytes.

        Args:
            serialized_key (bytes): previously serialized secret key

        Raises:
            TypeError: if wrong types for input arguments

        Returns:
            LweSecretKey: deserialized object
        """
        if not isinstance(serialized_key, bytes):
            raise TypeError(
                f"serialized_key must be of type bytes, not {type(serialized_key)}"
            )
        if not isinstance(param, LweSecretKeyParam):
            raise TypeError(
                f"param must be of type LweSecretKeyParam, not {type(param)}"
            )
        return LweSecretKey.wrap(_LweSecretKey.deserialize(serialized_key, param.cpp()))

    def serialize_as_glwe(self, glwe_dim: int, poly_size: int) -> bytes:
        """Serialize key as a glwe secret key.

        Args:
            glwe_dim (int): glwe dimension of the key
            poly_size (int): polynomial size of the key

        Raises:
            TypeError: if wrong types for input arguments

        Returns:
            bytes: serialized key
        """
        if not isinstance(glwe_dim, int):
            raise TypeError(f"glwe_dim must be of type int, not {type(glwe_dim)}")
        if not isinstance(poly_size, int):
            raise TypeError(f"poly_size must be of type int, not {type(poly_size)}")
        return self.cpp().serialize_as_glwe(glwe_dim, poly_size)

    @staticmethod
    def deserialize_from_glwe(
        serialized_glwe_key: bytes, param: LweSecretKeyParam
    ) -> "LweSecretKey":
        """Deserialize LweSecretKey from glwe secret key bytes.

        Args:
            serialized_glwe_key (bytes): previously serialized glwe secret key

        Raises:
            TypeError: if wrong types for input arguments

        Returns:
            LweSecretKey: deserialized object
        """
        if not isinstance(serialized_glwe_key, bytes):
            raise TypeError(
                f"serialized_glwe_key must be of type bytes, not {type(serialized_glwe_key)}"
            )
        if not isinstance(param, LweSecretKeyParam):
            raise TypeError(
                f"param must be of type LweSecretKeyParam, not {type(param)}"
            )
        return LweSecretKey.wrap(
            _LweSecretKey.deserialize_from_glwe(serialized_glwe_key, param.cpp())
        )

    @property
    def param(self) -> LweSecretKeyParam:
        """LWE Secret Key Parameters"""
        return LweSecretKeyParam.wrap(self.cpp().param)
