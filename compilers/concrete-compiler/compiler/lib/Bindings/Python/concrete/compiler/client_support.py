#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""Client support."""
from typing import List, Optional, Union
import numpy as np

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import ClientSupport as _ClientSupport

# pylint: enable=no-name-in-module,import-error

from .public_result import PublicResult
from .key_set import KeySet
from .key_set_cache import KeySetCache
from .client_parameters import ClientParameters
from .public_arguments import PublicArguments
from .lambda_argument import LambdaArgument
from .wrapper import WrapperCpp
from .utils import ACCEPTED_INTS, ACCEPTED_NUMPY_UINTS, ACCEPTED_TYPES


class ClientSupport(WrapperCpp):
    """Client interface for doing key generation and encryption.

    It provides features that are needed on the client side:
    - Generation of public and private keys required for the encrypted computation
    - Encryption and preparation of public arguments, used later as input to the computation
    - Decryption of public result returned after execution
    """

    def __init__(self, client_support: _ClientSupport):
        """Wrap the native Cpp object.

        Args:
            client_support (_ClientSupport): object to wrap

        Raises:
            TypeError: if client_support is not of type _ClientSupport
        """
        if not isinstance(client_support, _ClientSupport):
            raise TypeError(
                f"client_support must be of type _ClientSupport, not {type(client_support)}"
            )
        super().__init__(client_support)

    @staticmethod
    # pylint: disable=arguments-differ
    def new() -> "ClientSupport":
        """Build a ClientSupport.

        Returns:
            ClientSupport
        """
        return ClientSupport.wrap(_ClientSupport())

    # pylint: enable=arguments-differ

    @staticmethod
    def key_set(
        client_parameters: ClientParameters,
        keyset_cache: Optional[KeySetCache] = None,
        seed_msb: int = 0,
        seed_lsb: int = 0,
    ) -> KeySet:
        """Generate a key set according to the client parameters.

        If the cache is set, and include equivalent keys as specified by the client parameters,
        the keyset is loaded, otherwise, a new keyset is generated and saved in the cache.

        Args:
            client_parameters (ClientParameters): client parameters specification
            keyset_cache (Optional[KeySetCache], optional): keyset cache. Defaults to None.
            seed_msb (int): msb of seed
            seed_lsb (int): lsb of seed

        Raises:
            TypeError: if client_parameters is not of type ClientParameters
            TypeError: if keyset_cache is not of type KeySetCache
            AssertionError: if seed components is not uint64

        Returns:
            KeySet: generated or loaded keyset
        """
        assert 0 <= seed_msb < 2**64
        assert 0 <= seed_lsb < 2**64

        if keyset_cache is not None and not isinstance(keyset_cache, KeySetCache):
            raise TypeError(
                f"keyset_cache must be None or of type KeySetCache, not {type(keyset_cache)}"
            )

        cpp_cache = None if keyset_cache is None else keyset_cache.cpp()
        return KeySet.wrap(
            _ClientSupport.key_set(
                client_parameters.cpp(),
                cpp_cache,
                seed_msb,
                seed_lsb,
            ),
        )

    @staticmethod
    def encrypt_arguments(
        client_parameters: ClientParameters,
        keyset: KeySet,
        args: List[Union[int, np.ndarray]],
    ) -> PublicArguments:
        """Prepare arguments for encrypted computation.

        Pack public arguments by encrypting the ones that requires encryption, and leaving the rest as plain.
        It also pack public materials (public keys) that are required during the computation.

        Args:
            client_parameters (ClientParameters): client parameters specification
            keyset (KeySet): keyset used to encrypt arguments that require encryption
            args (List[Union[int, np.ndarray]]): list of scalar or tensor arguments

        Raises:
            TypeError: if client_parameters is not of type ClientParameters
            TypeError: if keyset is not of type KeySet

        Returns:
            PublicArguments: public arguments for execution
        """
        if not isinstance(client_parameters, ClientParameters):
            raise TypeError(
                f"client_parameters must be of type ClientParameters, not {type(client_parameters)}"
            )
        if not isinstance(keyset, KeySet):
            raise TypeError(f"keyset must be of type KeySet, not {type(keyset)}")

        signs = client_parameters.input_signs()
        if len(signs) != len(args):
            raise RuntimeError(
                f"function has arity {len(signs)} but is applied to too many arguments"
            )

        lambda_arguments = [
            ClientSupport._create_lambda_argument(arg, signed)
            for arg, signed in zip(args, signs)
        ]
        return PublicArguments.wrap(
            _ClientSupport.encrypt_arguments(
                client_parameters.cpp(),
                keyset.cpp(),
                [arg.cpp() for arg in lambda_arguments],
            )
        )

    @staticmethod
    def decrypt_result(
        client_parameters: ClientParameters,
        keyset: KeySet,
        public_result: PublicResult,
    ) -> Union[int, np.ndarray]:
        """Decrypt a public result using the keyset.

        Args:
            client_parameters (ClientParameters): client parameters for decryption
            keyset (KeySet): keyset used for decryption
            public_result: public result to decrypt

        Raises:
            TypeError: if keyset is not of type KeySet
            TypeError: if public_result is not of type PublicResult
            RuntimeError: if the result is of an unknown type

        Returns:
            Union[int, np.ndarray]: plain result
        """
        if not isinstance(keyset, KeySet):
            raise TypeError(f"keyset must be of type KeySet, not {type(keyset)}")
        if not isinstance(public_result, PublicResult):
            raise TypeError(
                f"public_result must be of type PublicResult, not {type(public_result)}"
            )
        lambda_arg = LambdaArgument.wrap(
            _ClientSupport.decrypt_result(keyset.cpp(), public_result.cpp())
        )

        output_signs = client_parameters.output_signs()
        assert len(output_signs) == 1

        is_signed = lambda_arg.is_signed()
        if lambda_arg.is_scalar():
            return (
                lambda_arg.get_signed_scalar() if is_signed else lambda_arg.get_scalar()
            )

        if lambda_arg.is_tensor():
            return np.array(
                lambda_arg.get_signed_tensor_data()
                if is_signed
                else lambda_arg.get_tensor_data(),
                dtype=(np.int64 if is_signed else np.uint64),
            ).reshape(lambda_arg.get_tensor_shape())

        raise RuntimeError("unknown return type")

    @staticmethod
    def _create_lambda_argument(
        value: Union[int, np.ndarray], signed: bool
    ) -> LambdaArgument:
        """Create a lambda argument holding either an int or tensor value.

        Args:
            value (Union[int, numpy.array]): value of the argument, either an int, or a numpy array
            signed (bool): whether the value is signed

        Raises:
            TypeError: if the values aren't in the expected range, or using a wrong type

        Returns:
            LambdaArgument: lambda argument holding the appropriate value
        """

        # pylint: disable=too-many-return-statements,too-many-branches

        if not isinstance(value, ACCEPTED_TYPES):
            raise TypeError(
                "value of lambda argument must be either int, numpy.array or numpy.(u)int{8,16,32,64}"
            )
        if isinstance(value, ACCEPTED_INTS):
            if (
                isinstance(value, int)
                and not np.iinfo(np.int64).min <= value < np.iinfo(np.uint64).max
            ):
                raise TypeError(
                    "single integer must be in the range [-2**63, 2**64 - 1]"
                )
            if signed:
                return LambdaArgument.from_signed_scalar(value)
            return LambdaArgument.from_scalar(value)
        assert isinstance(value, np.ndarray)
        if value.dtype not in ACCEPTED_NUMPY_UINTS:
            raise TypeError("numpy.array must be of dtype (u)int{8,16,32,64}")
        if value.shape == ():
            if isinstance(value, np.ndarray):
                # extract the single element
                value = value.max()
            # should be a single uint here
            if signed:
                return LambdaArgument.from_signed_scalar(value)
            return LambdaArgument.from_scalar(value)
        if value.dtype == np.uint8:
            return LambdaArgument.from_tensor_u8(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.uint16:
            return LambdaArgument.from_tensor_u16(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.uint32:
            return LambdaArgument.from_tensor_u32(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.uint64:
            return LambdaArgument.from_tensor_u64(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.int8:
            return LambdaArgument.from_tensor_i8(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.int16:
            return LambdaArgument.from_tensor_i16(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.int32:
            return LambdaArgument.from_tensor_i32(
                value.flatten().tolist(), list(value.shape)
            )
        if value.dtype == np.int64:
            return LambdaArgument.from_tensor_i64(
                value.flatten().tolist(), list(value.shape)
            )
        raise TypeError("numpy.array must be of dtype (u)int{8,16,32,64}")
