#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""LambdaArgument."""
from typing import List

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    LambdaArgument as _LambdaArgument,
)

# pylint: enable=no-name-in-module,import-error
from .utils import ACCEPTED_INTS
from .wrapper import WrapperCpp


class LambdaArgument(WrapperCpp):
    """LambdaArgument holds scalar or tensor values."""

    def __init__(self, lambda_argument: _LambdaArgument):
        """Wrap the native Cpp object.

        Args:
            lambda_argument (_LambdaArgument): object to wrap

        Raises:
            TypeError: if lambda_argument is not of type _LambdaArgument
        """
        if not isinstance(lambda_argument, _LambdaArgument):
            raise TypeError(
                f"lambda_argument must be of type _LambdaArgument, not {type(lambda_argument)}"
            )
        super().__init__(lambda_argument)

    @staticmethod
    def new(*args, **kwargs):
        """Use from_scalar or from_tensor instead.

        Raises:
            RuntimeError
        """
        raise RuntimeError(
            "you should call from_scalar or from_tensor according to the argument type"
        )

    @staticmethod
    def from_scalar(scalar: int) -> "LambdaArgument":
        """Build a LambdaArgument containing the given scalar value.

        Args:
            scalar (int or numpy.uint): scalar value to embed in LambdaArgument

        Raises:
            TypeError: if scalar is not of type int or numpy.uint

        Returns:
            LambdaArgument
        """
        if not isinstance(scalar, ACCEPTED_INTS):
            raise TypeError(
                f"scalar must be of type int or numpy.int, not {type(scalar)}"
            )
        return LambdaArgument.wrap(_LambdaArgument.from_scalar(scalar))

    @staticmethod
    def from_signed_scalar(scalar: int) -> "LambdaArgument":
        """Build a LambdaArgument containing the given scalar value.

        Args:
            scalar (int or numpy.int): scalar value to embed in LambdaArgument

        Raises:
            TypeError: if scalar is not of type int or numpy.uint

        Returns:
            LambdaArgument
        """
        if not isinstance(scalar, ACCEPTED_INTS):
            raise TypeError(
                f"scalar must be of type int or numpy.uint, not {type(scalar)}"
            )
        return LambdaArgument.wrap(_LambdaArgument.from_signed_scalar(scalar))

    @staticmethod
    def from_tensor_u8(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_u8(data, shape))

    @staticmethod
    def from_tensor_u16(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_u16(data, shape))

    @staticmethod
    def from_tensor_u32(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_u32(data, shape))

    @staticmethod
    def from_tensor_u64(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_u64(data, shape))

    @staticmethod
    def from_tensor_i8(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_i8(data, shape))

    @staticmethod
    def from_tensor_i16(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_i16(data, shape))

    @staticmethod
    def from_tensor_i32(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_i32(data, shape))

    @staticmethod
    def from_tensor_i64(data: List[int], shape: List[int]) -> "LambdaArgument":
        """Build a LambdaArgument containing the given tensor.

        Args:
            data (List[int]): flattened tensor data
            shape (List[int]): shape of original tensor before flattening

        Returns:
            LambdaArgument
        """
        return LambdaArgument.wrap(_LambdaArgument.from_tensor_i64(data, shape))

    def is_signed(self) -> bool:
        """Check if the contained argument is signed.

        Returns:
            bool
        """
        return self.cpp().is_signed()

    def is_scalar(self) -> bool:
        """Check if the contained argument is a scalar.

        Returns:
            bool
        """
        return self.cpp().is_scalar()

    def get_scalar(self) -> int:
        """Return the contained scalar value.

        Returns:
            int
        """
        return self.cpp().get_scalar()

    def get_signed_scalar(self) -> int:
        """Return the contained scalar value.

        Returns:
            int
        """
        return self.cpp().get_signed_scalar()

    def is_tensor(self) -> bool:
        """Check if the contained argument is a tensor.

        Returns:
            bool
        """
        return self.cpp().is_tensor()

    def get_tensor_shape(self) -> List[int]:
        """Return the shape of the contained tensor.

        Returns:
            List[int]: tensor shape
        """
        return self.cpp().get_tensor_shape()

    def get_tensor_data(self) -> List[int]:
        """Return the contained flattened tensor data.

        Returns:
            List[int]
        """
        return self.cpp().get_tensor_data()

    def get_signed_tensor_data(self) -> List[int]:
        """Return the contained flattened tensor data.

        Returns:
            List[int]
        """
        return self.cpp().get_signed_tensor_data()
