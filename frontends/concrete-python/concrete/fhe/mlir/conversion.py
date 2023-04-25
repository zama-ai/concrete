"""
Declaration of `ConversionType` and `Conversion` classes.
"""

# pylint: disable=import-error,

import re
from typing import Optional, Tuple

from mlir.ir import OpResult as MlirOperation
from mlir.ir import Type as MlirType

from ..representation import Node

# pylint: enable=import-error


SCALAR_INT_SEARCH_REGEX = re.compile(r"^i([0-9]+)$")
SCALAR_EINT_SEARCH_REGEX = re.compile(r"^!FHE\.e(s)?int<([0-9]+)>$")

TENSOR_INT_SEARCH_REGEX = re.compile(r"^tensor<(([0-9]+x)+)i([0-9]+)>$")
TENSOR_EINT_SEARCH_REGEX = re.compile(r"^tensor<(([0-9]+x)+)!FHE\.e(s)?int<([0-9]+)>>$")


class ConversionType:
    """
    ConversionType class, to make it easier to work with MLIR types.
    """

    mlir: MlirType

    bit_width: int
    is_encrypted: bool
    is_signed: bool
    shape: Tuple[int, ...]

    def __init__(self, mlir: MlirType):
        self.mlir = mlir
        mlir_type_str = str(mlir)

        search = SCALAR_INT_SEARCH_REGEX.search(mlir_type_str)
        if search:
            (matched_bit_width,) = search.groups()

            self.bit_width = int(matched_bit_width)
            self.is_encrypted = False
            self.is_signed = True
            self.shape = ()

            return

        search = SCALAR_EINT_SEARCH_REGEX.search(mlir_type_str)
        if search:
            matched_is_signed, matched_bit_width = search.groups()

            self.bit_width = int(matched_bit_width)
            self.is_encrypted = True
            self.is_signed = matched_is_signed is not None
            self.shape = ()

            return

        search = TENSOR_INT_SEARCH_REGEX.search(mlir_type_str)
        if search:
            matched_shape, _, matched_bit_width = search.groups()

            self.bit_width = int(matched_bit_width)
            self.is_encrypted = False
            self.is_signed = True
            self.shape = tuple(int(size) for size in matched_shape.rstrip("x").split("x"))

            return

        search = TENSOR_EINT_SEARCH_REGEX.search(mlir_type_str)
        if search:
            matched_shape, _, matched_is_signed, matched_bit_width = search.groups()

            self.bit_width = int(matched_bit_width)
            self.is_encrypted = True
            self.is_signed = matched_is_signed is not None
            self.shape = tuple(int(size) for size in matched_shape.rstrip("x").split("x"))

            return

        self.is_encrypted = False
        self.bit_width = 64
        self.is_signed = False
        self.shape = ()

    # pylint: disable=missing-function-docstring

    @property
    def is_clear(self) -> bool:
        return not self.is_encrypted

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()

    @property
    def is_tensor(self) -> bool:
        return self.shape != ()

    @property
    def is_unsigned(self) -> bool:
        return not self.is_signed

    # pylint: enable=missing-function-docstring


class Conversion:
    """
    Conversion class, to store MLIR operations with additional information.
    """

    origin: Node

    type: ConversionType
    result: MlirOperation

    _original_bit_width: Optional[int]

    def __init__(self, origin: Node, result: MlirOperation):
        self.origin = origin

        self.type = ConversionType(result.type)
        self.result = result

        self._original_bit_width = None

    def set_original_bit_width(self, original_bit_width: int):
        """
        Set the original bit-width of the conversion.
        """
        self._original_bit_width = original_bit_width

    @property
    def original_bit_width(self) -> int:
        """
        Get the original bit-width of the conversion.

        If not explicitly set, defaults to the actual bit width.
        """
        return self._original_bit_width if self._original_bit_width is not None else self.bit_width

    # pylint: disable=missing-function-docstring

    @property
    def bit_width(self) -> int:
        return self.type.bit_width

    @property
    def is_clear(self) -> bool:
        return self.type.is_clear

    @property
    def is_encrypted(self) -> bool:
        return self.type.is_encrypted

    @property
    def is_scalar(self) -> bool:
        return self.type.is_scalar

    @property
    def is_signed(self) -> bool:
        return self.type.is_signed

    @property
    def is_tensor(self) -> bool:
        return self.type.is_tensor

    @property
    def is_unsigned(self) -> bool:
        return self.type.is_unsigned

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.type.shape

    # pylint: enable=missing-function-docstring
