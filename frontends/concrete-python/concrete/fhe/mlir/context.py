"""
Declaration of `Context` class.
"""

# pylint: disable=import-error,no-name-in-module

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from concrete.lang.dialects import fhe, fhelinalg
from concrete.lang.dialects.fhe import EncryptedIntegerType, EncryptedSignedIntegerType
from mlir.dialects import arith, tensor
from mlir.ir import ArrayAttr as MlirArrayAttr
from mlir.ir import Attribute as MlirAttribute
from mlir.ir import BoolAttr as MlirBoolAttr
from mlir.ir import Context as MlirContext
from mlir.ir import DenseElementsAttr as MlirDenseElementsAttr
from mlir.ir import DenseI64ArrayAttr as MlirDenseI64ArrayAttr
from mlir.ir import IndexType
from mlir.ir import IntegerAttr as MlirIntegerAttr
from mlir.ir import IntegerType
from mlir.ir import OpResult as MlirOperation
from mlir.ir import RankedTensorType
from mlir.ir import Type as MlirType

from ..dtypes import Integer
from ..representation import Graph, Node
from ..values import ValueDescription
from .conversion import Conversion, ConversionType
from .processors import GraphProcessor
from .utils import MAXIMUM_TLU_BIT_WIDTH, _FromElementsOp

# pylint: enable=import-error,no-name-in-module


class Context:
    """
    Context class, to perform operations on conversions.
    """

    context: MlirContext

    graph: Graph

    conversions: Dict[Node, Conversion]
    converting: Node

    conversion_cache: Dict[Tuple, Conversion]
    constant_cache: Dict[MlirAttribute, MlirOperation]

    def __init__(self, context: MlirContext, graph: Graph):
        self.context = context

        self.graph = graph

        self.conversions = {}

        self.conversion_cache = {}
        self.constant_cache = {}

    # types

    def i(self, width: int) -> ConversionType:
        """
        Get clear signless integer type (e.g., i3, i5).
        """
        return ConversionType(IntegerType.get_signless(width))

    def eint(self, width: int) -> ConversionType:
        """
        Get encrypted unsigned integer type (e.g., !FHE.eint<3>, !FHE.eint<5>).
        """
        return ConversionType(EncryptedIntegerType.get(self.context, width))

    def esint(self, width: int) -> ConversionType:
        """
        Get encrypted signed integer type (e.g., !FHE.esint<3>, !FHE.esint<5>).
        """
        return ConversionType(EncryptedSignedIntegerType.get(self.context, width))

    def index_type(self) -> MlirType:
        """
        Get index type.
        """
        return ConversionType(IndexType.parse("index"))

    def tensor(self, element_type: ConversionType, shape: Tuple[int, ...]) -> ConversionType:
        """
        Get tensor type (e.g., tensor<5xi3>, tensor<3x2x!FHE.eint<5>>).
        """
        return ConversionType(RankedTensorType.get(shape, element_type.mlir))

    def typeof(self, value: Union[ValueDescription, Node]) -> ConversionType:
        """
        Get type corresponding to a value or a node.
        """
        if isinstance(value, Node):
            value = value.output

        assert isinstance(value.dtype, Integer)
        bit_width = value.dtype.bit_width

        if value.is_clear:
            result = self.i(bit_width)
        elif value.dtype.is_signed:
            result = self.esint(bit_width)
        else:
            result = self.eint(bit_width)

        return result if value.is_scalar else self.tensor(result, value.shape)

    # utilities

    def attribute(self, resulting_type: ConversionType, value: Any) -> MlirAttribute:
        """
        Create an MLIR attribute.

        Args:
            resulting_type (ConversionType):
                type of the attribute

            value (Any):
                value of the attribute

        Returns:
            MlirAttribute:
                resulting MLIR attribute
        """

        is_tensor = isinstance(value, list) or (isinstance(value, np.ndarray) and value.shape != ())
        is_numpy = not isinstance(value, (list, int))

        if is_tensor:
            value = value.tolist() if is_numpy else value
            return MlirAttribute.parse(f"dense<{value}> : {resulting_type.mlir}")

        return MlirAttribute.parse(f"{value} : {resulting_type.mlir}")

    def error(self, highlights: Mapping[Node, Union[str, List[str]]]):
        """
        Fail compilation with an error.

        Args:
            highlights (Mapping[Node, Union[str, List[str]]]):
                nodes to highlight along with messages
        """

        GraphProcessor.error(self.graph, highlights)

    def is_bit_width_compatible(self, *args: Optional[Union[ConversionType, Conversion]]) -> bool:
        """
        Check if conversion types are compatible in terms of bit-width.
        """

        assert len(args) >= 2
        args = tuple(arg.type if isinstance(arg, Conversion) else arg for arg in args)

        def check(type1, type2):
            return (
                (type1.bit_width + 1) == type2.bit_width
                if type1.is_encrypted and type2.is_clear
                else (
                    type1.bit_width == (type2.bit_width + 1)
                    if type1.is_clear and type2.is_encrypted
                    else type1.bit_width == type2.bit_width
                )
            )

        reference = args[0]

        is_compatible = True
        for arg in args[1:]:
            if arg is None:
                continue
            is_compatible = check(reference, arg)
            if not is_compatible:
                break  # pragma: no cover

        return is_compatible

    def operation(
        self,
        operation: Callable,
        resulting_type: ConversionType,
        *args,
        original_bit_width: Optional[int] = None,
        **kwargs,
    ) -> Conversion:
        """
        Create a conversion from an MLIR operation.

        Args:
            operation (Callable):
                MLIR operation to create (e.g., fhe.AddEintOp)

            resulting_type (ConversionType):
                type of the output of the operation

            *args (Any):
                args to pass to the operation

            original_bit_width (Optional[int], default = None):
                original bit width of the resulting conversion

            *kwargs (Any):
                kwargs to pass to the operation
        Returns:
            Conversion:
                resulting conversion
        """

        cache_key = (resulting_type.mlir, operation, *args, *kwargs)
        cached_conversion = self.conversion_cache.get(cache_key)

        if cached_conversion is None:
            if operation not in [tensor.ExtractOp, tensor.InsertSliceOp]:
                result = operation(resulting_type.mlir, *args, **kwargs).result
            else:
                result = operation(*args, **kwargs).result

            cached_conversion = Conversion(self.converting, result)
            if original_bit_width is not None:
                cached_conversion.set_original_bit_width(original_bit_width)

            self.conversion_cache[cache_key] = cached_conversion

        return cached_conversion

    def pack_to_chunk_groups_and_map(
        self,
        resulting_type: ConversionType,
        bit_width: int,
        chunk_size: int,
        x: Conversion,
        y: Conversion,
        mapper: Callable,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> List[Conversion]:
        """
        Extract the chunks of two values, pack them in a single integer and map the integer.

        Args:
            resulting_type (ConversionType):
                type of the outputs of the operation

            bit_width (int):
                bit width of the operation

            chunk_size (int):
                chunks size of the operation

            x (Conversion):
                first operand

            y (Conversion):
                second operand

            mapper (Callable):
                mapping function

            x_offset (int, default=0):
                optional offset for x during chunk extraction

            y_offset (int, default=0):
                optional offset for x during chunk extraction

        Returns:
            List[Conversion]:
                result of mapping chunks of x and y
        """

        result = []
        for chunk_index, offset in enumerate(range(0, bit_width, chunk_size)):
            bits_to_process = min(chunk_size, bit_width - offset)
            right_shift_by = bit_width - offset - bits_to_process
            mask = (2**bits_to_process) - 1

            chunk_x = self.tlu(
                resulting_type,
                x,
                [
                    ((((x + x_offset) >> right_shift_by) & mask) << bits_to_process)
                    for x in range(2**bit_width)
                ],
            )
            chunk_y = self.tlu(
                resulting_type,
                y,
                [((y + y_offset) >> right_shift_by) & mask for y in range(2**bit_width)],
            )

            packed_chunks = self.add(resulting_type, chunk_x, chunk_y)
            mapped_chunks = self.tlu(
                resulting_type,
                packed_chunks,
                [mapper(chunk_index, x, y) for x in range(mask + 1) for y in range(mask + 1)],
            )

            result.append(mapped_chunks)

        return result

    # operations

    # each operation is checked for compatibility
    # between the inputs and the resulting type
    # on following constraints:

    # - bit width
    # - encryption status
    # - signedness
    # - tensorization
    # - operation specific constraints

    # for incompatibilities that are impossible to be caused by users, assertions are used
    # for incompatibilities that can be caused by users, self.error(highlights) is used

    # pylint: disable=missing-function-docstring

    def add(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        if x.is_clear and y.is_clear:
            highlights = {
                x.origin: "lhs is clear",
                y.origin: "rhs is clear" if x.origin is not y.origin else "operand is clear",
                self.converting: "but clear-clear additions are not supported",
            }
            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)

        x = self.to_signedness(x, of=resulting_type)
        y = self.to_signedness(y, of=resulting_type)

        use_linalg = x.is_tensor or y.is_tensor

        x = self.tensorize(x) if use_linalg else x
        y = self.tensorize(y) if use_linalg else y

        if x.is_clear:
            x, y = y, x

        dialect = fhelinalg if use_linalg else fhe
        operation = dialect.AddEintIntOp if y.is_clear else dialect.AddEintOp

        return self.operation(
            operation,
            resulting_type,
            x.result,
            y.result,
        )

    def array(self, resulting_type: ConversionType, elements: List[Conversion]) -> Conversion:
        assert resulting_type.is_encrypted
        assert self.is_bit_width_compatible(resulting_type, *elements)

        sanitized_elements = []
        for element in elements:
            assert element.is_scalar

            if resulting_type.is_clear or element.is_encrypted:
                sanitized_elements.append(element)
                continue

            encrypted_element_type = self.typeof(
                ValueDescription(
                    dtype=Integer(
                        is_signed=resulting_type.is_signed,
                        bit_width=resulting_type.bit_width,
                    ),
                    shape=(),
                    is_encrypted=True,
                )
            )

            encrypted_element = self.encrypt(encrypted_element_type, element)
            encrypted_element.set_original_bit_width(element.original_bit_width)

            encrypted_element = self.to_signedness(encrypted_element, of=resulting_type)

            sanitized_elements.append(encrypted_element)

        original_bit_width = max(element.original_bit_width for element in sanitized_elements)
        mlir_elements = [element.result for element in sanitized_elements]
        return self.operation(
            _FromElementsOp, resulting_type, *mlir_elements, original_bit_width=original_bit_width
        )

    def assign_static(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        index: Sequence[Union[int, np.integer, slice]],
    ):
        if x.is_clear and y.is_encrypted:
            highlights = {
                x.origin: "tensor is clear",
                y.origin: "assigned value is encrypted",
                self.converting: "but encrypted values cannot be assigned to clear tensors",
            }
            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)

        index = list(index)
        while len(index) < len(x.shape):
            index.append(slice(None, None, None))

        offsets = []
        sizes = []
        strides = []

        for indexing_element, dimension_size in zip(index, x.shape):
            if isinstance(indexing_element, slice):
                size = int(np.zeros(dimension_size)[indexing_element].shape[0])
                stride = int(indexing_element.step if indexing_element.step is not None else 1)
                offset = int(
                    (
                        indexing_element.start
                        if indexing_element.start >= 0
                        else indexing_element.start + dimension_size
                    )
                    if indexing_element.start is not None
                    else (0 if stride > 0 else dimension_size - 1)
                )

            else:
                size = 1
                stride = 1
                offset = int(
                    indexing_element if indexing_element >= 0 else indexing_element + dimension_size
                )

            offsets.append(offset)
            sizes.append(size)
            strides.append(stride)

        if x.is_encrypted and y.is_clear:
            encrypted_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                    shape=y.shape,
                    is_encrypted=True,
                )
            )
            y = self.encrypt(encrypted_type, y)

        required_y_shape_list = []
        for i, indexing_element in enumerate(index):
            if isinstance(indexing_element, slice):
                n = len(np.zeros(x.shape[i])[indexing_element])
                required_y_shape_list.append(n)
            else:
                required_y_shape_list.append(1)

        required_y_shape = tuple(required_y_shape_list)
        try:
            np.reshape(np.zeros(y.shape), required_y_shape)
            y = self.reshape(y, required_y_shape)
        except Exception:  # pylint: disable=broad-except
            np.broadcast_to(np.zeros(y.shape), required_y_shape)
            y = self.broadcast_to(y, required_y_shape)

        y = self.to_signedness(y, of=resulting_type)

        return self.operation(
            tensor.InsertSliceOp,
            resulting_type,
            y.result,
            x.result,
            (),
            (),
            (),
            MlirDenseI64ArrayAttr.get(offsets),
            MlirDenseI64ArrayAttr.get(sizes),
            MlirDenseI64ArrayAttr.get(strides),
            original_bit_width=x.original_bit_width,
        )

    def bitwise(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        operation: Callable[[int, int], int],
    ) -> Conversion:
        if x.is_signed or y.is_signed:
            highlights: Dict[Node, Union[str, List[str]]] = {
                self.converting: "but only unsigned-unsigned bitwise operations are supported",
            }
            if x.is_signed:
                highlights[x.origin] = "lhs is signed"
            if y.is_signed:
                highlights[y.origin] = (
                    "rhs is signed" if x.origin is not y.origin else "operand is signed"
                )
            self.error(highlights)

        if x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit bitwise operations are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to a bitwise operation"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(  # type: ignore
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)
        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        if x.original_bit_width + y.original_bit_width <= resulting_type.bit_width:
            shifter_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=False, bit_width=(x.bit_width + 1)),
                    shape=(),
                    is_encrypted=False,
                )
            )
            shifter = self.constant(shifter_type, 2**y.original_bit_width)

            shifted_x = self.mul(x.type, x, shifter)
            packed_x_and_y = self.add(resulting_type, shifted_x, y)

            return self.tlu(
                resulting_type,
                packed_x_and_y,
                [
                    operation(x_value, y_value)
                    for x_value in range(2**x.original_bit_width)
                    for y_value in range(2**y.original_bit_width)
                ],
            )

        bit_width = resulting_type.bit_width
        original_bit_width = max(x.original_bit_width, y.original_bit_width)

        chunk_size = max(int(np.floor(bit_width / 2)), 1)
        mask = (2**chunk_size) - 1

        chunks = []
        for offset in range(0, original_bit_width, chunk_size):
            x_lut = [((x >> offset) & mask) << chunk_size for x in range(2**bit_width)]
            y_lut = [(y >> offset) & mask for y in range(2**bit_width)]

            x_chunk = self.tlu(x.type, x, x_lut)
            y_chunk = self.tlu(y.type, y, y_lut)

            packed_x_and_y_chunks = self.add(resulting_type, x_chunk, y_chunk)
            result_chunk = self.tlu(
                resulting_type,
                packed_x_and_y_chunks,
                [
                    operation(x, y) << offset
                    for x in range(2**chunk_size)
                    for y in range(2**chunk_size)
                ],
            )

            chunks.append(result_chunk)

        return self.tree_add(resulting_type, chunks)

    def bitwise_and(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.bitwise(resulting_type, x, y, lambda a, b: a & b)

    def bitwise_or(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.bitwise(resulting_type, x, y, lambda a, b: a | b)

    def bitwise_xor(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.bitwise(resulting_type, x, y, lambda a, b: a ^ b)

    def broadcast_to(self, x: Conversion, shape: Tuple[int, ...]):
        if x.is_clear:
            highlights = {
                x.origin: "value is clear",
                self.converting: "but clear values cannot be broadcasted",
            }
            self.error(highlights)

        if x.shape == shape:
            return x

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=shape,
                is_encrypted=x.is_encrypted,
            )
        )

        return self.add(resulting_type, x, self.zeros(resulting_type))

    def concatenate(
        self,
        resulting_type: ConversionType,
        xs: List[Conversion],
        axis: Optional[int],
    ) -> Conversion:
        if resulting_type.is_clear:
            highlights = {x.origin: "value is clear" for x in xs}
            highlights[self.converting] = "but clear concatenation is not supported"
            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, *xs)

        sanitized_xs = []
        for x in xs:
            if x.is_clear:
                encrypted_type = self.typeof(
                    ValueDescription(
                        dtype=Integer(
                            is_signed=resulting_type.is_signed,
                            bit_width=resulting_type.bit_width,
                        ),
                        shape=x.shape,
                        is_encrypted=True,
                    )
                )
                x = self.encrypt(encrypted_type, x)
            sanitized_xs.append(self.to_signedness(x, of=resulting_type))

        if axis is None:
            return self.operation(
                fhelinalg.ConcatOp,
                resulting_type,
                (self.flatten(x).result for x in sanitized_xs),
                axis=MlirIntegerAttr.get(self.i(64).mlir, 0),
            )

        if axis < 0:
            axis += len(sanitized_xs[0].shape)

        return self.operation(
            fhelinalg.ConcatOp,
            resulting_type,
            (x.result for x in sanitized_xs),
            axis=MlirIntegerAttr.get(self.i(64).mlir, axis),
        )

    def constant(self, resulting_type: ConversionType, data: Any) -> Conversion:
        assert resulting_type.is_clear

        attribute = self.attribute(resulting_type, data)

        cache_key = attribute
        cached_conversion = self.constant_cache.get(cache_key)

        if cached_conversion is None:
            cached_conversion = Conversion(
                self.converting,
                arith.ConstantOp(resulting_type, attribute),
            )

            try:
                cached_conversion.set_original_bit_width(Integer.that_can_represent(data).bit_width)
            except Exception:  # pylint: disable=broad-except
                pass

            self.constant_cache[cache_key] = cached_conversion

        return cached_conversion

    def conv2d(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        weight: Conversion,
        bias: Optional[Conversion],
        strides: Sequence[int],
        dilations: Sequence[int],
        pads: Sequence[int],
        group: int,
    ):
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear convolutions are not supported",
            }
            self.error(highlights)

        if weight.is_encrypted:
            highlights = {
                weight.origin: "weight is encrypted",
                self.converting: "but convolutions with encrypted weights are not supported",
            }
            self.error(highlights)

        if bias is not None and bias.is_encrypted:
            highlights = {
                bias.origin: "bias is encrypted",
                self.converting: "but convolutions with encrypted biases are not supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted
        assert self.is_bit_width_compatible(resulting_type, x, weight, bias)

        x = self.to_signedness(x, of=resulting_type)
        weight = self.to_signedness(weight, of=resulting_type)
        bias = self.to_signedness(bias, of=resulting_type) if bias is not None else None

        strides = MlirDenseElementsAttr.get(
            np.array(list(strides), dtype=np.uint64),
            type=self.i(64).mlir,
        )
        dilations = MlirDenseElementsAttr.get(
            np.array(list(dilations), dtype=np.uint64),
            type=self.i(64).mlir,
        )
        padding = MlirDenseElementsAttr.get(
            np.array(list(pads), dtype=np.uint64),
            type=self.i(64).mlir,
        )
        group = MlirIntegerAttr.get(self.i(64).mlir, group)

        return self.operation(
            fhelinalg.Conv2dOp,
            resulting_type,
            x.result,
            weight.result,
            bias=(bias.result if bias is not None else None),
            padding=padding,
            strides=strides,
            dilations=dilations,
            group=group,
        )

    def dot(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        if x.is_clear and y.is_clear:
            highlights: Dict[Node, Union[str, List[str]]] = {
                x.origin: "lhs is clear",
                y.origin: "rhs is clear" if x.origin is not y.origin else "operand is clear",
                self.converting: "but clear-clear dot products are not supported",
            }
            self.error(highlights)

        if (x.is_encrypted and y.is_encrypted) and (
            x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH
        ):
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit "
                    f"encrypted dot products are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to an encrypted dot products"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(  # type: ignore
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)

        if x.is_scalar or y.is_scalar:
            return self.mul(resulting_type, x, y)

        operation = fhelinalg.DotEint if x.is_encrypted and y.is_encrypted else fhelinalg.Dot

        if x.is_clear:
            x, y = y, x

        if (x.is_signed or y.is_signed) and resulting_type.is_unsigned:
            x = self.to_signed(x)
            y = self.to_signed(y)

            signed_resulting_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=True, bit_width=resulting_type.bit_width),
                    shape=resulting_type.shape,
                    is_encrypted=resulting_type.is_encrypted,
                )
            )
            intermediate_result = self.operation(
                operation,
                signed_resulting_type,
                x.result,
                y.result,
            )

            return self.to_unsigned(intermediate_result)

        x = self.to_signedness(x, of=resulting_type)
        y = self.to_signedness(y, of=resulting_type)

        return self.operation(operation, resulting_type, x.result, y.result)

    def encrypt(self, resulting_type: ConversionType, x: Conversion) -> Conversion:
        assert self.is_bit_width_compatible(resulting_type, x)
        assert resulting_type.is_encrypted and x.is_clear
        assert x.shape == resulting_type.shape

        return self.add(resulting_type, x, self.zeros(resulting_type))

    def equality(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        equals: bool,
    ) -> Conversion:
        if x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit "
                    f"comparison operations are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to a comparison operation"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)
        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        x_dtype = Integer(is_signed=x.is_signed, bit_width=x.original_bit_width)
        y_dtype = Integer(is_signed=y.is_signed, bit_width=y.original_bit_width)

        x_minus_y_min = x_dtype.min() - y_dtype.max()
        x_minus_y_max = x_dtype.max() - y_dtype.min()

        x_minus_y_range = [x_minus_y_min, x_minus_y_max]
        x_minus_y_dtype = Integer.that_can_represent(x_minus_y_range)

        bit_width = resulting_type.bit_width

        signed_offset = 2 ** (bit_width - 1)
        sanitizer = self.constant(self.i(resulting_type.bit_width + 1), signed_offset)

        if x_minus_y_dtype.bit_width <= bit_width:
            x_minus_y = self.sub(resulting_type, x, y)
            sanitized_x_minus_y = self.add(resulting_type, x_minus_y, sanitizer)

            zero_position = 2 ** (bit_width - 1)
            if equals:
                operation_lut = [int(i == zero_position) for i in range(2**bit_width)]
            else:
                operation_lut = [int(i != zero_position) for i in range(2**bit_width)]

            return self.tlu(resulting_type, sanitized_x_minus_y, operation_lut)

        x = self.broadcast_to(x, resulting_type.shape)
        y = self.broadcast_to(y, resulting_type.shape)

        chunk_size = max(int(np.floor(bit_width / 2)), 1)
        number_of_chunks = int(np.ceil(bit_width / chunk_size))

        if x.is_signed != y.is_signed:
            number_of_chunks += 1

        greater_than_half_lut = [
            int(i >= signed_offset) << (number_of_chunks - 1) for i in range(2**bit_width)
        ]
        if x.is_unsigned and y.is_signed:
            is_unsigned_greater_than_half = self.tlu(
                resulting_type,
                x,
                greater_than_half_lut,
            )
        elif x.is_signed and y.is_unsigned:
            is_unsigned_greater_than_half = self.tlu(
                resulting_type,
                y,
                greater_than_half_lut,
            )
        else:
            is_unsigned_greater_than_half = None

        offset_x_by = 0
        offset_y_by = 0

        if x.is_signed or y.is_signed:
            if x.is_signed:
                x = self.add(resulting_type, x, sanitizer)
            else:
                offset_x_by = signed_offset

            if y.is_signed:
                y = self.add(resulting_type, y, sanitizer)
            else:
                offset_y_by = signed_offset

        carries = self.pack_to_chunk_groups_and_map(
            resulting_type,
            max(x.original_bit_width, y.original_bit_width),
            chunk_size,
            x,
            y,
            lambda _, a, b: int(a != b),
            x_offset=offset_x_by,
            y_offset=offset_y_by,
        )

        if is_unsigned_greater_than_half:
            carries.append(is_unsigned_greater_than_half)

        return self.tlu(
            resulting_type,
            self.tree_add(resulting_type, carries),
            [int(i == 0 if equals else i != 0) for i in range(2**bit_width)],
        )

    def flatten(self, x: Conversion) -> Conversion:
        return self.reshape(x, shape=(int(np.prod(x.shape)),))

    def greater(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        equal: bool = False,
    ) -> Conversion:
        # pylint: disable=arguments-out-of-order
        return self.less(resulting_type, y, x, equal=equal)
        # pylint: enable=arguments-out-of-order

    def greater_equal(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.greater(resulting_type, x, y, equal=True)

    def index_static(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        index: Sequence[Union[int, np.integer, slice]],
    ) -> Conversion:
        assert self.is_bit_width_compatible(resulting_type, x)
        assert resulting_type.is_encrypted == x.is_encrypted

        x = self.to_signedness(x, of=resulting_type)

        index = list(index)
        while len(index) < len(x.shape):
            index.append(slice(None, None, None))

        if all(isinstance(i, (int, np.integer)) for i in index):
            indices = []
            for indexing_element, dimension_size in zip(index, x.shape):
                indexing_element = int(indexing_element)  # type: ignore
                if indexing_element < 0:
                    indexing_element += dimension_size
                indices.append(self.constant(self.index_type(), indexing_element).result)

            return self.operation(
                tensor.ExtractOp,
                resulting_type,
                x.result,
                tuple(indices),
                original_bit_width=x.original_bit_width,
            )

        offsets = []
        sizes = []
        strides = []

        destroyed_dimensions = []
        for dimension, (indexing_element, dimension_size) in enumerate(zip(index, x.shape)):
            if isinstance(indexing_element, slice):
                size = int(np.zeros(dimension_size)[indexing_element].shape[0])
                stride = int(indexing_element.step if indexing_element.step is not None else 1)
                offset = int(
                    (
                        indexing_element.start
                        if indexing_element.start >= 0
                        else indexing_element.start + dimension_size
                    )
                    if indexing_element.start is not None
                    else (0 if stride > 0 else dimension_size - 1)
                )

            else:
                destroyed_dimensions.append(dimension)
                size = 1
                stride = 1
                offset = int(
                    indexing_element
                    if indexing_element >= 0
                    else indexing_element + dimension_size,
                )

            offsets.append(offset)
            sizes.append(size)
            strides.append(stride)

        if len(destroyed_dimensions) == 0:
            return self.operation(
                tensor.ExtractSliceOp,
                resulting_type,
                x.result,
                (),
                (),
                (),
                MlirDenseI64ArrayAttr.get(offsets),
                MlirDenseI64ArrayAttr.get(sizes),
                MlirDenseI64ArrayAttr.get(strides),
                original_bit_width=x.original_bit_width,
            )

        intermediate_shape = list(resulting_type.shape)
        for dimension in destroyed_dimensions:
            intermediate_shape.insert(dimension, 1)

        intermediate_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=tuple(intermediate_shape),
                is_encrypted=x.is_encrypted,
            )
        )

        intermediate = self.operation(
            tensor.ExtractSliceOp,
            intermediate_type,
            x.result,
            (),
            (),
            (),
            MlirDenseI64ArrayAttr.get(offsets),
            MlirDenseI64ArrayAttr.get(sizes),
            MlirDenseI64ArrayAttr.get(strides),
        )

        reassociaton = []

        current_intermediate_dimension = 0
        for _ in range(len(resulting_type.shape)):
            indices = [current_intermediate_dimension]
            while current_intermediate_dimension in destroyed_dimensions:
                current_intermediate_dimension += 1
                indices.append(current_intermediate_dimension)

            reassociaton.append(indices)
            current_intermediate_dimension += 1
        while current_intermediate_dimension < len(intermediate_shape):
            reassociaton[-1].append(current_intermediate_dimension)
            current_intermediate_dimension += 1

        return self.operation(
            tensor.CollapseShapeOp,
            resulting_type,
            intermediate.result,
            MlirArrayAttr.get(
                [
                    MlirArrayAttr.get(
                        [MlirIntegerAttr.get(self.i(64).mlir, index) for index in indices],
                    )
                    for indices in reassociaton
                ],
            ),
        )

    def less(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        equal: bool = False,
    ) -> Conversion:
        if x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit "
                    f"comparison operations are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to a comparison operation"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)
        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        x_dtype = Integer(is_signed=x.is_signed, bit_width=x.original_bit_width)
        y_dtype = Integer(is_signed=y.is_signed, bit_width=x.original_bit_width)

        x_minus_y_min = x_dtype.min() - y_dtype.max()
        x_minus_y_max = x_dtype.max() - y_dtype.min()

        x_minus_y_range = [x_minus_y_min, x_minus_y_max]
        x_minus_y_dtype = Integer.that_can_represent(x_minus_y_range)

        bit_width = resulting_type.bit_width

        if x_minus_y_dtype.bit_width <= bit_width:
            x_minus_y = self.to_signed(self.sub(resulting_type, x, y))

            accept_equal = int(equal)
            accept_greater = 0
            accept_less = 1

            all_cells = 2**bit_width

            equal_cells = 1
            greater_cells = (2 ** (bit_width - 1)) - 1
            less_cells = 2 ** (bit_width - 1)

            assert equal_cells + greater_cells + less_cells == all_cells

            table = (
                [accept_equal] * equal_cells
                + [accept_greater] * greater_cells
                + [accept_less] * less_cells
            )
            return self.tlu(resulting_type, x_minus_y, table)

        # Comparison between signed and unsigned is tricky.
        # To deal with them, we add -min of the signed number to both operands
        # such that they are both positive. To avoid overflowing
        # the unsigned operand this addition is done "virtually"
        # while constructing one of the luts.

        # A flag ("is_unsigned_greater_than_half") is emitted in MLIR to keep track
        # if the unsigned operand was greater than the max signed number as it
        # is needed to determine the result of the comparison.

        # Exemple: to compare x and y where x is an int3 and y and uint3, when y
        # is greater than 4 we are sure than x will be less than x.

        x = self.broadcast_to(x, resulting_type.shape)
        y = self.broadcast_to(y, resulting_type.shape)

        offset_x_by = 0
        offset_y_by = 0

        signed_offset = 2 ** (bit_width - 1)
        sanitizer = self.constant(self.i(resulting_type.bit_width + 1), signed_offset)

        x_was_signed = x.is_signed
        y_was_signed = y.is_signed

        x_was_unsigned = x.is_unsigned
        y_was_unsigned = y.is_unsigned

        if x.is_signed or y.is_signed:
            if x.is_signed:
                x = self.to_unsigned(self.add(resulting_type, x, sanitizer))
            else:
                offset_x_by = signed_offset

            if y.is_signed:
                y = self.to_unsigned(self.add(resulting_type, y, sanitizer))
            else:
                offset_y_by = signed_offset

        # pylint: disable=invalid-name

        EQUAL = 0b00
        LESS = 0b01
        GREATER = 0b10
        UNUSED = 0b11

        # pylint: enable=invalid-name

        def compare(a, b):
            if a < b:
                return LESS

            if a > b:
                return GREATER

            return EQUAL

        chunk_size = int(np.floor(bit_width / 2))
        carries = self.pack_to_chunk_groups_and_map(
            resulting_type,
            max(x.original_bit_width, y.original_bit_width),
            chunk_size,
            x,
            y,
            lambda i, a, b: compare(a, b) << (min(i, 1) * 2),
            x_offset=offset_x_by,
            y_offset=offset_y_by,
        )

        # This is the reduction step -- we have an array where the entry i is the
        # result of comparing the chunks of x and y at position i.

        all_comparisons = [EQUAL, LESS, GREATER, UNUSED]
        pick_first_not_equal_lut = [
            int(current_comparison if previous_comparison == EQUAL else previous_comparison)
            for current_comparison in all_comparisons
            for previous_comparison in all_comparisons
        ]

        carry = carries[0]
        for next_carry in carries[1:]:
            combined_carries = self.add(resulting_type, next_carry, carry)
            carry = self.tlu(resulting_type, combined_carries, pick_first_not_equal_lut)

        if x_was_signed != y_was_signed:
            carry_bit_width = 2
            is_less_mask = int(LESS)

            is_unsigned_greater_than_half = self.tlu(
                resulting_type,
                x if x_was_unsigned else y,
                [int(value >= signed_offset) << carry_bit_width for value in range(2**bit_width)],
            )
            packed_carry_and_is_unsigned_greater_than_half = self.add(
                resulting_type,
                is_unsigned_greater_than_half,
                carry,
            )

            # this function is actually converting either
            # - lhs < rhs
            # - lhs <= rhs

            # in the implementation, we call
            # - x = lhs
            # - y = rhs

            # so if y is unsigned and greater than half
            # - y is definitely bigger than x
            # - is_unsigned_greater_than_half == 1
            # - result ==  (lhs < rhs) == (x < y) == 1

            # so if x is unsigned and greater than half
            # - x is definitely bigger than y
            # - is_unsigned_greater_than_half == 1
            # - result ==  (lhs < rhs) == (x < y) == 0

            if y_was_unsigned:
                result_table = [
                    1 if (i >> carry_bit_width) else (i & is_less_mask) for i in range(2**3)
                ]
            else:
                result_table = [
                    0 if (i >> carry_bit_width) else (i & is_less_mask) for i in range(2**3)
                ]

            result = self.tlu(
                resulting_type,
                packed_carry_and_is_unsigned_greater_than_half,
                result_table,
            )
        else:
            accept = {LESS}
            if equal:
                accept.add(EQUAL)

            boolean_result_lut = [int(comparison in accept) for comparison in all_comparisons]
            result = self.tlu(resulting_type, carry, boolean_result_lut)

        return result

    def less_equal(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.less(resulting_type, x, y, equal=True)

    def matmul(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        if x.is_clear and y.is_clear:
            highlights: Dict[Node, Union[str, List[str]]] = {
                x.origin: "lhs is clear",
                y.origin: "rhs is clear" if x.origin is not y.origin else "operand is clear",
                self.converting: "but clear-clear matrix multiplications are not supported",
            }
            self.error(highlights)

        if (x.is_encrypted and y.is_encrypted) and (
            x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH
        ):
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit "
                    f"encrypted matrix multiplications are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to an encrypted matrix multiplication"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(  # type: ignore
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)

        if resulting_type.shape == ():
            if x.is_clear:
                x, y = y, x
            operation = fhelinalg.DotEint if x.is_encrypted and y.is_encrypted else fhelinalg.Dot
        elif x.is_encrypted and y.is_encrypted:
            operation = fhelinalg.MatMulEintEintOp
        else:
            operation = fhelinalg.MatMulEintIntOp if x.is_encrypted else fhelinalg.MatMulIntEintOp

        if (x.is_signed or y.is_signed) and resulting_type.is_unsigned:
            x = self.to_signed(x)
            y = self.to_signed(y)

            signed_resulting_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=True, bit_width=resulting_type.bit_width),
                    shape=resulting_type.shape,
                    is_encrypted=resulting_type.is_encrypted,
                )
            )
            intermediate_result = self.operation(
                operation,
                signed_resulting_type,
                x.result,
                y.result,
            )

            return self.to_unsigned(intermediate_result)

        x = self.to_signedness(x, of=resulting_type)
        y = self.to_signedness(y, of=resulting_type)

        return self.operation(operation, resulting_type, x.result, y.result)

    def maxpool2d(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        kernel_shape: Tuple[int, ...],
        strides: Sequence[int],
        dilations: Sequence[int],
    ):
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear maxpooling is not supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted
        assert self.is_bit_width_compatible(resulting_type, x)

        kernel_shape = MlirDenseElementsAttr.get(
            np.array(list(kernel_shape), dtype=np.uint64),
            type=self.i(64).mlir,
        )
        strides = MlirDenseElementsAttr.get(
            np.array(list(strides), dtype=np.uint64),
            type=self.i(64).mlir,
        )
        dilations = MlirDenseElementsAttr.get(
            np.array(list(dilations), dtype=np.uint64),
            type=self.i(64).mlir,
        )

        same_signedness_resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=resulting_type.shape,
                is_encrypted=True,
            )
        )
        result = self.operation(
            fhelinalg.Maxpool2dOp,
            same_signedness_resulting_type,
            x.result,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
        )

        return self.to_signedness(result, of=resulting_type)

    def mul(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        if x.is_clear and y.is_clear:
            highlights = {
                x.origin: ["lhs is clear"],
                y.origin: ["rhs is clear" if x.origin is not y.origin else "operand is clear"],
                self.converting: ["but clear-clear multiplications are not supported"],
            }
            self.error(highlights)

        if (x.is_encrypted and y.is_encrypted) and (
            x.bit_width > MAXIMUM_TLU_BIT_WIDTH or y.bit_width > MAXIMUM_TLU_BIT_WIDTH
        ):
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit "
                    f"encrypted multiplications are supported"
                ],
            }

            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to an encrypted multiplication"
                    ]
                    if operand.bit_width != operand.original_bit_width:
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, y)

        use_linalg = x.is_tensor or y.is_tensor

        x = self.tensorize(x) if use_linalg else x
        y = self.tensorize(y) if use_linalg else y

        if x.is_clear:
            x, y = y, x

        dialect = fhelinalg if use_linalg else fhe
        operation = dialect.MulEintIntOp if y.is_clear else dialect.MulEintOp

        if (x.is_signed or y.is_signed) and resulting_type.is_unsigned:
            x = self.to_signed(x)
            y = self.to_signed(y)

            signed_resulting_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=True, bit_width=resulting_type.bit_width),
                    shape=resulting_type.shape,
                    is_encrypted=resulting_type.is_encrypted,
                )
            )
            intermediate_result = self.operation(
                operation,
                signed_resulting_type,
                x.result,
                y.result,
            )

            return self.to_unsigned(intermediate_result)

        x = self.to_signedness(x, of=resulting_type)
        y = self.to_signedness(y, of=resulting_type)

        return self.operation(
            operation,
            resulting_type,
            x.result,
            y.result,
        )

    def multi_tlu(
        self,
        resulting_type: ConversionType,
        on: Conversion,
        tables: Any,
        mapping: Any,
    ):
        if on.is_clear:
            highlights = {
                on.origin: "this clear value is used as an input to a table lookup",
                self.converting: "but only encrypted table lookups are supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted
        assert on.bit_width <= MAXIMUM_TLU_BIT_WIDTH

        mapping = np.array(mapping, dtype=np.uint64)

        on = self.broadcast_to(on, mapping.shape)

        assert mapping.shape == on.shape
        assert mapping.min() == 0
        assert mapping.max() == len(tables) - 1

        for table in tables:
            assert len(table) == 2**on.bit_width

        mapping_shape = mapping.shape
        tables_shape = (len(tables), 2**on.bit_width)

        mapping = self.constant(self.tensor(self.index_type(), shape=mapping_shape), mapping)
        tables = self.constant(self.tensor(self.i(64), shape=tables_shape), tables)

        return self.operation(
            fhelinalg.ApplyMappedLookupTableEintOp,
            resulting_type,
            on.result,
            tables.result,
            mapping.result,
        )

    def neg(self, resulting_type: ConversionType, x: Conversion) -> Conversion:
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear negations are not supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted

        dialect = fhe if x.is_scalar else fhelinalg
        operation = dialect.NegEintOp

        x = self.to_signed(x)
        result = self.operation(operation, x.type, x.result)

        return self.to_signedness(result, of=resulting_type)

    def ones(self, resulting_type: ConversionType) -> Conversion:
        assert resulting_type.is_encrypted

        one_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=False, bit_width=(resulting_type.bit_width + 1)),
                shape=(),
                is_encrypted=False,
            )
        )
        one = self.constant(one_type, 1)

        return self.add(resulting_type, one, self.zeros(resulting_type))

    def reshape(self, x: Conversion, shape: Tuple[int, ...]) -> Conversion:
        if x.is_scalar:
            x = self.tensorize(x)

        input_shape = x.shape
        output_shape = shape

        if input_shape == output_shape:
            return x

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=output_shape,
                is_encrypted=x.is_encrypted,
            )
        )

        # we can either collapse or expand, which changes the number of dimensions
        # this is a limitation of the current compiler, it will be improved in the future (#1060)
        can_be_converted_directly = len(input_shape) != len(output_shape)

        reassociation: List[List[int]] = []
        if can_be_converted_directly:
            if len(output_shape) == 1:
                # output is 1 dimensional so collapse every dimension into the same dimension
                reassociation.append(list(range(len(input_shape))))
            else:
                # input is m dimensional
                # output is n dimensional
                # and m is different from n

                # we don't want to duplicate code, so we forget about input and output,
                # and we focus on smaller shape and bigger shape

                smaller_shape, bigger_shape = (
                    (output_shape, input_shape)
                    if len(output_shape) < len(input_shape)
                    else (input_shape, output_shape)
                )
                s_index, b_index = 0, 0

                # now we will figure out how to group the bigger shape to get the smaller shape
                # think of the algorithm below as
                #     keep merging the dimensions of the bigger shape
                #     until we have a match on the smaller shape
                #     then try to match the next dimension of the smaller shape
                #     if all dimensions of the smaller shape is matched
                #     we can convert it

                group = []
                size = 1
                while s_index < len(smaller_shape) and b_index < len(bigger_shape):
                    # dimension `b_index` of `bigger_shape` belongs to current group
                    group.append(b_index)

                    # and current group has `size * bigger_shape[b_index]` elements now
                    size *= bigger_shape[b_index]

                    # if current group size matches the dimension `s_index` of `smaller_shape`
                    if size == smaller_shape[s_index]:
                        # we finalize this group and reset everything
                        size = 1
                        reassociation.append(group)
                        group = []

                        # now try to match the next dimension of `smaller_shape`
                        s_index += 1

                    # now process the next dimension of `bigger_shape`
                    b_index += 1

                # handle the case where bigger shape has proceeding 1s
                # e.g., (5,) -> (5, 1)
                while b_index < len(bigger_shape) and bigger_shape[b_index] == 1:
                    reassociation[-1].append(b_index)
                    b_index += 1

                # if not all dimensions of both shapes are processed exactly
                if s_index != len(smaller_shape) or b_index != len(bigger_shape):
                    # we cannot convert
                    can_be_converted_directly = False

        if can_be_converted_directly:
            operation = (
                tensor.CollapseShapeOp
                if len(output_shape) < len(input_shape)
                else tensor.ExpandShapeOp
            )
            reassociation_attr = MlirArrayAttr.get(
                [
                    MlirArrayAttr.get(
                        [MlirIntegerAttr.get(self.i(64).mlir, dimension) for dimension in group]
                    )
                    for group in reassociation
                ]
            )
            return self.operation(
                operation,
                resulting_type,
                x.result,
                reassociation_attr,
            )

        flattened_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=(int(np.prod(input_shape)),),
                is_encrypted=x.is_encrypted,
            )
        )
        flattened = self.operation(
            tensor.CollapseShapeOp,
            flattened_type,
            x.result,
            MlirArrayAttr.get(
                [
                    MlirArrayAttr.get(
                        [MlirIntegerAttr.get(self.i(64).mlir, i) for i in range(len(input_shape))]
                    )
                ]
            ),
        )

        return self.operation(
            tensor.ExpandShapeOp,
            resulting_type,
            flattened.result,
            MlirArrayAttr.get(
                [
                    MlirArrayAttr.get(
                        [MlirIntegerAttr.get(self.i(64).mlir, i) for i in range(len(output_shape))]
                    )
                ]
            ),
        )

    def round_bit_pattern(self, x: Conversion, lsbs_to_remove: int) -> Conversion:
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear round bit pattern is not supported",
            }
            self.error(highlights)

        assert x.bit_width > lsbs_to_remove

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=(x.bit_width - lsbs_to_remove)),
                shape=x.shape,
                is_encrypted=x.is_encrypted,
            )
        )

        return self.operation(
            fhe.RoundEintOp if x.is_scalar else fhelinalg.RoundOp,
            resulting_type,
            x.result,
        )

    def shift(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        b: Conversion,
        orientation: str,
        original_resulting_bit_width: int,
    ) -> Conversion:
        if x.bit_width > MAXIMUM_TLU_BIT_WIDTH or 2**b.original_bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights: Dict[Node, Union[str, List[str]]] = {
                self.converting: [
                    f"but only up to {round(np.log2(MAXIMUM_TLU_BIT_WIDTH))}-bit shift operations "
                    f"on up to {MAXIMUM_TLU_BIT_WIDTH}-bit operands are supported"
                ],
            }

            if x.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                highlights[x.origin] = [
                    f"this {x.bit_width}-bit value " f"is used as the operand of a shift operation"
                ]
                if x.bit_width != x.original_bit_width:
                    assert isinstance(highlights[x.origin], list)
                    highlights[x.origin].append(  # type: ignore
                        "("
                        f"note that it's assigned {x.bit_width}-bits "
                        f"during compilation because of its relation with other operations"
                        ")"
                    )

            if 2**b.original_bit_width > MAXIMUM_TLU_BIT_WIDTH:
                highlights[b.origin] = [
                    f"this {b.original_bit_width}-bit value "
                    f"is used as the shift amount of a shift operation"
                ]

            self.error(highlights)

        if x.is_signed or b.is_signed:
            highlights = {
                self.converting: "but only unsigned-unsigned bitwise shifts are supported",
            }
            if x.is_signed:
                highlights[x.origin] = "lhs is signed"
            if b.is_signed:
                highlights[b.origin] = (
                    "rhs is signed" if x.origin is not b.origin else "operand is signed"
                )
            self.error(highlights)

        assert self.is_bit_width_compatible(resulting_type, x, b)
        assert resulting_type.is_encrypted and x.is_encrypted and b.is_encrypted

        bit_width = resulting_type.bit_width

        if x.shape != resulting_type.shape:
            x = self.add(resulting_type, x, self.zeros(resulting_type))

        if x.original_bit_width + b.original_bit_width <= bit_width:
            shift_multiplier_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=False, bit_width=(x.bit_width + 1)),
                    shape=(),
                    is_encrypted=False,
                )
            )
            shift_multiplier = self.constant(shift_multiplier_type, 2**b.original_bit_width)

            shifted_x = self.mul(resulting_type, x, shift_multiplier)
            packed_x_and_b = self.add(resulting_type, shifted_x, b)

            return self.tlu(
                resulting_type,
                packed_x_and_b,
                [
                    (x_value << b_value) if orientation == "left" else (x_value >> b_value)
                    for x_value in range(2**x.original_bit_width)
                    for b_value in range(2**b.original_bit_width)
                ],
            )

        # Left_shifts of x << b can be done as follows:
        # - left shift of x by 8 if b & 0b1000 > 0
        # - left shift of x by 4 if b & 0b0100 > 0
        # - left shift of x by 2 if b & 0b0010 > 0
        # - left shift of x by 1 if b & 0b0001 > 0

        # Encoding this condition is non-trivial -- however,
        # it can be done using the following trick:

        # y = (b & 0b1000 > 0) * ((x << 8) - x) + x

        # When b & 0b1000, then:
        #   y = 1 * ((x << 8) - x) + x = (x << 8) - x + x = x << 8

        # When b & 0b1000 == 0 then:
        #   y = 0 * ((x << 8) - x) + x = x

        # The same trick can be used for right shift but with:
        # y = x - (b & 0b1000 > 0) * (x - (x >> 8))

        original_bit_width = original_resulting_bit_width
        chunk_size = min(original_bit_width, bit_width - 1)

        for i in reversed(range(b.original_bit_width)):
            to_check = 2**i

            should_shift = self.tlu(
                b.type,
                b,
                [int((b & to_check) > 0) for b in range(2**bit_width)],
            )
            shifted_x = self.tlu(
                resulting_type,
                x,
                (
                    [(x << to_check) - x for x in range(2**bit_width)]
                    if orientation == "left"
                    else [x - (x >> to_check) for x in range(2**bit_width)]
                ),
            )

            chunks = []
            for offset in range(0, original_bit_width, chunk_size):
                bits_to_process = min(chunk_size, original_bit_width - offset)
                right_shift_by = original_bit_width - offset - bits_to_process
                mask = (2**bits_to_process) - 1

                chunk_x = self.tlu(
                    resulting_type,
                    shifted_x,
                    [(((x >> right_shift_by) & mask) << 1) for x in range(2**bit_width)],
                )
                packed_chunk_x_and_should_shift = self.add(
                    resulting_type,
                    chunk_x,
                    should_shift,
                )

                chunk = self.tlu(
                    resulting_type,
                    packed_chunk_x_and_should_shift,
                    [
                        (x << right_shift_by) if b else 0
                        for x in range(2**chunk_size)
                        for b in [0, 1]
                    ],
                )
                chunks.append(chunk)

            difference = chunks[0]
            for chunk in chunks[1:]:
                difference = self.add(resulting_type, difference, chunk)

            x = (
                self.add(resulting_type, difference, x)
                if orientation == "left"
                else self.sub(resulting_type, x, difference)
            )

        return x

    def sub(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        assert self.is_bit_width_compatible(resulting_type, x, y)

        if x.is_clear and y.is_clear:
            highlights = {
                x.origin: "lhs is clear",
                y.origin: "rhs is clear" if x.origin is not y.origin else "operand is clear",
                self.converting: "but clear-clear subtractions are not supported",
            }
            self.error(highlights)

        x = self.to_signedness(x, of=resulting_type)
        y = self.to_signedness(y, of=resulting_type)

        use_linalg = x.is_tensor or y.is_tensor

        x = self.tensorize(x) if use_linalg else x
        y = self.tensorize(y) if use_linalg else y

        dialect = fhelinalg if use_linalg else fhe
        operation = (
            dialect.SubEintOp
            if x.is_encrypted and y.is_encrypted
            else (dialect.SubEintIntOp if y.is_clear else dialect.SubIntEintOp)
        )

        return self.operation(
            operation,
            resulting_type,
            x.result,
            y.result,
        )

    def sum(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        axes: Optional[Union[int, Sequence[int]]] = (),
        keep_dims: bool = False,
    ) -> Conversion:
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear summation is not supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted
        assert self.is_bit_width_compatible(resulting_type, x)

        if axes is None:
            axes = []
        elif isinstance(axes, int):
            axes = [axes]
        else:
            axes = list(axes)

        input_dimensions = len(x.shape)
        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] += input_dimensions

        same_signedness_resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=resulting_type.shape,
                is_encrypted=True,
            )
        )
        result = self.operation(
            fhelinalg.SumOp,
            same_signedness_resulting_type,
            x.result,
            axes=MlirArrayAttr.get(
                [MlirIntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]
            ),
            keep_dims=MlirBoolAttr.get(keep_dims),
            original_bit_width=x.original_bit_width,
        )

        return self.to_signedness(result, of=resulting_type)

    def tensorize(self, x: Conversion) -> Conversion:
        if x.is_tensor:
            return x

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
                shape=(1,),
                is_encrypted=x.is_encrypted,
            )
        )

        return self.operation(
            _FromElementsOp, resulting_type, x.result, original_bit_width=x.original_bit_width
        )

    def tlu(self, resulting_type: ConversionType, on: Conversion, table: Sequence[int]):
        if on.is_clear:
            highlights = {
                on.origin: "this clear value is used as an input to a table lookup",
                self.converting: "but only encrypted table lookups are supported",
            }
            self.error(highlights)

        assert resulting_type.is_encrypted
        assert on.bit_width <= MAXIMUM_TLU_BIT_WIDTH

        table = list(table)
        table += [0] * ((2**on.bit_width) - len(table))

        dialect = fhe if on.is_scalar else fhelinalg
        operation = dialect.ApplyLookupTableEintOp

        lut = self.constant(self.tensor(self.i(64), shape=(len(table),)), table)
        return self.operation(operation, resulting_type, on.result, lut.result)

    def to_signed(self, x: Conversion) -> Conversion:
        if x.is_signed or x.is_clear:
            return x

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=True, bit_width=x.bit_width),
                shape=x.shape,
                is_encrypted=True,
            )
        )

        dialect = fhelinalg if x.is_tensor else fhe
        operation = dialect.ToSignedOp

        return self.operation(
            operation,
            resulting_type,
            x.result,
            original_bit_width=x.original_bit_width,
        )

    def to_signedness(self, x: Conversion, of: ConversionType) -> Conversion:
        return self.to_signed(x) if of.is_signed else self.to_unsigned(x)

    def to_unsigned(self, x: Conversion) -> Conversion:
        if x.is_unsigned or x.is_clear:
            return x

        resulting_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=False, bit_width=x.bit_width),
                shape=x.shape,
                is_encrypted=True,
            )
        )

        dialect = fhelinalg if x.is_tensor else fhe
        operation = dialect.ToUnsignedOp

        return self.operation(
            operation,
            resulting_type,
            x.result,
            original_bit_width=x.original_bit_width,
        )

    def transpose(self, resulting_type: ConversionType, x: Conversion, axes: Sequence[int] = ()):
        return self.operation(
            fhelinalg.TransposeOp,
            resulting_type,
            x.result,
            axes=MlirArrayAttr.get(
                [MlirIntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]
            ),
            original_bit_width=x.original_bit_width,
        )

    def tree_add(self, resulting_type: ConversionType, xs: List[Conversion]) -> Conversion:
        while len(xs) > 1:
            a = xs.pop()
            b = xs.pop()

            result = self.add(resulting_type, a, b)
            xs.insert(0, result)

        return xs[0]

    def zeros(self, resulting_type: ConversionType) -> Conversion:
        assert resulting_type.is_encrypted

        dialect = fhe
        operation = dialect.ZeroEintOp if resulting_type.is_scalar else dialect.ZeroTensorOp

        return self.operation(
            operation,
            resulting_type,
            original_bit_width=1,
        )

    # pylint: enable=missing-function-docstring
