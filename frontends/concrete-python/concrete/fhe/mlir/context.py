"""
Declaration of `Context` class.
"""

# pylint: disable=import-error,no-name-in-module

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

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
from mlir.ir import Location as MlirLocation
from mlir.ir import OpResult as MlirOperation
from mlir.ir import RankedTensorType
from mlir.ir import Type as MlirType

from ..compilation.configuration import BitwiseStrategy, ComparisonStrategy, MinMaxStrategy
from ..dtypes import Integer
from ..extensions.bits import MAX_EXTRACTABLE_BIT, MIN_EXTRACTABLE_BIT
from ..representation import Graph, Node
from ..values import ValueDescription
from .conversion import Conversion, ConversionType
from .processors import GraphProcessor
from .utils import MAXIMUM_TLU_BIT_WIDTH, Comparison, _FromElementsOp

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
        return (
            ConversionType(RankedTensorType.get(shape, element_type.mlir))
            if shape != ()
            else element_type
        )

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

    def location(self) -> MlirLocation:
        """
        Create an MLIR location from the node that is being converted.
        """

        path, lineno = self.converting.location.rsplit(":", maxsplit=1)

        tag = "" if self.converting.tag == "" else f"@{self.converting.tag} | "
        return MlirLocation.file(
            f"{tag}{path}",
            line=int(lineno),
            col=0,
            context=self.context,
        )

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
            # since the operations are cached
            # if an operation is repeated in a different location,
            # it'll have the location of the first instance of that operation
            if operation not in [tensor.ExtractOp, tensor.InsertSliceOp]:
                result = operation(resulting_type.mlir, *args, **kwargs, loc=self.location()).result
            else:
                result = operation(*args, **kwargs, loc=self.location()).result

            cached_conversion = Conversion(self.converting, result)
            if original_bit_width is not None:
                cached_conversion.set_original_bit_width(original_bit_width)

            self.conversion_cache[cache_key] = cached_conversion

        return cached_conversion

    # comparisons

    def comparison(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        accept: Set[Comparison],
    ) -> Conversion:
        """
        Compare two encrypted values.

        Args:
            resulting_type (ConversionType):
                resulting type

            x (Conversion):
                lhs of comparison

            y (Conversion):
                rhs of comparison

            accept (Set[Comparison]):
                set of accepted comparison outcomes

        Returns:
            Conversion:
                result of comparison
        """

        if accept == {Comparison.GREATER}:  # pragma: no cover
            return self.comparison(resulting_type, y, x, accept={Comparison.LESS})

        if accept == {Comparison.GREATER, Comparison.EQUAL}:  # pragma: no cover
            return self.comparison(resulting_type, y, x, accept={Comparison.LESS, Comparison.EQUAL})

        assert accept in (
            # equal
            {Comparison.EQUAL},
            # not equal
            {Comparison.LESS, Comparison.GREATER},
            # less and inverted greater
            {Comparison.LESS},
            # less equal and inverted greater equal
            {Comparison.LESS, Comparison.EQUAL},
        )

        maximum_input_bit_width = max(x.bit_width, y.bit_width)
        if maximum_input_bit_width > MAXIMUM_TLU_BIT_WIDTH:
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
                    if operand.bit_width != operand.original_bit_width:  # pragma: no cover
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )
            self.error(highlights)

        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        x_dtype = Integer(is_signed=x.is_signed, bit_width=x.original_bit_width)
        y_dtype = Integer(is_signed=y.is_signed, bit_width=y.original_bit_width)

        x_minus_y_min = x_dtype.min() - y_dtype.max()
        x_minus_y_max = x_dtype.max() - y_dtype.min()

        x_minus_y_range = [x_minus_y_min, x_minus_y_max]
        x_minus_y_dtype = Integer.that_can_represent(x_minus_y_range)

        if x_minus_y_dtype.bit_width <= maximum_input_bit_width:
            return self.comparison_with_subtraction_trick(
                resulting_type,
                x,
                y,
                accept,
                x_minus_y_dtype,
            )

        if x.original_bit_width != y.original_bit_width:
            result = self.try_comparison_with_clipping_trick(resulting_type, x, y, accept)
            if result is not None:
                return result

        strategy_preference = self.converting.properties["strategy"]
        if strategy_preference == ComparisonStrategy.THREE_TLU_CASTED:
            return self.comparison_with_subtraction_trick(
                resulting_type,
                x,
                y,
                accept,
                x_minus_y_dtype,
            )

        return self.comparison_with_chunks(resulting_type, x, y, accept)

    def compare_with_subtraction(
        self,
        resulting_type: ConversionType,
        subtraction: Conversion,
        accept: Set[Comparison],
    ) -> Conversion:
        """
        Apply the final comparison table and return comparison result.
        """

        accept_equal = int(Comparison.EQUAL in accept)
        accept_greater = int(Comparison.GREATER in accept)
        accept_less = int(Comparison.LESS in accept)

        all_cells = 2**subtraction.bit_width

        equal_cells = 1
        greater_cells = (2 ** (subtraction.bit_width - 1)) - 1
        less_cells = 2 ** (subtraction.bit_width - 1)

        if subtraction.is_unsigned:
            greater_cells += less_cells
            less_cells = 0

        assert equal_cells + greater_cells + less_cells == all_cells

        return self.tlu(
            resulting_type,
            subtraction,
            (
                [accept_equal] * equal_cells
                + [accept_greater] * greater_cells
                + [accept_less] * less_cells
            ),
        )

    def comparison_with_subtraction_trick(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        accept: Set[Comparison],
        x_minus_y_dtype: Integer,
    ) -> Conversion:
        """
        Compare encrypted values using subtraction trick.

        Idea:
            x [.] y <==> (x - y) [.] 0 where [.] is one of <,<=,==,!=,>=,>

        Additional Args:
            x_minus_y_dtype (Integer):
                minimal dtype that can be used to store x - y without overflows
        """

        maximum_input_bit_width = max(x.bit_width, y.bit_width)

        intermediate_bit_width = max(x_minus_y_dtype.bit_width, maximum_input_bit_width)
        intermediate_scalar_type = self.esint(intermediate_bit_width)

        if x.bit_width != intermediate_bit_width:
            x = self.cast(self.tensor(intermediate_scalar_type, x.shape), x)
        if y.bit_width != intermediate_bit_width:
            y = self.cast(self.tensor(intermediate_scalar_type, y.shape), y)

        assert x.bit_width == y.bit_width

        x = self.to_signed(x)
        y = self.to_signed(y)

        intermediate_type = self.tensor(intermediate_scalar_type, resulting_type.shape)
        x_minus_y = self.sub(intermediate_type, x, y)

        if not x_minus_y_dtype.is_signed:
            x_minus_y = self.to_unsigned(x_minus_y)

        return self.compare_with_subtraction(resulting_type, x_minus_y, accept)

    def try_comparison_with_clipping_trick(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        accept: Set[Comparison],
    ) -> Optional[Conversion]:
        """
        Compare encrypted values using clipping trick.

        Idea:
            x [.] y <==> (clipped(x) - y) [.] 0 where [.] is one of <,<=,==,!=,>=,>
            or
            x [.] y <==> (x - clipped(y)) [.] 0 where [.] is one of <,<=,==,!=,>=,>
            where
            clipped(value) = np.clip(value, smaller.min() - 1, smaller.max() + 1)

        Additional Args:
            smaller_minus_clipped_bigger_dtype (Integer):
                minimal dtype that can be used to store smaller - clipped(bigger) without overflows

            clipped_bigger_minus_smaller_dtype (Integer):
                minimal dtype that can be used to store clipped(bigger) - smaller without overflows

            smaller_bounds (Tuple[int, int]):
                bounds of smaller

            smaller_is_lhs (bool):
                whether smaller is lhs of the comparison

            smaller_is_rhs (bool):
                whether smaller is rhs of the comparison
        """

        assert x.original_bit_width != y.original_bit_width

        x_is_smaller = x.original_bit_width < y.original_bit_width
        smaller, bigger = (x, y) if x_is_smaller else (y, x)

        smaller_dtype = Integer(smaller.is_signed, bit_width=smaller.original_bit_width)
        bigger_dtype = Integer(bigger.is_signed, bit_width=bigger.original_bit_width)

        smaller_bounds = [
            smaller_dtype.min(),
            smaller_dtype.max(),
        ]
        clipped_bigger_bounds = [
            np.clip(smaller_dtype.min() - 1, bigger_dtype.min(), bigger_dtype.max()),
            np.clip(smaller_dtype.max() + 1, bigger_dtype.min(), bigger_dtype.max()),
        ]

        assert clipped_bigger_bounds[0] >= smaller_dtype.min() - 1
        assert clipped_bigger_bounds[1] <= smaller_dtype.max() + 1

        assert clipped_bigger_bounds[0] >= bigger_dtype.min()
        assert clipped_bigger_bounds[1] <= bigger_dtype.max()

        smaller_minus_clipped_bigger_range = [
            smaller_bounds[0] - clipped_bigger_bounds[1],
            smaller_bounds[1] - clipped_bigger_bounds[0],
        ]
        smaller_minus_clipped_bigger_dtype = Integer.that_can_represent(
            smaller_minus_clipped_bigger_range
        )

        clipped_bigger_minus_smaller_range = [
            clipped_bigger_bounds[0] - smaller_bounds[1],
            clipped_bigger_bounds[1] - smaller_bounds[0],
        ]
        clipped_bigger_minus_smaller_dtype = Integer.that_can_represent(
            clipped_bigger_minus_smaller_range
        )

        do_clipped_bigger_minus_smaller = (
            clipped_bigger_minus_smaller_dtype.bit_width
            <= smaller_minus_clipped_bigger_dtype.bit_width
        )
        if do_clipped_bigger_minus_smaller:
            subtraction_order = (bigger, smaller)
            subtraction_dtype = clipped_bigger_minus_smaller_dtype
            subtraction_bit_width = clipped_bigger_minus_smaller_dtype.bit_width
        else:
            subtraction_order = (smaller, bigger)
            subtraction_dtype = smaller_minus_clipped_bigger_dtype
            subtraction_bit_width = smaller_minus_clipped_bigger_dtype.bit_width

        if not (smaller.original_bit_width < subtraction_bit_width <= bigger.original_bit_width):
            return None

        intermediate_bit_width = max(smaller.bit_width, subtraction_bit_width)
        intermediate_scalar_type = self.esint(intermediate_bit_width)

        if smaller.bit_width != intermediate_bit_width:
            smaller = self.cast(self.tensor(intermediate_scalar_type, smaller.shape), smaller)

        low_bound = smaller_bounds[0] - 1
        high_bound = smaller_bounds[1] + 1

        clipper_lut = []
        if bigger.is_unsigned:
            clipper_lut += [
                np.clip(i, low_bound, high_bound) for i in range(0, 2**bigger.bit_width)
            ]
        else:
            clipper_lut += [
                np.clip(i, low_bound, high_bound) for i in range(0, 2 ** (bigger.bit_width - 1))
            ]
            clipper_lut += [
                np.clip(i, low_bound, high_bound) for i in range(-(2 ** (bigger.bit_width - 1)), 0)
            ]
        clipped_bigger = self.tlu(
            self.tensor(intermediate_scalar_type, bigger.shape),
            bigger,
            clipper_lut,
        )

        comparison_order = (x, y)
        if subtraction_order != comparison_order:  # pragma: no cover
            new_accept = set()
            if Comparison.EQUAL in accept:
                new_accept.add(Comparison.EQUAL)
            if Comparison.LESS in accept:
                new_accept.add(Comparison.GREATER)
            if Comparison.GREATER in accept:
                new_accept.add(Comparison.LESS)
            accept = new_accept

        intermediate_type = self.tensor(intermediate_scalar_type, resulting_type.shape)
        if do_clipped_bigger_minus_smaller:
            subtraction = self.sub(intermediate_type, clipped_bigger, smaller)
        else:
            subtraction = self.sub(intermediate_type, smaller, clipped_bigger)

        if not subtraction_dtype.is_signed:
            subtraction = self.to_unsigned(subtraction)

        return self.compare_with_subtraction(resulting_type, subtraction, accept)

    def comparison_with_chunks(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        accept: Set[Comparison],
    ) -> Conversion:
        """
        Compare encrypted values using chunks.

        Idea:
            split x and y into small chunks
            compare the chunks using table lookups
            reduce chunk comparisons to a final result
        """

        x_offset = 0
        y_offset = 0

        x_was_signed = x.is_signed
        y_was_signed = y.is_signed

        if x.is_signed or y.is_signed:
            if x.is_signed:
                signed_offset = 2 ** (x.original_bit_width - 1)
                sanitizer = self.constant(self.i(x.bit_width + 1), signed_offset)
                x = self.to_unsigned(self.add(x.type, x, sanitizer))
                y_offset += signed_offset
            if y.is_signed:
                signed_offset = 2 ** (y.original_bit_width - 1)
                sanitizer = self.constant(self.i(y.bit_width + 1), signed_offset)
                y = self.to_unsigned(self.add(y.type, y, sanitizer))
                x_offset += signed_offset

        min_offset = min(x_offset, y_offset)

        x_offset -= min_offset
        y_offset -= min_offset

        chunk_ranges = self.best_chunk_ranges(x, x_offset, y, y_offset)
        assert 1 <= len(chunk_ranges) <= 3

        if accept in ({Comparison.EQUAL}, {Comparison.LESS, Comparison.GREATER}):
            return self.comparison_with_chunks_equals(
                resulting_type,
                x,
                y,
                accept,
                x_offset,
                y_offset,
                x_was_signed,
                y_was_signed,
                chunk_ranges,
            )

        carry_bit_width = 2
        intermediate_scalar_type = self.eint(2 * carry_bit_width)

        def compare(a, b):
            if a < b:
                return Comparison.LESS

            if a > b:
                return Comparison.GREATER

            return Comparison.EQUAL

        carries = self.convert_to_chunks_and_map(
            intermediate_scalar_type,
            resulting_type.shape,
            chunk_ranges,
            x,
            x_offset,
            y,
            y_offset,
            lambda i, a, b: compare(a, b) << (min(i, 1) * 2),
        )

        carry_type = self.tensor(intermediate_scalar_type, shape=resulting_type.shape)

        all_comparisons = [
            Comparison.EQUAL,
            Comparison.LESS,
            Comparison.GREATER,
            Comparison.MASK,
        ]
        pick_first_not_equal_lut = [
            (
                int(current_comparison)
                if previous_comparison == Comparison.EQUAL
                else int(previous_comparison)
            )
            for current_comparison in all_comparisons
            for previous_comparison in all_comparisons
        ]

        carry = carries[0]
        for next_carry in carries[1:-1]:
            combined_carries = self.add(carry_type, next_carry, carry)
            carry = self.tlu(carry_type, combined_carries, pick_first_not_equal_lut)

        if x_was_signed != y_was_signed:
            if len(carries) > 1:
                combined_carries = self.add(carry_type, carries[-1], carry)
                carry = self.tlu(carry_type, combined_carries, pick_first_not_equal_lut)

            signed_input = x if x_was_signed else y
            unsigned_input = x if not x_was_signed else y

            signed_offset = 2 ** (signed_input.original_bit_width - 1)
            is_unsigned_greater_lut = [
                int(value >= signed_offset) << carry_bit_width
                for value in range(2**unsigned_input.bit_width)
            ]

            is_unsigned_greater = None
            if not all(value == 0 for value in is_unsigned_greater_lut):
                is_unsigned_greater = self.tlu(
                    self.tensor(intermediate_scalar_type, shape=unsigned_input.shape),
                    unsigned_input,
                    is_unsigned_greater_lut,
                )

            packed_carry_and_is_unsigned_greater = (
                carry
                if is_unsigned_greater is None
                else self.add(
                    carry_type,
                    is_unsigned_greater,
                    carry,
                )
            )

            # this function is actually converting either
            # - lhs < rhs
            # - lhs <= rhs

            # in the implementation, we call
            # - x = lhs
            # - y = rhs

            # so if y is unsigned and greater than half
            # - y is definitely bigger than x
            # - is_unsigned_greater == 1
            # - result ==  (lhs < rhs) == (x < y) == 1

            # so if x is unsigned and greater than half
            # - x is definitely bigger than y
            # - is_unsigned_greater == 1
            # - result ==  (lhs < rhs) == (x < y) == 0

            if not y_was_signed:
                result_lut = [
                    (1 if (i >> carry_bit_width) else int((i & Comparison.MASK) in accept))
                    for i in range(2**3)
                ]
            else:
                result_lut = [
                    (0 if (i >> carry_bit_width) else int((i & Comparison.MASK) in accept))
                    for i in range(2**3)
                ]

            result = self.tlu(
                resulting_type,
                packed_carry_and_is_unsigned_greater,
                result_lut,
            )
        else:
            result_lut = [int(comparison in accept) for comparison in all_comparisons]
            if len(carries) > 1:
                carry = self.add(carry_type, carries[-1], carry)
                result_lut = [result_lut[carry] for carry in pick_first_not_equal_lut]

            result = self.tlu(resulting_type, carry, result_lut)

        return result

    def comparison_with_chunks_equals(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        accept: Set[Comparison],
        x_offset: int,
        y_offset: int,
        x_was_signed: bool,
        y_was_signed: bool,
        chunk_ranges: List[Tuple[int, int]],
    ) -> Conversion:
        """
        Check equality of encrypted values using chunks.
        """

        number_of_chunks = len(chunk_ranges)
        intermediate_scalar_type = self.eint(Integer.that_can_represent(number_of_chunks).bit_width)

        is_unsigned_greater = None
        if x_was_signed != y_was_signed:
            signed_input = x if x_was_signed else y
            unsigned_input = x if not x_was_signed else y

            signed_offset = 2 ** (signed_input.original_bit_width - 1)
            is_unsigned_greater_lut = [
                int(value >= signed_offset) for value in range(2**unsigned_input.bit_width)
            ]

            if not all(value == 0 for value in is_unsigned_greater_lut):
                number_of_chunks += 1
                intermediate_scalar_type = self.eint(
                    Integer.that_can_represent(number_of_chunks).bit_width,
                )

                is_unsigned_greater = self.tlu(
                    self.tensor(intermediate_scalar_type, unsigned_input.shape),
                    unsigned_input,
                    is_unsigned_greater_lut,
                )

        carries = self.convert_to_chunks_and_map(
            intermediate_scalar_type,
            resulting_type.shape,
            chunk_ranges,
            x,
            x_offset,
            y,
            y_offset,
            lambda _, a, b: int(a != b),
        )

        if is_unsigned_greater:
            carries.append(is_unsigned_greater)

        return self.tlu(
            resulting_type,
            self.tree_add(self.tensor(intermediate_scalar_type, resulting_type.shape), carries),
            [
                int(i == 0 if Comparison.EQUAL in accept else i != 0)
                for i in range(2**intermediate_scalar_type.bit_width)
            ],
        )

    def best_chunk_ranges(
        self,
        x: Conversion,
        x_offset: int,
        y: Conversion,
        y_offset: int,
    ) -> List[Tuple[int, int]]:
        """
        Calculate best chunk ranges for given operands.

        Args:
            x (Conversion)
                lhs of the operation

            x_offset (int)
                lhs offset

            y (Conversion)
                rhs of the operation

            y_offset (int)
                rhs offset

        Returns:
            List[Tuple[int, int]]:
                best chunk ranges for the arguments
        """

        lhs_bit_width = x.original_bit_width
        rhs_bit_width = y.original_bit_width

        if x_offset != 0:
            lhs_bit_width = max(lhs_bit_width, int(np.log2(x_offset))) + 1
        if y_offset != 0:
            rhs_bit_width = max(rhs_bit_width, int(np.log2(y_offset))) + 1

        max_original_input_bit_width = max(lhs_bit_width, rhs_bit_width)
        min_original_input_bit_width = min(lhs_bit_width, rhs_bit_width)

        max_input_bit_width = max(x.bit_width, y.bit_width)
        chunk_size = max(1, int(np.floor(max_input_bit_width / 2)))

        if chunk_size >= min_original_input_bit_width:
            chunk_ranges = [
                (0, min_original_input_bit_width),
            ]
        else:
            optimal_chunk_size = min(chunk_size, int(np.ceil(min_original_input_bit_width / 2)))
            chunk_ranges = [
                (0, optimal_chunk_size),
                (optimal_chunk_size, min(2 * optimal_chunk_size, min_original_input_bit_width)),
            ]

        last_chunk_range = chunk_ranges[-1]
        _, last_chunk_end = last_chunk_range

        if last_chunk_end != max_original_input_bit_width:
            chunk_ranges.append((last_chunk_end, max_original_input_bit_width))

        return list(reversed(chunk_ranges))

    def convert_to_chunks_and_map(
        self,
        resulting_scalar_type: ConversionType,
        resulting_shape: Tuple[int, ...],
        chunk_ranges: List[Tuple[int, int]],
        x: Conversion,
        x_offset: int,
        y: Conversion,
        y_offset: int,
        mapper: Callable,
    ) -> List[Conversion]:
        """
        Extract the chunks of two values, pack them in a single integer and map the integer.

        Args:
            resulting_scalar_type (ConversionType):
                scalar type of the results

            resulting_shape (ConversionType):
                shape of the output of the operation

            chunk_ranges (List[Tuple[int, int]]):
                chunks ranges for the operation

            x (Conversion):
                first operand

            y (Conversion):
                second operand

            mapper (Callable):
                mapping function

            x_offset (int, default = 0):
                optional offset for x during chunk extraction

            y_offset (int, default = 0):
                optional offset for x during chunk extraction

        Returns:
            List[Conversion]:
                result of mapping chunks of x and y
        """

        # this function computes the following in FHE
        # -------------------------------------------
        # result = []
        # for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
        #     x_chunk = (x + x_offset).bits[chunk_start:chunk_end]
        #     y_chunk = (y + y_offset).bits[chunk_start:chunk_end]
        #     result.append(mapper(chunk_index, x_chunk, y_chunk))
        # return result
        # -------------------------------------------
        # - x_chunk :: { bit_width: (2 * chunk_size), shape: x.shape }
        # - y_chunk :: { bit_width: (2 * chunk_size), shape: y.shape }
        # - results :: { bit_width: resulting_scalar_type.bit_width, shape: resulting_shape }

        assert resulting_scalar_type.is_scalar

        result = []
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            chunk_size = chunk_end - chunk_start

            shift_to_clear_lsbs = chunk_start
            mask_to_clear_msbs = (2**chunk_size) - 1

            x_chunk_lut = [
                (((x + x_offset) >> shift_to_clear_lsbs) & mask_to_clear_msbs) << chunk_size
                for x in range(2**x.bit_width)
            ]
            y_chunk_lut = [
                (((y + y_offset) >> shift_to_clear_lsbs) & mask_to_clear_msbs)
                for y in range(2**y.bit_width)
            ]

            x_chunk_is_constant = all(x == x_chunk_lut[0] for x in x_chunk_lut)
            y_chunk_is_constant = all(y == y_chunk_lut[0] for y in y_chunk_lut)

            if x_chunk_is_constant and y_chunk_is_constant:  # pragma: no cover
                result.append(
                    self.constant(
                        self.i(resulting_scalar_type.bit_width + 1),
                        mapper(chunk_index, x_chunk_lut[0] >> chunk_size, y_chunk_lut[0]),
                    )
                )
                continue

            elif x_chunk_is_constant:
                result.append(
                    self.broadcast_to(
                        self.tlu(
                            self.tensor(resulting_scalar_type, shape=y.shape),
                            y,
                            [
                                mapper(chunk_index, x_chunk_lut[0] >> chunk_size, y)
                                for y in y_chunk_lut
                            ],
                        ),
                        resulting_shape,
                    )
                )
                continue

            elif y_chunk_is_constant:
                result.append(
                    self.broadcast_to(
                        self.tlu(
                            self.tensor(resulting_scalar_type, shape=x.shape),
                            x,
                            [mapper(chunk_index, x, y_chunk_lut[0]) for x in x_chunk_lut],
                        ),
                        resulting_shape,
                    )
                )
                continue

            packed_chunks_type = self.eint(chunk_size * 2)

            x_chunk = self.tlu(self.tensor(packed_chunks_type, shape=x.shape), x, x_chunk_lut)
            y_chunk = self.tlu(self.tensor(packed_chunks_type, shape=y.shape), y, y_chunk_lut)

            packed_chunks = self.add(
                self.tensor(packed_chunks_type, shape=resulting_shape),
                x_chunk,
                y_chunk,
            )
            mapped_chunks = self.tlu(
                self.tensor(resulting_scalar_type, shape=resulting_shape),
                packed_chunks,
                [
                    mapper(chunk_index, x, y)
                    for x in range(mask_to_clear_msbs + 1)
                    for y in range(mask_to_clear_msbs + 1)
                ],
            )

            result.append(mapped_chunks)

        return result

    def pack_multivariate_inputs(self, xs: List[Conversion]) -> Conversion:
        """
        Packs inputs of multivariate table lookups.

        Args:
            xs (List[Conversion]):
                operands

        Returns:
            Conversion:
                packed operands
        """

        assert all(x.is_encrypted for x in xs)

        required_bit_width = sum(x.original_bit_width for x in xs)
        maximum_bit_width = max(x.bit_width for x in xs)
        bit_width_to_pack = max(required_bit_width, maximum_bit_width)

        shifted_xs = []
        next_shift_amount = 0

        for x in xs:
            needs_cast = x.bit_width != bit_width_to_pack
            needs_sanitize = x.is_signed
            needs_shift = next_shift_amount > 0

            shift_amount = next_shift_amount
            next_shift_amount += x.original_bit_width

            if not (needs_cast or needs_sanitize or needs_shift):
                shifted_xs.append(x)
                continue

            if not needs_cast:
                if needs_sanitize:
                    sanitizer = self.constant(
                        self.i(bit_width_to_pack + 1),
                        2 ** (x.original_bit_width - 1),
                    )
                    x = self.to_unsigned(self.add(x.type, x, sanitizer))

                if needs_shift:
                    shifter = self.constant(
                        self.i(bit_width_to_pack + 1),
                        2**shift_amount,
                    )
                    x = self.mul(x.type, x, shifter)

                shifted_xs.append(x)
                continue

            dtype = Integer(bit_width=x.bit_width, is_signed=x.is_signed)

            shift_table = list(range(dtype.max() + 1))
            if x.is_signed:
                shift_table += list(range(dtype.min(), 0))

            if needs_sanitize:
                shift_table = [value + 2 ** (x.original_bit_width - 1) for value in shift_table]

            if needs_shift:
                shift_table = [value * (2**shift_amount) for value in shift_table]

            shifted_xs.append(
                self.tlu(
                    self.tensor(self.eint(bit_width_to_pack), shape=x.shape),
                    x,
                    shift_table,
                )
            )

        assert next_shift_amount <= MAXIMUM_TLU_BIT_WIDTH

        return self.tree_add(
            self.tensor(
                self.eint(bit_width_to_pack),
                shape=(sum(np.zeros(x.shape) for x in xs)).shape,  # type: ignore
            ),
            shifted_xs,
        )

    def minimum_maximum_with_trick(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        x_minus_y_dtype: Integer,
        intermediate_table: List[int],
    ) -> Conversion:
        """
        Calculate minimum or maximum between two encrypted values using minimum or maximum trick.

        Idea:
            min(x, y) <==> min(x - y, 0) + y
            max(x, y) <==> max(x - y, 0) + y

        Additional Args:
            x_minus_y_dtype (Integer):
                minimal dtype that can be used to store x - y without overflows
        """

        maximum_input_bit_width = max(x.bit_width, y.bit_width)

        intermediate_bit_width = max(x_minus_y_dtype.bit_width, maximum_input_bit_width)
        intermediate_scalar_type = self.esint(intermediate_bit_width)

        final_addition_rhs = y

        if x.bit_width != intermediate_bit_width:
            x = self.cast(self.tensor(intermediate_scalar_type, x.shape), x)
        if y.bit_width != intermediate_bit_width:
            y = self.cast(self.tensor(intermediate_scalar_type, y.shape), y)

        assert x.bit_width == y.bit_width

        x = self.to_signed(x)
        y = self.to_signed(y)

        intermediate_type = self.tensor(intermediate_scalar_type, resulting_type.shape)
        x_minus_y = self.sub(intermediate_type, x, y)

        if not x_minus_y_dtype.is_signed:
            x_minus_y = self.to_unsigned(x_minus_y)

        final_addition_lhs = self.tlu(
            resulting_type,
            x_minus_y,
            intermediate_table,
        )

        if final_addition_rhs.bit_width != resulting_type.bit_width:
            final_addition_rhs_type = (self.eint if final_addition_rhs.is_unsigned else self.esint)(
                resulting_type.bit_width
            )
            final_addition_rhs = self.cast(
                self.tensor(final_addition_rhs_type, shape=final_addition_rhs.shape),
                final_addition_rhs,
            )

        return self.add(resulting_type, final_addition_lhs, final_addition_rhs)

    def minimum_maximum_with_chunks(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
        operation: str,
    ) -> Conversion:
        """
        Calculate minimum or maximum between two encrypted values using chunks.
        """

        # make sure the operation is supported
        assert operation in ["min", "max"]

        # compare x and y
        comparison_type = self.tensor(self.eint(1), resulting_type.shape)
        if operation == "min":
            select_x = self.less(comparison_type, x, y)
        else:
            select_x = self.greater(comparison_type, x, y)

        # remember original bit widths of x and y
        x_original_bit_width = x.original_bit_width
        y_original_bit_width = y.original_bit_width

        # remember original signednesses of x and y
        x_is_signed = x.is_signed
        y_is_signed = y.is_signed

        # sanitize x
        if x_is_signed:
            signed_offset = 2 ** (x_original_bit_width - 1)
            sanitizer = self.constant(self.i(x.bit_width + 1), signed_offset)
            x = self.to_unsigned(self.add(x.type, x, sanitizer))

        # sanitize y
        if y_is_signed:
            signed_offset = 2 ** (y_original_bit_width - 1)
            sanitizer = self.constant(self.i(y.bit_width + 1), signed_offset)
            y = self.to_unsigned(self.add(y.type, y, sanitizer))

        # multiply sanitized `x` with `select_x`
        if x.bit_width > x_original_bit_width:
            shifted_x = self.mul(x.type, x, self.constant(self.i(x.bit_width + 1), 2))
            packing = self.add(
                self.tensor(self.eint(x.bit_width), select_x.shape),
                shifted_x,
                self.cast(self.tensor(self.eint(x.bit_width), select_x.shape), select_x),
            )

            table = []
            for i in range(2**x.bit_width):
                if i % 2 == 0:
                    table.append(0)
                    continue

                i = i >> 1
                if x_is_signed:
                    i -= 2 ** (x_original_bit_width - 1)

                table.append(i)

            x_contribution = self.tlu(
                resulting_type,
                packing,
                table,
            )
        else:
            mid_point = x_original_bit_width // 2
            chunk_ranges = [
                (0, mid_point),
                (mid_point, x_original_bit_width),
            ]

            x_contribution_chunks = []
            for chunk_start, chunk_end in chunk_ranges:
                chunk_size = chunk_end - chunk_start
                if chunk_size == 0:
                    continue

                shift_to_clear_lsbs = chunk_start
                mask_to_clear_msbs = (2**chunk_size) - 1

                x_chunk_lut = [
                    ((x >> shift_to_clear_lsbs) & mask_to_clear_msbs) << 1
                    for x in range(2**x.bit_width)
                ]

                packing_element_type = self.eint(chunk_size + 1)
                x_chunk = self.tlu(self.tensor(packing_element_type, shape=x.shape), x, x_chunk_lut)

                packing = self.add(
                    self.tensor(packing_element_type, shape=select_x.shape),
                    x_chunk,
                    self.cast(self.tensor(packing_element_type, select_x.shape), select_x),
                )

                x_contribution_chunk = self.tlu(
                    resulting_type,
                    packing,
                    [
                        (packing >> 1) << chunk_start if packing % 2 == 1 else 0
                        for packing in range(2**packing_element_type.bit_width)
                    ],
                )
                x_contribution_chunks.append(x_contribution_chunk)

            x_contribution = (
                x_contribution_chunks[0]
                if len(x_contribution_chunks) == 1
                else self.add(
                    resulting_type,
                    x_contribution_chunks[0],
                    x_contribution_chunks[1],
                )
            )

            # - x_contribution is either
            #   - 0
            #   - x (if x was unsigned)
            #   - sanitized x (if x was signed)

            # remove sanitization of x if it's signed and selected
            if x_is_signed:
                signed_offset = 2 ** (x_original_bit_width - 1)
                sanitizer_or_zero = self.mul(
                    resulting_type,
                    self.cast(resulting_type, select_x),
                    self.constant(self.i(resulting_type.bit_width + 1), signed_offset),
                )
                x_contribution = self.sub(
                    resulting_type,
                    self.to_signed(x_contribution),
                    sanitizer_or_zero,
                )

        # multiply sanitized `y` with `1 - select_y`
        if y.bit_width > y_original_bit_width:
            shifted_y = self.mul(y.type, y, self.constant(self.i(y.bit_width + 1), 2))
            packing = self.add(
                self.tensor(self.eint(y.bit_width), select_x.shape),
                shifted_y,
                self.cast(self.tensor(self.eint(y.bit_width), select_x.shape), select_x),
            )

            table = []
            for i in range(2**y.bit_width):
                if i % 2 == 1:
                    table.append(0)
                    continue

                i = i >> 1
                if y_is_signed:
                    i -= 2 ** (y_original_bit_width - 1)

                table.append(i)

            y_contribution = self.tlu(
                resulting_type,
                packing,
                table,
            )
        else:
            mid_point = y_original_bit_width // 2
            chunk_ranges = [
                (0, mid_point),
                (mid_point, y_original_bit_width),
            ]

            y_contribution_chunks = []
            for chunk_start, chunk_end in chunk_ranges:
                chunk_size = chunk_end - chunk_start
                if chunk_size == 0:
                    continue

                shift_to_clear_lsbs = chunk_start
                mask_to_clear_msbs = (2**chunk_size) - 1

                y_chunk_lut = [
                    ((y >> shift_to_clear_lsbs) & mask_to_clear_msbs) << 1
                    for y in range(2**y.bit_width)
                ]

                packing_element_type = self.eint(chunk_size + 1)
                y_chunk = self.tlu(self.tensor(packing_element_type, shape=y.shape), y, y_chunk_lut)

                packing = self.add(
                    self.tensor(packing_element_type, shape=select_x.shape),
                    y_chunk,
                    self.cast(self.tensor(packing_element_type, select_x.shape), select_x),
                )

                y_contribution_chunk = self.tlu(
                    resulting_type,
                    packing,
                    [
                        (packing >> 1) << chunk_start if packing % 2 == 0 else 0
                        for packing in range(2**packing_element_type.bit_width)
                    ],
                )
                y_contribution_chunks.append(y_contribution_chunk)

            y_contribution = (
                y_contribution_chunks[0]
                if len(y_contribution_chunks) == 1
                else self.add(
                    resulting_type,
                    y_contribution_chunks[0],
                    y_contribution_chunks[1],
                )
            )

            # - y_contribution is either
            #   - 0
            #   - y (if y was unsigned)
            #   - sanitized y (if y was signed)

            # remove sanitization of y if it's signed and selected
            if y_is_signed:
                signed_offset = 2 ** (y_original_bit_width - 1)
                sanitizer_or_zero = self.mul(
                    resulting_type,
                    self.cast(
                        resulting_type,
                        self.sub(
                            self.tensor(self.eint(1), select_x.shape),
                            self.constant(self.i(2), 1),
                            select_x,
                        ),
                    ),
                    self.constant(self.i(resulting_type.bit_width + 1), signed_offset),
                )
                y_contribution = self.sub(
                    resulting_type,
                    self.to_signed(y_contribution),
                    sanitizer_or_zero,
                )

        # add contributions of x and y to compute the result
        return self.add(resulting_type, x_contribution, y_contribution)

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

        x = self.to_signedness(x, of=resulting_type)
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

        maximum_input_bit_width = max(x.bit_width, y.bit_width)
        if maximum_input_bit_width > MAXIMUM_TLU_BIT_WIDTH:
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
                    if operand.bit_width != operand.original_bit_width:  # pragma: no cover
                        highlights[operand.origin].append(  # type: ignore
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )

            self.error(highlights)

        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        strategy_preference = self.converting.properties["strategy"]
        if x.original_bit_width + y.original_bit_width <= maximum_input_bit_width or (
            strategy_preference == BitwiseStrategy.THREE_TLU_CASTED
        ):
            intermediate_bit_width = max(
                x.original_bit_width + y.original_bit_width,
                maximum_input_bit_width,
            )
            intermediate_scalar_type = self.eint(intermediate_bit_width)

            if x.bit_width != intermediate_bit_width:
                x = self.cast(self.tensor(intermediate_scalar_type, x.shape), x)
            if y.bit_width != intermediate_bit_width:
                y = self.cast(self.tensor(intermediate_scalar_type, y.shape), y)

            shifter_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=False, bit_width=(x.bit_width + 1)),
                    shape=(),
                    is_encrypted=False,
                )
            )
            shifter = self.constant(shifter_type, 2**y.original_bit_width)

            shifted_x = self.mul(x.type, x, shifter)
            packed_x_and_y = self.add(
                self.tensor(intermediate_scalar_type, resulting_type.shape),
                shifted_x,
                y,
            )

            return self.tlu(
                resulting_type,
                packed_x_and_y,
                [
                    operation(x_value, y_value)
                    for x_value in range(2**x.original_bit_width)
                    for y_value in range(2**y.original_bit_width)
                ],
            )

        chunks = []
        for start, end in self.best_chunk_ranges(x, 0, y, 0):
            size = end - start
            mask = (2**size) - 1

            intermediate_bit_width = size * 2
            intermediate_scalar_type = self.eint(intermediate_bit_width)

            x_chunk_lut = [((x >> start) & mask) << size for x in range(2**x.original_bit_width)]
            y_chunk_lut = [(y >> start) & mask for y in range(2**y.original_bit_width)]

            x_chunk_is_constant = all(x == x_chunk_lut[0] for x in x_chunk_lut)
            y_chunk_is_constant = all(y == y_chunk_lut[0] for y in y_chunk_lut)

            if x_chunk_is_constant and y_chunk_is_constant:  # pragma: no cover
                chunks.append(
                    self.constant(
                        self.i(resulting_type.bit_width + 1),
                        operation(x_chunk_lut[0] >> size, y_chunk_lut[0]) << start,
                    )
                )
                continue

            elif x_chunk_is_constant:
                chunks.append(
                    self.tlu(
                        self.tensor(self.eint(resulting_type.bit_width), shape=y.shape),
                        y,
                        [operation(x_chunk_lut[0] >> size, y) << start for y in y_chunk_lut],
                    ),
                )
                continue

            elif y_chunk_is_constant:
                chunks.append(
                    self.tlu(
                        self.tensor(self.eint(resulting_type.bit_width), shape=x.shape),
                        x,
                        [operation(x >> size, y_chunk_lut[0]) << start for x in x_chunk_lut],
                    ),
                )
                continue

            x_chunk = self.tlu(self.tensor(intermediate_scalar_type, x.shape), x, x_chunk_lut)
            y_chunk = self.tlu(self.tensor(intermediate_scalar_type, y.shape), y, y_chunk_lut)

            packed_x_and_y_chunks = self.add(
                self.tensor(intermediate_scalar_type, resulting_type.shape),
                x_chunk,
                y_chunk,
            )
            result_chunk = self.tlu(
                resulting_type,
                packed_x_and_y_chunks,
                [operation(x, y) << start for x in range(2**size) for y in range(2**size)],
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

    def cast(self, resulting_type: ConversionType, x: Conversion) -> Conversion:
        assert x.original_bit_width <= resulting_type.bit_width

        if x.bit_width != resulting_type.bit_width:
            dtype = Integer(bit_width=x.bit_width, is_signed=x.is_signed)
            original_bit_width = x.original_bit_width

            identity_lut = list(range(dtype.max() + 1))
            if x.is_signed:
                identity_lut += list(range(dtype.min(), 0))

            x = self.tlu(resulting_type, x, identity_lut)
            x.set_original_bit_width(original_bit_width)

        return x

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
                arith.ConstantOp(resulting_type, attribute, loc=self.location()),
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

        if x.is_encrypted and y.is_encrypted:
            assert self.is_bit_width_compatible(x, y)
        else:
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

    def dynamic_tlu(
        self,
        resulting_type: ConversionType,
        on: Conversion,
        table: Conversion,
    ) -> Conversion:
        assert table.is_clear and on.is_encrypted

        if table.shape != (2**on.bit_width,):
            highlights: Dict[Node, Union[str, List[str]]] = {
                table.origin: [
                    f"table has the shape {table.shape}",
                ],
                on.origin: [
                    f"table lookup input is {on.bit_width}-bits",
                ],
                self.converting: [
                    "so table cannot be looked up with this input",
                    f"table shape should have been {(2**on.bit_width,)}",
                ],
            }
            if on.bit_width != on.original_bit_width:  # pragma: no cover
                highlights[on.origin].append(  # type: ignore
                    "("
                    f"note that it's assigned {on.bit_width}-bits "
                    f"during compilation because of its relation with other operations"
                    ")"
                )
            self.error(highlights)

        dialect = fhe if on.is_scalar else fhelinalg
        operation = dialect.ApplyLookupTableEintOp

        return self.operation(operation, resulting_type, on.result, table.result)

    def encrypt(self, resulting_type: ConversionType, x: Conversion) -> Conversion:
        assert self.is_bit_width_compatible(resulting_type, x)
        assert resulting_type.is_encrypted and x.is_clear
        assert x.shape == resulting_type.shape

        return self.add(resulting_type, x, self.zeros(resulting_type))

    def equal(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.EQUAL})

    def extract_bits(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        bits: Union[int, np.integer, slice],
    ) -> Conversion:
        if x.is_clear:
            highlights: Dict[Node, Union[str, List[str]]] = {
                x.origin: "operand is clear",
                self.converting: "but clear bit extraction is not supported",
            }
            self.error(highlights)

        if isinstance(bits, (int, np.integer)):
            bits = slice(bits, bits + 1, 1)

        assert isinstance(bits, slice)

        step = bits.step or 1

        assert step != 0
        if step < 0:
            assert bits.start is not None

        start = bits.start or MIN_EXTRACTABLE_BIT
        stop = bits.stop or (MAX_EXTRACTABLE_BIT if step > 0 else (MIN_EXTRACTABLE_BIT - 1))

        bits_and_their_positions = []
        for position, bit in enumerate(range(start, stop, step)):
            bits_and_their_positions.append((bit, position))

        bits_and_their_positions = sorted(
            bits_and_their_positions,
            key=lambda bit_and_its_position: bit_and_its_position[0],
        )

        current_bit = 0
        max_bit = x.original_bit_width

        lsb: Optional[Conversion] = None
        result: Optional[Conversion] = None

        for index, (bit, position) in enumerate(bits_and_their_positions):
            if bit >= max_bit and x.is_unsigned:
                break

            last = index == len(bits_and_their_positions) - 1
            while bit != (current_bit - 1):
                if bit == (max_bit - 1) and x.bit_width == 1 and x.is_unsigned:
                    lsb = x
                elif last and bit == current_bit:
                    lsb = self.lsb(resulting_type, x)
                else:
                    lsb = self.lsb(x.type, x)

                current_bit += 1

                if current_bit >= max_bit:
                    break

                if not last or bit != (current_bit - 1):
                    cleared = self.sub(x.type, x, lsb)
                    x = self.reinterpret(cleared, bit_width=(x.bit_width - 1))

            assert lsb is not None
            lsb = self.to_signedness(lsb, of=resulting_type)

            if lsb.bit_width > resulting_type.bit_width:
                difference = (lsb.bit_width - resulting_type.bit_width) + position
                shifter = self.constant(self.i(lsb.bit_width + 1), 2**difference)
                shifted = self.mul(lsb.type, lsb, shifter)
                lsb = self.reinterpret(shifted, bit_width=resulting_type.bit_width)

            elif lsb.bit_width < resulting_type.bit_width:
                shift = 2 ** (lsb.bit_width - 1)
                if shift != 1:
                    shifter = self.constant(self.i(lsb.bit_width + 1), shift)
                    shifted = self.mul(lsb.type, lsb, shifter)
                    lsb = self.reinterpret(shifted, bit_width=1)
                lsb = self.tlu(resulting_type, lsb, [0 << position, 1 << position])

            elif position != 0:
                shifter = self.constant(self.i(lsb.bit_width + 1), 2**position)
                lsb = self.mul(lsb.type, lsb, shifter)

            assert lsb is not None
            result = lsb if result is None else self.add(resulting_type, result, lsb)

        return result if result is not None else self.zeros(resulting_type)

    def flatten(self, x: Conversion) -> Conversion:
        return self.reshape(x, shape=(int(np.prod(x.shape)),))

    def greater(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.GREATER})

    def greater_equal(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.GREATER, Comparison.EQUAL})

    def index_static(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        index: Sequence[Union[int, np.integer, slice, np.ndarray, list]],
    ) -> Conversion:
        assert self.is_bit_width_compatible(resulting_type, x)
        assert resulting_type.is_encrypted == x.is_encrypted

        x = self.to_signedness(x, of=resulting_type)

        if any(isinstance(indexing_element, (list, np.ndarray)) for indexing_element in index):
            return self.index_static_fancy(resulting_type, x, index)

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
                assert isinstance(indexing_element, (int, np.integer))
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

    def index_static_fancy(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        index: Sequence[Union[int, np.integer, slice, np.ndarray, list]],
    ) -> Conversion:
        resulting_element_type = (self.eint if resulting_type.is_unsigned else self.esint)(
            resulting_type.bit_width
        )

        result = self.zeros(resulting_type)
        for destination_position in np.ndindex(resulting_type.shape):
            source_position = []
            for indexing_element in index:
                if isinstance(indexing_element, (int, np.integer)):
                    source_position.append(indexing_element)

                elif isinstance(indexing_element, (list, np.ndarray)):
                    position = indexing_element[destination_position[0]]
                    for n in range(1, len(destination_position)):
                        position = position[destination_position[n]]
                    source_position.append(position)

                else:  # pragma: no cover
                    message = f"invalid indexing element of type {type(indexing_element)}"
                    raise AssertionError(message)

            element = self.index_static(resulting_element_type, x, tuple(source_position))
            result = self.assign_static(resulting_type, result, element, destination_position)

        return result

    def less(self, resulting_type: ConversionType, x: Conversion, y: Conversion) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.LESS})

    def less_equal(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.LESS, Comparison.EQUAL})

    def lsb(self, resulting_type: ConversionType, x: Conversion) -> Conversion:
        assert resulting_type.shape == x.shape
        assert resulting_type.is_encrypted and x.is_encrypted

        operation = fhe.LsbEintOp if x.is_scalar else fhelinalg.LsbEintOp
        return self.operation(operation, resulting_type, x.result)

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

        if x.is_encrypted and y.is_encrypted:
            assert self.is_bit_width_compatible(x, y)
        else:
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

    def maximum(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        maximum_input_bit_width = max(x.bit_width, y.bit_width)
        if maximum_input_bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit maximum operation is supported"
                ],
            }
            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to a maximum operation"
                    ]
                    if operand.bit_width != operand.original_bit_width:  # pragma: no cover
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )
            self.error(highlights)

        if y.bit_width != resulting_type.bit_width:
            x_can_be_added_directly = resulting_type.bit_width == x.bit_width
            x_has_smaller_bit_width = x.bit_width < y.bit_width
            if x_can_be_added_directly or x_has_smaller_bit_width:
                x, y = y, x

        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        x_dtype = Integer(is_signed=x.is_signed, bit_width=x.original_bit_width)
        y_dtype = Integer(is_signed=y.is_signed, bit_width=y.original_bit_width)

        x_minus_y_min = x_dtype.min() - y_dtype.max()
        x_minus_y_max = x_dtype.max() - y_dtype.min()

        x_minus_y_range = [x_minus_y_min, x_minus_y_max]
        x_minus_y_dtype = Integer.that_can_represent(x_minus_y_range)

        maximum_trick_table = []
        if x_minus_y_dtype.is_signed:
            maximum_trick_table += [i for i in range(2 ** (x_minus_y_dtype.bit_width - 1))]
            maximum_trick_table += [0] * 2 ** (x_minus_y_dtype.bit_width - 1)
        else:
            maximum_trick_table += [i for i in range(2**x_minus_y_dtype.bit_width)]

        if x_minus_y_dtype.bit_width <= maximum_input_bit_width:
            return self.minimum_maximum_with_trick(
                resulting_type,
                x,
                y,
                x_minus_y_dtype,
                maximum_trick_table,
            )

        strategy_preference = self.converting.properties["strategy"]
        if strategy_preference == MinMaxStrategy.THREE_TLU_CASTED:
            return self.minimum_maximum_with_trick(
                resulting_type,
                x,
                y,
                x_minus_y_dtype,
                maximum_trick_table,
            )

        return self.minimum_maximum_with_chunks(resulting_type, x, y, operation="max")

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

    def minimum(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        maximum_input_bit_width = max(x.bit_width, y.bit_width)
        if maximum_input_bit_width > MAXIMUM_TLU_BIT_WIDTH:
            highlights = {
                self.converting: [
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit minimum operation is supported"
                ],
            }
            for operand in [x, y]:
                if operand.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                    highlights[operand.origin] = [
                        f"this {operand.bit_width}-bit value "
                        f"is used as an operand to a minimum operation"
                    ]
                    if operand.bit_width != operand.original_bit_width:  # pragma: no cover
                        highlights[operand.origin].append(
                            "("
                            f"note that it's assigned {operand.bit_width}-bits "
                            f"during compilation because of its relation with other operations"
                            ")"
                        )
            self.error(highlights)

        if y.bit_width != resulting_type.bit_width:
            x_can_be_added_directly = resulting_type.bit_width == x.bit_width
            x_has_smaller_bit_width = x.bit_width < y.bit_width
            if x_can_be_added_directly or x_has_smaller_bit_width:
                x, y = y, x

        assert resulting_type.is_encrypted and x.is_encrypted and y.is_encrypted

        x_dtype = Integer(is_signed=x.is_signed, bit_width=x.original_bit_width)
        y_dtype = Integer(is_signed=y.is_signed, bit_width=y.original_bit_width)

        x_minus_y_min = x_dtype.min() - y_dtype.max()
        x_minus_y_max = x_dtype.max() - y_dtype.min()

        x_minus_y_range = [x_minus_y_min, x_minus_y_max]
        x_minus_y_dtype = Integer.that_can_represent(x_minus_y_range)

        minimum_trick_table = []
        if x_minus_y_dtype.is_signed:
            minimum_trick_table += [0] * 2 ** (x_minus_y_dtype.bit_width - 1)
            for value in range(-(2 ** (x_minus_y_dtype.bit_width - 1)), 0):
                minimum_trick_table.append(value)
        else:
            minimum_trick_table += [0] * 2**x_minus_y_dtype.bit_width

        if x_minus_y_dtype.bit_width <= maximum_input_bit_width:
            return self.minimum_maximum_with_trick(
                resulting_type,
                x,
                y,
                x_minus_y_dtype,
                minimum_trick_table,
            )

        strategy_preference = self.converting.properties["strategy"]
        if strategy_preference == MinMaxStrategy.THREE_TLU_CASTED:
            return self.minimum_maximum_with_trick(
                resulting_type,
                x,
                y,
                x_minus_y_dtype,
                minimum_trick_table,
            )

        return self.minimum_maximum_with_chunks(resulting_type, x, y, operation="min")

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

        if x.is_encrypted and y.is_encrypted:
            assert self.is_bit_width_compatible(x, y)
        else:
            assert self.is_bit_width_compatible(resulting_type, x, y)

        use_linalg = x.is_tensor or y.is_tensor

        x = self.tensorize(x) if use_linalg else x
        y = self.tensorize(y) if use_linalg else y

        if x.is_clear:
            x, y = y, x

        dialect = fhelinalg if use_linalg else fhe
        operation = dialect.MulEintIntOp if y.is_clear else dialect.MulEintOp

        if (
            (x.is_signed or y.is_signed)
            and resulting_type.is_unsigned
            and (x.is_encrypted and y.is_encrypted)
        ):
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

    def multivariate_tlu(
        self,
        resulting_type: ConversionType,
        xs: List[Conversion],
        table: Sequence[int],
    ) -> Conversion:
        assert resulting_type.is_encrypted

        packing = self.pack_multivariate_inputs(xs)
        return self.tlu(resulting_type, packing, table)

    def multivariate_multi_tlu(
        self,
        resulting_type: ConversionType,
        xs: List[Conversion],
        tables: Any,
        mapping: Any,
    ):
        assert resulting_type.is_encrypted

        packing = self.pack_multivariate_inputs(xs)
        return self.multi_tlu(resulting_type, packing, tables, mapping)

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

    def not_equal(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        y: Conversion,
    ) -> Conversion:
        return self.comparison(resulting_type, x, y, accept={Comparison.LESS, Comparison.GREATER})

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

    def round_bit_pattern(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        lsbs_to_remove: int,
    ) -> Conversion:
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear round bit pattern is not supported",
            }
            self.error(highlights)

        assert x.bit_width > lsbs_to_remove

        intermediate_type = self.typeof(
            ValueDescription(
                dtype=Integer(is_signed=x.is_signed, bit_width=(x.bit_width - lsbs_to_remove)),
                shape=x.shape,
                is_encrypted=x.is_encrypted,
            )
        )

        rounded = self.operation(
            fhe.RoundEintOp if x.is_scalar else fhelinalg.RoundOp,
            intermediate_type,
            x.result,
        )

        return self.to_signedness(rounded, of=resulting_type)

    def shift(
        self,
        resulting_type: ConversionType,
        x: Conversion,
        b: Conversion,
        orientation: str,
        original_resulting_bit_width: int,
    ) -> Conversion:
        if x.is_signed or b.is_signed:
            highlights: Dict[Node, Union[str, List[str]]] = {
                self.converting: "but only unsigned-unsigned bitwise shifts are supported",
            }
            if x.is_signed:
                highlights[x.origin] = "lhs is signed"
            if b.is_signed:
                highlights[b.origin] = (
                    "rhs is signed" if x.origin is not b.origin else "operand is signed"
                )
            self.error(highlights)

        maximum_input_bit_width = max(x.original_bit_width, b.original_bit_width)
        if (
            maximum_input_bit_width > MAXIMUM_TLU_BIT_WIDTH
            or original_resulting_bit_width > MAXIMUM_TLU_BIT_WIDTH
        ):
            highlights = {
                self.converting: [
                    f"this shift operation resulted in {resulting_type.bit_width}-bits "
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit shift operations are supported"
                ],
            }
            if resulting_type.bit_width != original_resulting_bit_width:  # pragma: no cover
                assert isinstance(highlights[self.converting], list)
                highlights[self.converting].append(  # type: ignore
                    "("
                    f"note that it's assigned {resulting_type.bit_width}-bits "
                    f"during compilation because of its relation with other operations"
                    ")"
                )

            highlights[x.origin] = [
                f"this {x.bit_width}-bit value is used as the operand of a shift operation"
            ]
            if x.bit_width != x.original_bit_width:  # pragma: no cover
                assert isinstance(highlights[x.origin], list)
                highlights[x.origin].append(  # type: ignore
                    "("
                    f"note that it's assigned {x.bit_width}-bits "
                    f"during compilation because of its relation with other operations"
                    ")"
                )

            highlights[b.origin] = [
                f"this {b.bit_width}-bit value is used as the shift amount of a shift operation"
            ]
            if b.bit_width != b.original_bit_width:  # pragma: no cover
                assert isinstance(highlights[b.origin], list)
                highlights[b.origin].append(  # type: ignore
                    "("
                    f"note that it's assigned {b.bit_width}-bits "
                    f"during compilation because of its relation with other operations"
                    ")"
                )

            self.error(highlights)

        assert resulting_type.is_encrypted and x.is_encrypted and b.is_encrypted

        strategy_preference = self.converting.properties["strategy"]
        if x.original_bit_width + b.original_bit_width <= maximum_input_bit_width or (
            strategy_preference == BitwiseStrategy.THREE_TLU_CASTED
        ):
            intermediate_bit_width = max(
                x.original_bit_width + b.original_bit_width,
                maximum_input_bit_width,
            )
            intermediate_scalar_type = self.eint(intermediate_bit_width)

            if x.bit_width != intermediate_bit_width:
                x = self.cast(self.tensor(intermediate_scalar_type, x.shape), x)
            if b.bit_width != intermediate_bit_width:
                b = self.cast(self.tensor(intermediate_scalar_type, b.shape), b)

            shift_multiplier_type = self.typeof(
                ValueDescription(
                    dtype=Integer(is_signed=False, bit_width=(x.bit_width + 1)),
                    shape=(),
                    is_encrypted=False,
                )
            )
            shift_multiplier = self.constant(shift_multiplier_type, 2**b.original_bit_width)

            shifted_x = self.mul(
                self.tensor(intermediate_scalar_type, x.shape),
                x,
                shift_multiplier,
            )
            packed_x_and_b = self.add(
                self.tensor(intermediate_scalar_type, resulting_type.shape),
                shifted_x,
                b,
            )

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

        for i in reversed(range(b.original_bit_width)):
            to_check = 2**i

            shifter = (
                [(x << to_check) - x for x in range(2**x.original_bit_width)]
                if orientation == "left"
                else [x - (x >> to_check) for x in range(2**x.original_bit_width)]
            )
            shifter_dtype = Integer.that_can_represent(shifter)

            assert not shifter_dtype.is_signed
            chunk_size = int(np.ceil(shifter_dtype.bit_width / 2))
            packing_scalar_type = self.eint(chunk_size + 1)

            should_shift = self.tlu(
                self.tensor(packing_scalar_type, b.shape),
                b,
                [int((b & to_check) > 0) for b in range(2**b.original_bit_width)],
            )

            chunks = []
            for offset in range(0, original_resulting_bit_width, chunk_size):
                bits_to_process = min(chunk_size, original_resulting_bit_width - offset)
                right_shift_by = original_resulting_bit_width - offset - bits_to_process
                mask = (2**bits_to_process) - 1

                chunk_x = self.tlu(
                    self.tensor(packing_scalar_type, x.shape),
                    x,
                    [
                        (((shifter[x] >> right_shift_by) & mask) << 1)
                        for x in range(2**x.original_bit_width)
                    ],
                )
                packed_chunk_x_and_should_shift = self.add(
                    self.tensor(packing_scalar_type, resulting_type.shape),
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

            if x.bit_width != resulting_type.bit_width:
                x.set_original_bit_width(original_resulting_bit_width)
                x = self.cast(self.tensor(self.eint(resulting_type.bit_width), x.shape), x)

            x = (
                self.add(resulting_type, difference, x)
                if orientation == "left"
                else self.sub(resulting_type, x, difference)
            )
            x.set_original_bit_width(original_resulting_bit_width)

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

        if all(value == table[0] for value in table[1:]):
            value = table[0]
            result = self.zeros(resulting_type)
            if value != 0:
                constant = self.constant(self.i(resulting_type.bit_width + 1), value)
                result = self.add(resulting_type, result, constant)
            return result

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
        resulting_element_type = (self.eint if resulting_type.is_unsigned else self.esint)(
            resulting_type.bit_width
        )
        while len(xs) > 1:
            a = xs.pop()
            b = xs.pop()

            intermediate_type = self.tensor(
                resulting_element_type,
                shape=(np.zeros(a.shape) + np.zeros(b.shape)).shape,
            )

            result = self.add(intermediate_type, a, b)
            xs.insert(0, result)

        return xs[0]

    def truncate_bit_pattern(self, x: Conversion, lsbs_to_remove: int) -> Conversion:
        if x.is_clear:
            highlights = {
                x.origin: "operand is clear",
                self.converting: "but clear truncate bit pattern is not supported",
            }
            self.error(highlights)

        assert x.bit_width > lsbs_to_remove

        resulting_bit_width = x.bit_width
        for i in range(lsbs_to_remove):
            lsb = self.lsb(x.type, x)
            cleared = self.sub(x.type, x, lsb)

            new_bit_width = (x.bit_width - 1) if i != (lsbs_to_remove - 1) else resulting_bit_width
            x = self.reinterpret(cleared, bit_width=new_bit_width)

        return x

    def reinterpret(self, x: Conversion, *, bit_width: int) -> Conversion:
        assert x.is_encrypted

        resulting_element_type = (self.eint if x.is_unsigned else self.esint)(bit_width)
        resulting_type = self.tensor(resulting_element_type, shape=x.shape)

        operation = (
            fhe.ReinterpretPrecisionEintOp if x.is_scalar else fhelinalg.ReinterpretPrecisionEintOp
        )
        return self.operation(operation, resulting_type, x.result)

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
