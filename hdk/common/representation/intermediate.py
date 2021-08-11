"""File containing code to represent source programs operations."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ..data_types import BaseValue
from ..data_types.base import BaseDataType
from ..data_types.dtypes_helpers import mix_values_determine_holding_dtype
from ..data_types.floats import Float
from ..data_types.integers import Integer, get_bits_to_represent_int
from ..data_types.scalars import Scalars
from ..data_types.values import ClearValue, EncryptedValue


class IntermediateNode(ABC):
    """Abstract Base Class to derive from to represent source program operations."""

    inputs: List[BaseValue]
    outputs: List[BaseValue]

    def __init__(
        self,
        inputs: Iterable[BaseValue],
    ) -> None:
        self.inputs = list(inputs)
        assert all(isinstance(x, BaseValue) for x in self.inputs)

    def _init_binary(
        self,
        inputs: Iterable[BaseValue],
    ) -> None:
        """__init__ for a binary operation, ie two inputs."""
        IntermediateNode.__init__(self, inputs)

        assert len(self.inputs) == 2

        self.outputs = [mix_values_determine_holding_dtype(self.inputs[0], self.inputs[1])]

    def _is_equivalent_to_binary_commutative(self, other: object) -> bool:
        """is_equivalent_to for a binary and commutative operation."""
        return (
            isinstance(other, self.__class__)
            and (self.inputs == other.inputs or self.inputs == other.inputs[::-1])
            and self.outputs == other.outputs
        )

    def _is_equivalent_to_binary_non_commutative(self, other: object) -> bool:
        """is_equivalent_to for a binary and non-commutative operation."""
        return (
            isinstance(other, self.__class__)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    @abstractmethod
    def is_equivalent_to(self, other: object) -> bool:
        """Alternative to __eq__ to check equivalence between IntermediateNodes.

        Overriding __eq__ has unwanted side effects, this provides the same facility without
        disrupting expected behavior too much

        Args:
            other (object): Other object to check against

        Returns:
            bool: True if the other object is equivalent
        """
        return (
            isinstance(other, IntermediateNode)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    @abstractmethod
    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        """Function to simulate what the represented computation would output for the given inputs.

        Args:
            inputs (Mapping[int, Any]): Mapping containing the inputs for the evaluation

        Returns:
            Any: the result of the computation
        """


class Add(IntermediateNode):
    """Addition between two values."""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] + inputs[1]


class Sub(IntermediateNode):
    """Subtraction between two values."""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_non_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] - inputs[1]


class Mul(IntermediateNode):
    """Multiplication between two values."""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] * inputs[1]


class Input(IntermediateNode):
    """Node representing an input of the program."""

    input_name: str
    program_input_idx: int

    def __init__(
        self,
        input_value: BaseValue,
        input_name: str,
        program_input_idx: int,
    ) -> None:
        super().__init__((input_value,))
        assert len(self.inputs) == 1
        self.input_name = input_name
        self.program_input_idx = program_input_idx
        self.outputs = [deepcopy(self.inputs[0])]

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0]

    def is_equivalent_to(self, other: object) -> bool:
        return (
            isinstance(other, Input)
            and self.input_name == other.input_name
            and self.program_input_idx == other.program_input_idx
            and super().is_equivalent_to(other)
        )


class ConstantInput(IntermediateNode):
    """Node representing a constant of the program."""

    constant_data: Scalars

    def __init__(
        self,
        constant_data: Scalars,
    ) -> None:
        super().__init__([])
        self.constant_data = constant_data

        assert isinstance(
            constant_data, (int, float)
        ), "Only int and float are support for constant input"
        if isinstance(constant_data, int):
            is_signed = constant_data < 0
            self.outputs = [
                ClearValue(Integer(get_bits_to_represent_int(constant_data, is_signed), is_signed))
            ]
        elif isinstance(constant_data, float):
            self.outputs = [ClearValue(Float(64))]

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return self.constant_data

    def is_equivalent_to(self, other: object) -> bool:
        return (
            isinstance(other, ConstantInput)
            and self.constant_data == other.constant_data
            and super().is_equivalent_to(other)
        )


class ArbitraryFunction(IntermediateNode):
    """Node representing a univariate arbitrary function, e.g. sin(x)."""

    # The arbitrary_func is not optional but mypy has a long standing bug and is not able to
    # understand this properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    arbitrary_func: Optional[Callable]
    op_args: Tuple[Any, ...]
    op_kwargs: Dict[str, Any]

    def __init__(
        self,
        input_base_value: BaseValue,
        arbitrary_func: Callable,
        output_dtype: BaseDataType,
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__([input_base_value])
        assert len(self.inputs) == 1
        self.arbitrary_func = arbitrary_func
        self.op_args = deepcopy(op_args) if op_args is not None else ()
        self.op_kwargs = deepcopy(op_kwargs) if op_kwargs is not None else {}
        # TLU/PBS has an encrypted output
        self.outputs = [EncryptedValue(output_dtype)]

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        # This is the continuation of the mypy bug workaround
        assert self.arbitrary_func is not None
        return self.arbitrary_func(inputs[0], *self.op_args, **self.op_kwargs)

    def is_equivalent_to(self, other: object) -> bool:
        # FIXME: comparing self.arbitrary_func to other.arbitrary_func will not work
        # Only evaluating over the same set of inputs and comparing will help
        return (
            isinstance(other, ArbitraryFunction)
            and self.op_args == other.op_args
            and self.op_kwargs == other.op_kwargs
            and super().is_equivalent_to(other)
        )
