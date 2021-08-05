"""File containing code to represent source programs operations"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..data_types import BaseValue
from ..data_types.dtypes_helpers import mix_values_determine_holding_dtype
from ..data_types.floats import Float
from ..data_types.integers import Integer, get_bits_to_represent_int
from ..data_types.scalars import Scalars
from ..data_types.values import ClearValue


class IntermediateNode(ABC):
    """Abstract Base Class to derive from to represent source program operations"""

    inputs: List[BaseValue]
    outputs: List[BaseValue]
    op_args: Optional[Tuple[Any, ...]]
    op_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.inputs = list(inputs)
        assert all(isinstance(x, BaseValue) for x in self.inputs)
        self.op_args = op_args
        self.op_kwargs = op_kwargs

    def _init_binary(
        self,
        inputs: Iterable[BaseValue],
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert op_args is None, f"Expected op_args to be None, got {op_args}"
        assert op_kwargs is None, f"Expected op_kwargs to be None, got {op_kwargs}"

        IntermediateNode.__init__(self, inputs, op_args=op_args, op_kwargs=op_kwargs)

        assert len(self.inputs) == 2

        self.outputs = [mix_values_determine_holding_dtype(self.inputs[0], self.inputs[1])]

    def _is_equivalent_to_binary_commutative(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and (self.inputs == other.inputs or self.inputs == other.inputs[::-1])
            and self.outputs == other.outputs
        )

    def _is_equivalent_to_binary_non_commutative(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def is_equivalent_to(self, other: object) -> bool:
        """Overriding __eq__ has unwanted side effects, this provides the same facility without
            disrupting expected behavior too much

        Args:
            other (object): Other object to check against

        Returns:
            bool: True if the other object is equivalent
        """
        return (
            isinstance(other, self.__class__)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
            and self.op_args == other.op_args
            and self.op_kwargs == other.op_kwargs
        )

    @abstractmethod
    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        """Function to simulate what the represented computation would output for the given inputs

        Args:
            inputs (Mapping[int, Any]): Mapping containing the inputs for the evaluation

        Returns:
            Any: the result of the computation
        """


class Add(IntermediateNode):
    """Addition between two values"""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] + inputs[1]


class Sub(IntermediateNode):
    """Subtraction between two values"""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_non_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] - inputs[1]


class Mul(IntermediateNode):
    """Multiplication between two values"""

    __init__ = IntermediateNode._init_binary
    is_equivalent_to = IntermediateNode._is_equivalent_to_binary_commutative

    def evaluate(self, inputs: Mapping[int, Any]) -> Any:
        return inputs[0] * inputs[1]


class Input(IntermediateNode):
    """Node representing an input of the program"""

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


class ConstantInput(IntermediateNode):
    """Node representing a constant of the program"""

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
