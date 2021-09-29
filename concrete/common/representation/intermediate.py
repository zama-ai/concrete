"""File containing code to represent source programs operations."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type

from loguru import logger

from ..data_types.base import BaseDataType
from ..data_types.dtypes_helpers import (
    get_base_value_for_python_constant_data,
    mix_values_determine_holding_dtype,
)
from ..data_types.integers import Integer
from ..debugging.custom_assert import custom_assert
from ..values import BaseValue, ClearScalar, EncryptedScalar, TensorValue

IR_MIX_VALUES_FUNC_ARG_NAME = "mix_values_func"

ALL_IR_NODES: Set[Type] = set()


class IntermediateNode(ABC):
    """Abstract Base Class to derive from to represent source program operations."""

    inputs: List[BaseValue]
    outputs: List[BaseValue]
    _n_in: int  # _n_in indicates how many inputs are required to evaluate the IntermediateNode

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        **_kwargs,  # This is to be able to feed arbitrary arguments to IntermediateNodes
    ) -> None:
        self.inputs = list(inputs)
        custom_assert(all(isinstance(x, BaseValue) for x in self.inputs))

    # Register all IR nodes
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ALL_IR_NODES.add(cls)

    def _init_binary(
        self,
        inputs: Iterable[BaseValue],
        mix_values_func: Callable[..., BaseValue] = mix_values_determine_holding_dtype,
        **_kwargs,  # Required to conform to __init__ typing
    ) -> None:
        """__init__ for a binary operation, ie two inputs."""
        IntermediateNode.__init__(self, inputs)

        custom_assert(len(self.inputs) == 2)

        self.outputs = [mix_values_func(self.inputs[0], self.inputs[1])]

    @abstractmethod
    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        """Simulate what the represented computation would output for the given inputs.

        Args:
            inputs (Dict[int, Any]): Dict containing the inputs for the evaluation

        Returns:
            Any: the result of the computation
        """

    @classmethod
    def n_in(cls) -> int:
        """Return how many inputs the node has.

        Returns:
            int: The number of inputs of the node.
        """
        return cls._n_in

    @classmethod
    def requires_mix_values_func(cls) -> bool:
        """Determine whether the Class requires a mix_values_func to be built.

        Returns:
            bool: True if __init__ expects a mix_values_func argument.
        """
        return cls.n_in() > 1

    @abstractmethod
    def label(self) -> str:
        """Get the label of the node.

        Returns:
            str: the label of the node

        """


class Add(IntermediateNode):
    """Addition between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] + inputs[1]

    def label(self) -> str:
        return "+"


class Sub(IntermediateNode):
    """Subtraction between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] - inputs[1]

    def label(self) -> str:
        return "-"


class Mul(IntermediateNode):
    """Multiplication between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] * inputs[1]

    def label(self) -> str:
        return "*"


class Input(IntermediateNode):
    """Node representing an input of the program."""

    input_name: str
    program_input_idx: int
    _n_in: int = 1

    def __init__(
        self,
        input_value: BaseValue,
        input_name: str,
        program_input_idx: int,
    ) -> None:
        super().__init__((input_value,))
        custom_assert(len(self.inputs) == 1)
        self.input_name = input_name
        self.program_input_idx = program_input_idx
        self.outputs = [deepcopy(self.inputs[0])]

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0]

    def label(self) -> str:
        return self.input_name


class Constant(IntermediateNode):
    """Node representing a constant of the program."""

    _constant_data: Any
    _n_in: int = 0

    def __init__(
        self,
        constant_data: Any,
        get_base_value_for_data_func: Callable[
            [Any], Callable[..., BaseValue]
        ] = get_base_value_for_python_constant_data,
    ) -> None:
        super().__init__([])

        base_value_class = get_base_value_for_data_func(constant_data)

        self._constant_data = constant_data
        self.outputs = [base_value_class(is_encrypted=False)]

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return self.constant_data

    @property
    def constant_data(self) -> Any:
        """Return the constant_data stored in the Constant node.

        Returns:
            Any: The constant data that was stored.
        """
        return self._constant_data

    def label(self) -> str:
        return str(self.constant_data)


class ArbitraryFunction(IntermediateNode):
    """Node representing a univariate arbitrary function, e.g. sin(x)."""

    # The arbitrary_func is not optional but mypy has a long standing bug and is not able to
    # understand this properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    arbitrary_func: Optional[Callable]
    op_args: Tuple[Any, ...]
    op_kwargs: Dict[str, Any]
    op_name: str
    _n_in: int = 1

    def __init__(
        self,
        input_base_value: BaseValue,
        arbitrary_func: Callable,
        output_dtype: BaseDataType,
        op_name: Optional[str] = None,
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__([input_base_value])
        custom_assert(len(self.inputs) == 1)
        self.arbitrary_func = arbitrary_func
        self.op_args = op_args if op_args is not None else ()
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}

        output = deepcopy(input_base_value)
        output.dtype = output_dtype
        self.outputs = [output]

        self.op_name = op_name if op_name is not None else self.__class__.__name__

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        # This is the continuation of the mypy bug workaround
        assert self.arbitrary_func is not None
        return self.arbitrary_func(inputs[0], *self.op_args, **self.op_kwargs)

    def label(self) -> str:
        return self.op_name

    def get_table(self) -> List[Any]:
        """Get the table for the current input value of this ArbitraryFunction.

        This function only works if the ArbitraryFunction input value is an unsigned Integer.

        Returns:
            List[Any]: The table.
        """
        # Check the input is an unsigned integer to be able to build a table
        assert isinstance(
            self.inputs[0].dtype, Integer
        ), "get_table only works for an unsigned Integer input"
        assert not self.inputs[
            0
        ].dtype.is_signed, "get_table only works for an unsigned Integer input"

        type_constructor = self.inputs[0].dtype.underlying_type_constructor
        if type_constructor is None:
            logger.info(
                f"{self.__class__.__name__} input data type constructor was None, defaulting to int"
            )
            type_constructor = int

        min_input_range = self.inputs[0].dtype.min_value()
        max_input_range = self.inputs[0].dtype.max_value() + 1

        table = [
            self.evaluate({0: type_constructor(input_value)})
            for input_value in range(min_input_range, max_input_range)
        ]

        return table


def default_dot_evaluation_function(lhs: Any, rhs: Any) -> Any:
    """Return the default python dot implementation for 1D iterable arrays.

    Args:
        lhs (Any): lhs vector of the dot.
        rhs (Any): rhs vector of the dot.

    Returns:
        Any: the result of the dot operation.
    """
    return sum(lhs * rhs for lhs, rhs in zip(lhs, rhs))


class Dot(IntermediateNode):
    """Return the node representing a dot product."""

    _n_in: int = 2
    # Optional, same issue as in ArbitraryFunction for mypy
    evaluation_function: Optional[Callable[[Any, Any], Any]]
    # Allows to use specialized implementations from e.g. numpy

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        output_dtype: BaseDataType,
        delegate_evaluation_function: Optional[
            Callable[[Any, Any], Any]
        ] = default_dot_evaluation_function,
    ) -> None:
        super().__init__(inputs)
        custom_assert(len(self.inputs) == 2)

        custom_assert(
            all(
                isinstance(input_value, TensorValue) and input_value.ndim == 1
                for input_value in self.inputs
            ),
            f"Dot only supports two vectors ({TensorValue.__name__} with ndim == 1)",
        )

        output_scalar_value = (
            EncryptedScalar
            if (self.inputs[0].is_encrypted or self.inputs[1].is_encrypted)
            else ClearScalar
        )

        self.outputs = [output_scalar_value(output_dtype)]
        self.evaluation_function = delegate_evaluation_function

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        # This is the continuation of the mypy bug workaround
        assert self.evaluation_function is not None
        return self.evaluation_function(inputs[0], inputs[1])

    def label(self) -> str:
        return "dot"
