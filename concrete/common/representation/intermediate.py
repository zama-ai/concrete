"""File containing code to represent source programs operations."""

from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from enum import Enum, unique
from math import floor
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, cast

import numpy as np
import torch
from loguru import logger

from ..data_types.base import BaseDataType
from ..data_types.dtypes_helpers import (
    get_base_value_for_python_constant_data,
    mix_values_determine_holding_dtype,
)
from ..data_types.integers import Integer
from ..debugging.custom_assert import assert_true
from ..helpers import indexing_helpers
from ..helpers.formatting_helpers import format_constant
from ..helpers.python_helpers import catch, update_and_return_dict
from ..values import BaseValue, ClearTensor, EncryptedTensor, TensorValue

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
        assert_true(all(isinstance(x, BaseValue) for x in self.inputs))

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

        assert_true(len(self.inputs) == 2)

        self.outputs = [mix_values_func(self.inputs[0], self.inputs[1])]

    def text_for_formatting(self, predecessors: List[str], _maximum_constant_length: int) -> str:
        """Get the formatted node (used in formatting operation graphs).

        Args:
            predecessors (List[str]): predecessor names to this node
            _maximum_constant_length (int): desired maximum constant length

        Returns:
            str: the formatted node
        """

        return f"{self.__class__.__name__.lower()}({', '.join(predecessors)})"

    @abstractmethod
    def text_for_drawing(self) -> str:
        """Get the label of the node (used in drawing operation graphs).

        Returns:
            str: the label of the node
        """

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


class Add(IntermediateNode):
    """Addition between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def text_for_drawing(self) -> str:
        return "+"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] + inputs[1]


class Sub(IntermediateNode):
    """Subtraction between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def text_for_drawing(self) -> str:
        return "-"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] - inputs[1]


class Mul(IntermediateNode):
    """Multiplication between two values."""

    _n_in: int = 2

    __init__ = IntermediateNode._init_binary

    def text_for_drawing(self) -> str:
        return "*"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] * inputs[1]


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
        assert_true(len(self.inputs) == 1)
        self.input_name = input_name
        self.program_input_idx = program_input_idx
        self.outputs = [deepcopy(self.inputs[0])]

    def text_for_formatting(self, predecessors: List[str], _maximum_constant_length: int) -> str:
        assert_true(len(predecessors) == 0)
        return self.input_name

    def text_for_drawing(self) -> str:
        return self.input_name

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0]


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

    def text_for_formatting(self, predecessors: List[str], maximum_constant_length: int) -> str:
        assert_true(len(predecessors) == 0)
        return format_constant(self.constant_data, maximum_constant_length)

    def text_for_drawing(self) -> str:
        return format_constant(self.constant_data)

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return self.constant_data

    @property
    def constant_data(self) -> Any:
        """Return the constant_data stored in the Constant node.

        Returns:
            Any: The constant data that was stored.
        """
        return self._constant_data


class Conv2D(IntermediateNode):
    """Return the node representing a 2d-convolution."""

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        output_dtype: BaseDataType,
        pads: Union[List[int], Tuple[int, int, int, int]],
        strides: Union[List[int], Tuple[int, int]],
        dilations: Union[List[int], Tuple[int, int]],
    ) -> None:

        # TODO: remove this when padding is supported (#427)
        assert all(pad == 0 for pad in pads), "conv2d doesn't support padding yet"

        super().__init__(inputs)
        self.pads = pads
        self.strides = strides
        self.dilations = dilations

        self._n_in = len(self.inputs)
        assert_true(len(self.inputs) == 2 or len(self.inputs) == 3)

        assert_true(
            all(
                isinstance(input_value, TensorValue) and input_value.ndim == 4
                for input_value in self.inputs[:2]
            ),
            f"Conv2D only supports input and weight tensors of 4 dimensions"
            f"({TensorValue.__name__} with ndim == 4)",
        )
        bias = cast(TensorValue, self.inputs[2]) if len(self.inputs) == 3 else None
        if bias is not None:
            assert_true(
                isinstance(bias, TensorValue) and bias.ndim == 1,
                f"Conv2D only supports bias 1 dimension ({TensorValue.__name__} with ndim == 1)",
            )

        x = cast(TensorValue, self.inputs[0])
        weight = cast(TensorValue, self.inputs[1])

        # Compute output shape
        input_n, _, input_h, input_w = x.shape
        weight_f, _, weight_h, weight_w = weight.shape
        pads_h = pads[0] + pads[2]
        pads_w = pads[1] + pads[3]
        output_h = floor((input_h + pads_h - dilations[0] * (weight_h - 1) - 1) / strides[0]) + 1
        output_w = floor((input_w + pads_w - dilations[1] * (weight_w - 1) - 1) / strides[1]) + 1
        output_shape = (input_n, weight_f, output_h, output_w)

        output_value = EncryptedTensor(dtype=output_dtype, shape=output_shape)
        self.outputs = [output_value]

    def text_for_drawing(self) -> str:
        return "conv2d"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:

        assert_true(
            len(inputs) == self._n_in, f"expected {self.n_in} inputs, but got {len(inputs)}"
        )
        x, weight = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) == 3 else np.zeros(weight.shape[0])

        return self.evaluate_conv2d(x, weight, bias, self.pads, self.strides, self.dilations)

    @staticmethod
    def evaluate_conv2d(
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        # TODO: use padding when supported (#427)
        _: Union[Tuple[int, int, int, int], List[int]],
        strides: Union[Tuple[int, int], List[int]],
        dilations: Union[Tuple[int, int], List[int]],
    ):
        """Evaluate 2D convolution.

        Args:
            x (np.ndarray): Input of shape (NxCxHxW)
            weight (np.ndarray): Weight (kernel) of shape (FxCxHxW)
            bias (np.ndarray): Bias vector of size (F)
            pads (Union[Tuple[int, int, int, int], List[int]]): Padding over each
                axis (H_beg, W_beg, H_end, W_end)
            strides (Union[Tuple[int, int], List[int]]): Stride over each
                axis (height and width)
            dilations (Union[Tuple[int, int], List[int]]): Dilation over each
                axis (height and width)

        Returns:
            np.ndarray: Result of the convolution of shape (NxCxHxW)
        """
        # pylint: disable=no-member
        return torch.conv2d(
            torch.tensor(x, dtype=torch.long),
            torch.tensor(weight, dtype=torch.long),
            torch.tensor(bias, dtype=torch.long),
            stride=strides,
            dilation=dilations,
        ).numpy()
        # pylint: enable=no-member


class IndexConstant(IntermediateNode):
    """Node representing a constant indexing in the program.

    What we mean by constant indexing is that the index part of the operation is a constant.
    Here are some examples: `x[2]`, `x[0, 1]`, `y[:, 0]`, `y[3:, :5]`

    The opposite is to have dynamic indexing, which this node does not support.
    Some examples of dynamic indexing are: `x[y]`, `x[y, z]`, `x[:, y]`
    """

    _n_in: int = 1

    index: Tuple[Union[int, slice], ...]

    def __init__(
        self,
        input_: BaseValue,
        index: Union[int, slice, Tuple[Union[int, slice], ...]],
    ) -> None:
        super().__init__((input_,))

        if not isinstance(self.inputs[0], TensorValue) or self.inputs[0].is_scalar:
            raise TypeError(f"Only tensors can be indexed but you tried to index {self.inputs[0]}")

        self.index = indexing_helpers.validate_index(index)

        output_dtype = self.inputs[0].dtype
        output_shape = indexing_helpers.determine_output_shape(self.inputs[0].shape, self.index)

        self.outputs = [
            EncryptedTensor(output_dtype, output_shape)
            if self.inputs[0].is_encrypted
            else ClearTensor(output_dtype, output_shape)
        ]

    def text_for_formatting(self, predecessors: List[str], _maximum_constant_length: int) -> str:
        assert_true(len(predecessors) == 1)
        elements = [indexing_helpers.format_indexing_element(element) for element in self.index]
        index = ", ".join(elements)
        return f"{predecessors[0]}[{index}]"

    def text_for_drawing(self) -> str:
        return self.text_for_formatting(["value"], 0)  # 0 is unused

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0][self.index]


def flood_replace_none_values(table: list):
    """Use a flooding algorithm to replace None values.

    Args:
        table (list): the list in which there are None values that need to be replaced by copies of
            the closest non None data from the list.
    """
    assert_true(any(value is not None for value in table))

    not_none_values_idx = deque(idx for idx, value in enumerate(table) if value is not None)
    while not_none_values_idx:
        current_idx = not_none_values_idx.popleft()
        current_value = table[current_idx]
        previous_idx = current_idx - 1
        next_idx = current_idx + 1
        if previous_idx >= 0 and table[previous_idx] is None:
            table[previous_idx] = deepcopy(current_value)
            not_none_values_idx.append(previous_idx)
        if next_idx < len(table) and table[next_idx] is None:
            table[next_idx] = deepcopy(current_value)
            not_none_values_idx.append(next_idx)

    assert_true(all(value is not None for value in table))


@unique
class GenericFunctionKind(str, Enum):
    """Enum to validate GenericFunction op_kind."""

    TLU = "TLU"
    MEMORY = "Memory"


class GenericFunction(IntermediateNode):
    """Node representing an arbitrary function with a single output, e.g. sin(x)."""

    # The arbitrary_func is not optional but mypy has a long standing bug and is not able to
    # understand this properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    # arbitrary_func can take more than one argument but during evaluation the input variable will
    # be the first argument passed to it. You can add other constant arguments needed for the proper
    # execution of the function through op_args and op_kwargs.
    arbitrary_func: Optional[Callable]
    op_kind: GenericFunctionKind
    op_name: str
    op_args: Tuple[Any, ...]
    op_kwargs: Dict[str, Any]
    op_attributes: Dict[str, Any]
    _n_in: int

    # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/798 have a proper
    # attribute system
    DEFAULT_OP_ATTRIBUTES: Dict[str, Any] = {"fusable": True}

    KWARGS_IGNORED_IN_FORMATTING: Set[str] = {
        "float_op_subgraph",
        "terminal_node",
    }

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        arbitrary_func: Callable,
        output_value: BaseValue,
        op_kind: Union[str, GenericFunctionKind],
        op_name: Optional[str] = None,
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
        op_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__([deepcopy(i) for i in inputs])
        self._n_in = len(self.inputs)
        self.arbitrary_func = arbitrary_func
        self.op_kind = GenericFunctionKind(op_kind)
        self.op_args = op_args if op_args is not None else ()
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.op_attributes = deepcopy(self.DEFAULT_OP_ATTRIBUTES)
        if op_attributes is not None:
            self.op_attributes.update(op_attributes)

        self.outputs = [output_value]

        self.op_name = op_name if op_name is not None else self.__class__.__name__

    def text_for_formatting(self, predecessors: List[str], maximum_constant_length: int) -> str:
        if self.op_name == "concat":
            all_args = ["(" + ", ".join(predecessors) + ")"]
        else:
            all_args = deepcopy(predecessors)

        all_args.extend(format_constant(value, maximum_constant_length) for value in self.op_args)
        all_args.extend(
            f"{name}={format_constant(value, maximum_constant_length)}"
            for name, value in self.op_kwargs.items()
            if name not in GenericFunction.KWARGS_IGNORED_IN_FORMATTING
        )

        return f"{self.op_name}({', '.join(all_args)})"

    def text_for_drawing(self) -> str:
        return self.op_name

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        # This is the continuation of the mypy bug workaround
        assert self.arbitrary_func is not None
        ordered_inputs = [inputs[idx] for idx in range(len(inputs))]
        if self.op_name == "concat":
            return self.arbitrary_func(tuple(ordered_inputs), *self.op_args, **self.op_kwargs)
        return self.arbitrary_func(*ordered_inputs, *self.op_args, **self.op_kwargs)

    def get_table(self, ordered_preds: List[IntermediateNode]) -> List[Any]:
        """Get the table for the current input value of this GenericFunction.

        This function only works if the GenericFunction variable input value is an Integer.
        This function only works if there is a single variable input node among ordered_preds.

        Args:
            ordered_preds (List[IntermediateNode]): List of predecessors of the node. This list must
                contain a single non constant node and any number of Constant nodes.

        Returns:
            List[Any]: The table.
        """

        variable_input_indices = [
            idx for idx, pred in enumerate(ordered_preds) if not isinstance(pred, Constant)
        ]

        assert_true(
            (non_constant_pred_count := len(variable_input_indices)) == 1,
            f"Can only have 1 non constant predecessor in {self.get_table.__name__}, "
            f"got {non_constant_pred_count}",
        )

        variable_input_idx = variable_input_indices[0]
        variable_input_dtype = self.inputs[variable_input_idx].dtype
        # Check the input is an integer to be able to build a table
        assert_true(
            isinstance(variable_input_dtype, Integer),
            f"{self.get_table.__name__} only works for an unsigned Integer input",
        )
        variable_input_dtype = cast(Integer, variable_input_dtype)

        input_value_constructor = self.inputs[variable_input_idx].underlying_constructor
        if input_value_constructor is None:
            logger.info(
                f"{self.__class__.__name__} input data type constructor was None, defaulting to int"
            )
            input_value_constructor = int

        min_input_range = variable_input_dtype.min_value()
        max_input_range = variable_input_dtype.max_value() + 1

        template_input_dict = {
            idx: node.evaluate({}) if isinstance(node, Constant) else None
            for idx, node in enumerate(ordered_preds)
        }

        table = [
            catch(
                self.evaluate,
                update_and_return_dict(
                    template_input_dict, {variable_input_idx: input_value_constructor(input_value)}
                ),
            )
            for input_value in range(min_input_range, max_input_range)
        ]

        flood_replace_none_values(table)

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
    # Optional, same issue as in GenericFunction for mypy
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
        assert_true(len(self.inputs) == 2)

        assert_true(
            all(
                isinstance(input_value, TensorValue) and input_value.ndim <= 1
                for input_value in self.inputs
            ),
            f"Dot only supports two scalars or vectors ({TensorValue.__name__} with ndim up to 1)",
        )

        lhs = cast(TensorValue, self.inputs[0])
        rhs = cast(TensorValue, self.inputs[1])

        if lhs.ndim == 1 and rhs.ndim == 1:
            assert_true(
                lhs.shape[0] == rhs.shape[0],
                f"Dot between vectors of shapes {lhs.shape} and {rhs.shape} is not supported",
            )

        output_shape: Tuple[int, ...]
        if (lhs.ndim == 1 and rhs.ndim == 1) or (lhs.ndim == 0 and rhs.ndim == 0):
            # numpy.dot(x, y) where x and y are both vectors or both scalars
            output_shape = ()
        elif lhs.ndim == 1:
            # numpy.dot(x, y) where x is a vector and y is a scalar
            output_shape = lhs.shape
        else:
            # numpy.dot(x, y) where x is a scalar and y is a vector
            output_shape = rhs.shape

        output_value = EncryptedTensor if (lhs.is_encrypted or rhs.is_encrypted) else ClearTensor

        self.outputs = [output_value(output_dtype, output_shape)]
        self.evaluation_function = delegate_evaluation_function

    def text_for_drawing(self) -> str:
        return "dot"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        # This is the continuation of the mypy bug workaround
        assert self.evaluation_function is not None
        return self.evaluation_function(inputs[0], inputs[1])


class MatMul(IntermediateNode):
    """Return the node representing a matrix multiplication."""

    _n_in: int = 2

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        output_dtype: BaseDataType,
        output_shape: Tuple[int, ...],
    ) -> None:
        super().__init__(inputs)
        assert_true(len(self.inputs) == 2)

        output_value = (
            EncryptedTensor(dtype=output_dtype, shape=output_shape)
            if (self.inputs[0].is_encrypted or self.inputs[1].is_encrypted)
            else ClearTensor(dtype=output_dtype, shape=output_shape)
        )

        self.outputs = [output_value]

    def text_for_drawing(self) -> str:
        return "matmul"

    def evaluate(self, inputs: Dict[int, Any]) -> Any:
        return inputs[0] @ inputs[1]
