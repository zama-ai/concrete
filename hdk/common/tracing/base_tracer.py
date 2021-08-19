"""This file holds the code that can be shared between tracers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Tuple, Type, Union

from ..data_types import BaseValue
from ..representation import intermediate as ir
from ..representation.intermediate import IR_MIX_VALUES_FUNC_ARG_NAME


class BaseTracer(ABC):
    """Base class for implementing tracers."""

    inputs: List["BaseTracer"]
    traced_computation: ir.IntermediateNode
    output: BaseValue
    _mix_values_func: Callable[..., BaseValue]

    def __init__(
        self,
        inputs: Iterable["BaseTracer"],
        traced_computation: ir.IntermediateNode,
        output_index: int,
    ) -> None:
        self.inputs = list(inputs)
        self.traced_computation = traced_computation
        self.output = traced_computation.outputs[output_index]

    @abstractmethod
    def _supports_other_operand(self, other: Any) -> bool:
        """Function to check if the current class supports tracing with the other operand.

        Args:
            other (Any): the operand to check compatibility with.

        Returns:
            bool: True if the tracer can manage operations with the other operand.
        """
        return isinstance(other, self.__class__)

    @abstractmethod
    def _make_const_input_tracer(self, constant_data: Any) -> "BaseTracer":
        """Helper function to create a tracer for a constant input.

        Args:
            constant_data (Any): The constant to store.

        Returns:
            BaseTracer: The BaseTracer for that constant.
        """

    @classmethod
    def _get_mix_values_func(cls):
        return cls._mix_values_func

    def instantiate_output_tracers(
        self,
        inputs: Iterable[Union["BaseTracer", Any]],
        computation_to_trace: Type[ir.IntermediateNode],
    ) -> Tuple["BaseTracer", ...]:
        """Helper functions to instantiate all output BaseTracer for a given computation.

        Args:
            inputs (Iterable[Union[BaseTracer, Any]]): Previous BaseTracer or data used as inputs
                for a new node.
            computation_to_trace (Type[ir.IntermediateNode]): The IntermediateNode class
                to instantiate for the computation being traced

        Returns:
            Tuple[BaseTracer, ...]: A tuple containing an BaseTracer per output function
        """
        # For inputs which are actually constant, first convert into a tracer
        def sanitize(inp):
            if not isinstance(inp, BaseTracer):
                return self._make_const_input_tracer(inp)
            return inp

        sanitized_inputs = [sanitize(inp) for inp in inputs]

        additional_parameters = (
            {IR_MIX_VALUES_FUNC_ARG_NAME: self._get_mix_values_func()}
            if computation_to_trace.requires_mix_values_func()
            else {}
        )

        traced_computation = computation_to_trace(
            (x.output for x in sanitized_inputs),
            **additional_parameters,
        )

        output_tracers = tuple(
            self.__class__(sanitized_inputs, traced_computation, output_index)
            for output_index in range(len(traced_computation.outputs))
        )

        return output_tracers

    def __add__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            ir.Add,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    # With that is that x + 1 and 1 + x have the same graph. If we want to keep
    # the order, we need to do as in __rsub__, ie mostly a copy of __sub__ +
    # some changes
    __radd__ = __add__

    def __sub__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            ir.Sub,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    def __rsub__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [other, self],
            ir.Sub,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    def __mul__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            ir.Mul,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    # With that is that x * 3 and 3 * x have the same graph. If we want to keep
    # the order, we need to do as in __rmul__, ie mostly a copy of __mul__ +
    # some changes
    __rmul__ = __mul__
