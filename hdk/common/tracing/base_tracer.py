"""This file holds the code that can be shared between tracers"""

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..data_types import BaseValue
from ..data_types.scalars import Scalars
from ..representation import intermediate as ir


class BaseTracer(ABC):
    """Base class for implementing tracers"""

    inputs: List["BaseTracer"]
    traced_computation: ir.IntermediateNode
    output: BaseValue

    def __init__(
        self,
        inputs: List["BaseTracer"],
        traced_computation: ir.IntermediateNode,
        output_index: int,
    ) -> None:
        self.inputs = inputs
        self.traced_computation = traced_computation
        self.output = traced_computation.outputs[output_index]

    def instantiate_output_tracers(
        self,
        inputs: List[Union["BaseTracer", Scalars]],
        computation_to_trace: Type[ir.IntermediateNode],
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["BaseTracer", ...]:
        """Helper functions to instantiate all output BaseTracer for a given computation

        Args:
            inputs (List[BaseTracer]): Previous BaseTracer used as inputs for a new node
            computation_to_trace (Type[ir.IntermediateNode]): The IntermediateNode class
                to instantiate for the computation being traced
            op_args: *args coming from the call being traced
            op_kwargs: **kwargs coming from the call being traced


        Returns:
            Tuple[BaseTracer, ...]: A tuple containing an BaseTracer per output function
        """

        # For inputs which are actually constant, first convert into a tracer
        def sanitize(inp):
            if not isinstance(inp, BaseTracer):
                return make_const_input_tracer(self.__class__, inp)
            return inp

        sanitized_inputs = [sanitize(inp) for inp in inputs]

        traced_computation = computation_to_trace(
            (x.output for x in sanitized_inputs),
            op_args=op_args,
            op_kwargs=op_kwargs,
        )

        output_tracers = tuple(
            self.__class__(sanitized_inputs, traced_computation, output_index)
            for output_index in range(len(traced_computation.outputs))
        )

        return output_tracers

    def __add__(self, other: Union["BaseTracer", Scalars]) -> "BaseTracer":

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

    def __sub__(self, other: Union["BaseTracer", Scalars]) -> "BaseTracer":

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            ir.Sub,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    def __rsub__(self, other: Union["BaseTracer", Scalars]) -> "BaseTracer":

        result_tracer = self.instantiate_output_tracers(
            [other, self],
            ir.Sub,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]

    def __mul__(self, other: Union["BaseTracer", Scalars]) -> "BaseTracer":
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


def make_const_input_tracer(tracer_class: Type[BaseTracer], constant_data: Scalars) -> BaseTracer:
    """Helper function to create a tracer for a constant input

    Args:
        tracer_class (Type[BaseTracer]): the class of tracer to create a ConstantInput for
        constant_data (Scalars): the constant

    Returns:
        BaseTracer: The BaseTracer for that constant
    """
    return tracer_class([], ir.ConstantInput(constant_data), 0)
