"""This file holds the code that can be shared between tracers"""

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type

from ..data_types import BaseValue
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
        inputs: List["BaseTracer"],
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
        traced_computation = computation_to_trace(
            map(lambda x: x.output, inputs),
            op_args=op_args,
            op_kwargs=op_kwargs,
        )

        output_tracers = tuple(
            self.__class__(inputs, traced_computation, output_index)
            for output_index in range(len(traced_computation.outputs))
        )

        return output_tracers

    def __add__(self, other: "BaseTracer") -> "BaseTracer":
        result_tracer = self.instantiate_output_tracers(
            [self, other],
            ir.Add,
        )

        assert len(result_tracer) == 1
        return result_tracer[0]
