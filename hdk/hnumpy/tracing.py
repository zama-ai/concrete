"""hnumpy tracing utilities"""
from typing import Callable, Dict

import numpy
from numpy.typing import DTypeLike

from ..common.data_types import BaseValue
from ..common.operator_graph import OPGraph
from ..common.representation import intermediate as ir
from ..common.tracing import BaseTracer, make_input_tracers, prepare_function_parameters
from .np_dtypes_helpers import convert_numpy_dtype_to_common_dtype


class NPTracer(BaseTracer):
    """Tracer class for numpy operations"""

    def astype(self, numpy_dtype: DTypeLike, *args, **kwargs) -> "NPTracer":
        """Support numpy astype feature, for now it only accepts a dtype and no additional
            parameters, *args and **kwargs are accepted for interface compatibility only

        Args:
            numpy_dtype (DTypeLike): The object describing a numpy type

        Returns:
            NPTracer: The NPTracer representing the casting operation
        """
        assert len(args) == 0, f"astype currently only supports tracing without *args, got {args}"
        assert (
            len(kwargs) == 0
        ), f"astype currently only supports tracing without **kwargs, got {kwargs}"

        normalized_numpy_dtype = numpy.dtype(numpy_dtype)
        output_dtype = convert_numpy_dtype_to_common_dtype(numpy_dtype)
        traced_computation = ir.ArbitraryFunction(
            input_base_value=self.output,
            arbitrary_func=normalized_numpy_dtype.type,
            output_dtype=output_dtype,
        )
        output_tracer = NPTracer([self], traced_computation=traced_computation, output_index=0)
        return output_tracer


def trace_numpy_function(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> OPGraph:
    """Function used to trace a numpy function

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedValue holding a 7bits unsigned Integer

    Returns:
        OPGraph: The graph containing the ir nodes representing the computation done in the input
            function
    """
    function_parameters = prepare_function_parameters(function_to_trace, function_parameters)

    input_tracers = make_input_tracers(NPTracer, function_parameters)

    # We could easily create a graph of NPTracer, but we may end up with dead nodes starting from
    # the inputs that's why we create the graph starting from the outputs
    output_tracers = function_to_trace(**input_tracers)
    if isinstance(output_tracers, NPTracer):
        output_tracers = (output_tracers,)

    op_graph = OPGraph(output_tracers)

    return op_graph
