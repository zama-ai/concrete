"""hnumpy tracing utilities."""
from typing import Callable, Dict, Mapping

import numpy
from numpy.typing import DTypeLike

from ..common.data_types import BaseValue
from ..common.operator_graph import OPGraph
from ..common.representation import intermediate as ir
from ..common.tracing import BaseTracer, make_input_tracers, prepare_function_parameters
from .np_dtypes_helpers import (
    convert_numpy_dtype_to_common_dtype,
    get_ufunc_numpy_output_dtype,
)


class NPTracer(BaseTracer):
    """Tracer class for numpy operations."""

    def __array_ufunc__(self, ufunc, method, *input_tracers, **kwargs):
        """Catch calls to numpy ufunc and routes them to tracing functions if supported.

        Read more: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        """
        if method == "__call__":
            tracing_func = self.get_tracing_func_for_np_ufunc(ufunc)
            assert (
                len(kwargs) == 0
            ), f"hnumpy does not support **kwargs currently for numpy ufuncs, ufunc: {ufunc}"
            return tracing_func(self, *input_tracers, **kwargs)
        raise NotImplementedError("Only __call__ method is supported currently")

    def astype(self, numpy_dtype: DTypeLike, *args, **kwargs) -> "NPTracer":
        r"""Support numpy astype feature.

        For now it only accepts a dtype and no additional parameters, \*args and
        \*\*kwargs are accepted for interface compatibility only

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
        output_tracer = self.__class__(
            [self], traced_computation=traced_computation, output_index=0
        )
        return output_tracer

    @staticmethod
    def get_tracing_func_for_np_ufunc(ufunc: numpy.ufunc) -> Callable:
        """Get the tracing function for a numpy ufunc.

        Args:
            ufunc (numpy.ufunc): The numpy ufunc that will be traced

        Raises:
            NotImplementedError: Raised if the passed ufunc is not supported by NPTracer

        Returns:
            Callable: the tracing function that needs to be called to trace ufunc
        """
        tracing_func = NPTracer.UFUNC_ROUTING.get(ufunc, None)
        if tracing_func is None:
            raise NotImplementedError(
                f"NPTracer does not yet manage the following ufunc: {ufunc.__name__}"
            )
        return tracing_func

    @staticmethod
    def _manage_dtypes(ufunc: numpy.ufunc, *input_tracers: "NPTracer"):
        output_dtypes = get_ufunc_numpy_output_dtype(
            ufunc, [input_tracer.output.data_type for input_tracer in input_tracers]
        )
        common_output_dtypes = [
            convert_numpy_dtype_to_common_dtype(dtype) for dtype in output_dtypes
        ]
        return common_output_dtypes

    def rint(self, *input_tracers: "NPTracer", **kwargs) -> "NPTracer":
        """Function to trace numpy.rint.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert len(input_tracers) == 1
        common_output_dtypes = self._manage_dtypes(numpy.rint, *input_tracers)
        assert len(common_output_dtypes) == 1

        traced_computation = ir.ArbitraryFunction(
            input_base_value=input_tracers[0].output,
            arbitrary_func=numpy.rint,
            output_dtype=common_output_dtypes[0],
            op_kwargs=kwargs,
        )
        output_tracer = self.__class__(
            input_tracers, traced_computation=traced_computation, output_index=0
        )
        return output_tracer

    UFUNC_ROUTING: Mapping[numpy.ufunc, Callable] = {
        numpy.rint: rint,
    }


def trace_numpy_function(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> OPGraph:
    """Function used to trace a numpy function.

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

    op_graph = OPGraph.from_output_tracers(output_tracers)

    return op_graph
