"""hnumpy tracing utilities."""
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.dtypes_helpers import mix_values_determine_holding_dtype
from ..common.operator_graph import OPGraph
from ..common.representation.intermediate import ArbitraryFunction, Constant, Dot
from ..common.tracing import BaseTracer, make_input_tracers, prepare_function_parameters
from ..common.values import BaseValue
from .np_dtypes_helpers import (
    SUPPORTED_NUMPY_DTYPES_CLASS_TYPES,
    convert_numpy_dtype_to_base_data_type,
    get_base_value_for_numpy_or_python_constant_data,
    get_numpy_function_output_dtype,
)

SUPPORTED_TYPES_FOR_TRACING = (int, float, numpy.ndarray) + tuple(
    SUPPORTED_NUMPY_DTYPES_CLASS_TYPES
)

NPConstant = partial(
    Constant,
    get_base_value_for_data_func=get_base_value_for_numpy_or_python_constant_data,
)


class NPTracer(BaseTracer):
    """Tracer class for numpy operations."""

    _mix_values_func: Callable[..., BaseValue] = mix_values_determine_holding_dtype

    def __array_ufunc__(self, ufunc, method, *input_tracers, **kwargs):
        """Catch calls to numpy ufunc and routes them to tracing functions if supported.

        Read more: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        """
        if method == "__call__":
            tracing_func = self.get_tracing_func_for_np_function(ufunc)
            assert (
                len(kwargs) == 0
            ), f"hnumpy does not support **kwargs currently for numpy ufuncs, ufunc: {ufunc}"
            return tracing_func(self, *input_tracers, **kwargs)
        raise NotImplementedError("Only __call__ method is supported currently")

    def __array_function__(self, func, _types, args, kwargs):
        """Catch calls to numpy function in routes them to hnp functions if supported.

        Read more: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        """
        tracing_func = self.get_tracing_func_for_np_function(func)
        assert (
            len(kwargs) == 0
        ), f"hnumpy does not support **kwargs currently for numpy functions, func: {func}"
        return tracing_func(*args, **kwargs)

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
        output_dtype = convert_numpy_dtype_to_base_data_type(numpy_dtype)
        traced_computation = ArbitraryFunction(
            input_base_value=self.output,
            arbitrary_func=normalized_numpy_dtype.type,
            output_dtype=output_dtype,
            op_name=f"astype({normalized_numpy_dtype})",
        )
        output_tracer = self.__class__(
            [self], traced_computation=traced_computation, output_index=0
        )
        return output_tracer

    @staticmethod
    def get_tracing_func_for_np_function(func: Union[numpy.ufunc, Callable]) -> Callable:
        """Get the tracing function for a numpy function.

        Args:
            func (Union[numpy.ufunc, Callable]): The numpy function that will be traced

        Raises:
            NotImplementedError: Raised if the passed function is not supported by NPTracer

        Returns:
            Callable: the tracing function that needs to be called to trace func
        """
        tracing_func: Optional[Callable]
        if isinstance(func, numpy.ufunc):
            tracing_func = NPTracer.UFUNC_ROUTING.get(func, None)
        else:
            tracing_func = NPTracer.FUNC_ROUTING.get(func, None)

        if tracing_func is None:
            raise NotImplementedError(
                f"NPTracer does not yet manage the following func: {func.__name__}"
            )
        return tracing_func

    def _supports_other_operand(self, other: Any) -> bool:
        return super()._supports_other_operand(other) or isinstance(
            other, SUPPORTED_TYPES_FOR_TRACING
        )

    def _make_const_input_tracer(self, constant_data: Any) -> "NPTracer":
        return self.__class__([], NPConstant(constant_data), 0)

    @staticmethod
    def _manage_dtypes(ufunc: Union[numpy.ufunc, Callable], *input_tracers: BaseTracer):
        output_dtypes = get_numpy_function_output_dtype(
            ufunc, [input_tracer.output.data_type for input_tracer in input_tracers]
        )
        common_output_dtypes = [
            convert_numpy_dtype_to_base_data_type(dtype) for dtype in output_dtypes
        ]
        return common_output_dtypes

    def _unary_operator(
        self, unary_operator, unary_operator_string, *input_tracers: "NPTracer", **kwargs
    ) -> "NPTracer":
        """Function to trace an unary operator.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert len(input_tracers) == 1
        common_output_dtypes = self._manage_dtypes(unary_operator, *input_tracers)
        assert len(common_output_dtypes) == 1

        traced_computation = ArbitraryFunction(
            input_base_value=input_tracers[0].output,
            arbitrary_func=unary_operator,
            output_dtype=common_output_dtypes[0],
            op_kwargs=deepcopy(kwargs),
            op_name=unary_operator_string,
        )
        output_tracer = self.__class__(
            input_tracers,
            traced_computation=traced_computation,
            output_index=0,
        )
        return output_tracer

    def rint(self, *input_tracers: "NPTracer", **kwargs) -> "NPTracer":
        """Function to trace numpy.rint.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self._unary_operator(numpy.rint, "np.rint", *input_tracers, **kwargs)

    def sin(self, *input_tracers: "NPTracer", **kwargs) -> "NPTracer":
        """Function to trace numpy.sin.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self._unary_operator(numpy.sin, "np.sin", *input_tracers, **kwargs)

    def cos(self, *input_tracers: "NPTracer", **kwargs) -> "NPTracer":
        """Function to trace numpy.cos.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self._unary_operator(numpy.cos, "np.cos", *input_tracers, **kwargs)

    def tan(self, *input_tracers: "NPTracer", **kwargs) -> "NPTracer":
        """Function to trace numpy.tan.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self._unary_operator(numpy.tan, "np.tan", *input_tracers, **kwargs)

    def dot(self, other_tracer: "NPTracer", **_kwargs) -> "NPTracer":
        """Function to trace numpy.dot.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        # input_tracers contains the other tracer of the dot product
        dot_inputs = (self, self._sanitize(other_tracer))

        common_output_dtypes = self._manage_dtypes(numpy.dot, *dot_inputs)
        assert len(common_output_dtypes) == 1

        traced_computation = Dot(
            [input_tracer.output for input_tracer in dot_inputs],
            common_output_dtypes[0],
            delegate_evaluation_function=numpy.dot,
        )

        output_tracer = self.__class__(
            dot_inputs,
            traced_computation=traced_computation,
            output_index=0,
        )
        return output_tracer

    UFUNC_ROUTING: Dict[numpy.ufunc, Callable] = {
        numpy.rint: rint,
        numpy.sin: sin,
        numpy.cos: cos,
        numpy.tan: tan,
    }

    FUNC_ROUTING: Dict[Callable, Callable] = {
        numpy.dot: dot,
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
