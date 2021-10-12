"""numpy tracing utilities."""
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.dtypes_helpers import mix_values_determine_holding_dtype
from ..common.debugging.custom_assert import assert_true
from ..common.operator_graph import OPGraph
from ..common.representation.intermediate import Constant, Dot, UnivariateFunction
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

    def __array_ufunc__(self, ufunc: numpy.ufunc, method, *args, **kwargs):
        """Catch calls to numpy ufunc and routes them to tracing functions if supported.

        Read more: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        """
        if method == "__call__":
            tracing_func = self.get_tracing_func_for_np_function(ufunc)
            assert_true(
                (len(kwargs) == 0),
                f"**kwargs are currently not supported for numpy ufuncs, ufunc: {ufunc.__name__}",
            )

            # Create constant tracers for args, numpy only passes ufunc.nin args so we can
            # sanitize all of them without issues
            sanitized_args = [self._sanitize(arg) for arg in args]
            return tracing_func(*sanitized_args, **kwargs)
        raise NotImplementedError("Only __call__ method is supported currently")

    def __array_function__(self, func, _types, args, kwargs):
        """Catch calls to numpy function in routes them to tracing functions if supported.

        Read more: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        """
        tracing_func = self.get_tracing_func_for_np_function(func)
        assert_true(
            (len(kwargs) == 0),
            f"**kwargs are currently not supported for numpy functions, func: {func}",
        )
        sanitized_args = [self._sanitize(arg) for arg in args]
        return tracing_func(self, *sanitized_args, **kwargs)

    def astype(self, numpy_dtype: DTypeLike, *args, **kwargs) -> "NPTracer":
        r"""Support numpy astype feature.

        For now it only accepts a dtype and no additional parameters, \*args and
        \*\*kwargs are accepted for interface compatibility only

        Args:
            numpy_dtype (DTypeLike): The object describing a numpy type

        Returns:
            NPTracer: The NPTracer representing the casting operation
        """
        assert_true(
            len(args) == 0, f"astype currently only supports tracing without *args, got {args}"
        )
        assert_true(
            (len(kwargs) == 0),
            f"astype currently only supports tracing without **kwargs, got {kwargs}",
        )

        normalized_numpy_dtype = numpy.dtype(numpy_dtype)
        output_dtype = convert_numpy_dtype_to_base_data_type(numpy_dtype)
        traced_computation = UnivariateFunction(
            input_base_value=self.output,
            arbitrary_func=normalized_numpy_dtype.type,
            output_dtype=output_dtype,
            op_name=f"astype({normalized_numpy_dtype})",
        )
        output_tracer = self.__class__([self], traced_computation=traced_computation, output_idx=0)
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
            ufunc, [input_tracer.output.dtype for input_tracer in input_tracers]
        )
        common_output_dtypes = [
            convert_numpy_dtype_to_base_data_type(dtype) for dtype in output_dtypes
        ]
        return common_output_dtypes

    @classmethod
    def _unary_operator(
        cls, unary_operator, unary_operator_string, *input_tracers: "NPTracer", **kwargs
    ) -> "NPTracer":
        """Trace an unary operator.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true(len(input_tracers) == 1)
        common_output_dtypes = cls._manage_dtypes(unary_operator, *input_tracers)
        assert_true(len(common_output_dtypes) == 1)

        traced_computation = UnivariateFunction(
            input_base_value=input_tracers[0].output,
            arbitrary_func=unary_operator,
            output_dtype=common_output_dtypes[0],
            op_kwargs=deepcopy(kwargs),
            op_name=unary_operator_string,
        )
        output_tracer = cls(
            input_tracers,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    @classmethod
    def _binary_operator(
        cls, binary_operator, binary_operator_string, *input_tracers: "NPTracer", **kwargs
    ) -> "NPTracer":
        """Trace a binary operator, supposing one of the input is a constant.

        If no input is a constant, raises an error.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true(len(input_tracers) == 2)

        # One of the inputs has to be constant
        if isinstance(input_tracers[0].traced_computation, Constant):
            in_which_input_is_constant = 0
            baked_constant = deepcopy(input_tracers[0].traced_computation.constant_data)
        elif isinstance(input_tracers[1].traced_computation, Constant):
            in_which_input_is_constant = 1
            baked_constant = deepcopy(input_tracers[1].traced_computation.constant_data)
        else:
            raise NotImplementedError(f"Can't manage binary operator {binary_operator}")

        in_which_input_is_variable = 1 - in_which_input_is_constant

        if in_which_input_is_constant == 0:

            def arbitrary_func(x, baked_constant, **kwargs):
                return binary_operator(baked_constant, x, **kwargs)

        else:

            def arbitrary_func(x, baked_constant, **kwargs):
                return binary_operator(x, baked_constant, **kwargs)

        common_output_dtypes = cls._manage_dtypes(binary_operator, *input_tracers)
        assert_true(len(common_output_dtypes) == 1)

        op_kwargs = deepcopy(kwargs)
        op_kwargs["baked_constant"] = baked_constant
        # Store info on the operation being treated
        # Currently: the base value and type corresponding to the baked constant and which input idx
        # it was feeding
        op_attributes = {
            "baked_constant_ir_node": deepcopy(
                input_tracers[in_which_input_is_constant].traced_computation
            ),
            "in_which_input_is_constant": in_which_input_is_constant,
        }

        traced_computation = UnivariateFunction(
            input_base_value=input_tracers[in_which_input_is_variable].output,
            arbitrary_func=arbitrary_func,
            output_dtype=common_output_dtypes[0],
            op_kwargs=op_kwargs,
            op_name=binary_operator_string,
            op_attributes=op_attributes,
        )
        output_tracer = cls(
            (input_tracers[in_which_input_is_variable],),
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def dot(self, *args: "NPTracer", **_kwargs) -> "NPTracer":
        """Trace numpy.dot.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true((num_args := len(args)) == 2, f"dot expects 2 inputs got {num_args}")

        common_output_dtypes = self._manage_dtypes(numpy.dot, *args)
        assert_true(len(common_output_dtypes) == 1)

        traced_computation = Dot(
            [input_tracer.output for input_tracer in args],
            common_output_dtypes[0],
            delegate_evaluation_function=numpy.dot,
        )

        output_tracer = self.__class__(
            args,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    # Supported functions are either univariate or bivariate for which one of the two
    # sources is a constant
    #
    # numpy.add, numpy.multiply and numpy.subtract are not there since already managed
    # by leveled operations
    #
    # numpy.conjugate is not there since working on complex numbers
    #
    # numpy.isnat is not there since it is about timings
    #
    # numpy.divmod, numpy.modf and numpy.frexp are not there since output two values
    LIST_OF_SUPPORTED_UFUNC: List[numpy.ufunc] = [
        numpy.absolute,
        numpy.arccos,
        numpy.arccosh,
        numpy.arcsin,
        numpy.arcsinh,
        numpy.arctan,
        numpy.arctan2,
        numpy.arctanh,
        numpy.bitwise_and,
        numpy.bitwise_or,
        numpy.bitwise_xor,
        numpy.cbrt,
        numpy.ceil,
        numpy.copysign,
        numpy.cos,
        numpy.cosh,
        numpy.deg2rad,
        numpy.degrees,
        numpy.equal,
        numpy.exp,
        numpy.exp2,
        numpy.expm1,
        numpy.fabs,
        numpy.float_power,
        numpy.floor,
        numpy.floor_divide,
        numpy.fmax,
        numpy.fmin,
        numpy.fmod,
        numpy.gcd,
        numpy.greater,
        numpy.greater_equal,
        numpy.heaviside,
        numpy.hypot,
        numpy.invert,
        numpy.isfinite,
        numpy.isinf,
        numpy.isnan,
        numpy.lcm,
        numpy.ldexp,
        numpy.left_shift,
        numpy.less,
        numpy.less_equal,
        numpy.log,
        numpy.log10,
        numpy.log1p,
        numpy.log2,
        numpy.logaddexp,
        numpy.logaddexp2,
        numpy.logical_and,
        numpy.logical_not,
        numpy.logical_or,
        numpy.logical_xor,
        # numpy.matmul,
        numpy.maximum,
        numpy.minimum,
        numpy.negative,
        numpy.nextafter,
        numpy.not_equal,
        numpy.positive,
        numpy.power,
        numpy.rad2deg,
        numpy.radians,
        numpy.reciprocal,
        numpy.remainder,
        numpy.right_shift,
        numpy.rint,
        numpy.sign,
        numpy.signbit,
        numpy.sin,
        numpy.sinh,
        numpy.spacing,
        numpy.sqrt,
        numpy.square,
        numpy.tan,
        numpy.tanh,
        numpy.true_divide,
        numpy.trunc,
    ]

    # We build UFUNC_ROUTING dynamically after the creation of the class,
    # because of some limits of python or our unability to do it properly
    # in the class with techniques which are compatible with the different
    # coding checks we use
    UFUNC_ROUTING: Dict[numpy.ufunc, Callable] = {}

    FUNC_ROUTING: Dict[Callable, Callable] = {
        numpy.dot: dot,
    }


def _get_unary_fun(function: numpy.ufunc):
    """Wrap _unary_operator in a lambda to populate NPTRACER.UFUNC_ROUTING."""

    # We have to access this method to be able to build NPTracer.UFUNC_ROUTING
    # dynamically
    # pylint: disable=protected-access
    return lambda *input_tracers, **kwargs: NPTracer._unary_operator(
        function, f"np.{function.__name__}", *input_tracers, **kwargs
    )
    # pylint: enable=protected-access


def _get_binary_fun(function: numpy.ufunc):
    """Wrap _binary_operator in a lambda to populate NPTRACER.UFUNC_ROUTING."""

    # We have to access this method to be able to build NPTracer.UFUNC_ROUTING
    # dynamically
    # pylint: disable=protected-access
    return lambda *input_tracers, **kwargs: NPTracer._binary_operator(
        function, f"np.{function.__name__}", *input_tracers, **kwargs
    )
    # pylint: enable=protected-access


# We are populating NPTracer.UFUNC_ROUTING dynamically
NPTracer.UFUNC_ROUTING = {
    fun: _get_unary_fun(fun) for fun in NPTracer.LIST_OF_SUPPORTED_UFUNC if fun.nin == 1
}

NPTracer.UFUNC_ROUTING.update(
    {fun: _get_binary_fun(fun) for fun in NPTracer.LIST_OF_SUPPORTED_UFUNC if fun.nin == 2}
)

list_of_not_supported = [
    (ufunc.__name__, ufunc.nin)
    for ufunc in NPTracer.LIST_OF_SUPPORTED_UFUNC
    if ufunc.nin not in [1, 2]
]

assert_true(len(list_of_not_supported) == 0, f"Not supported nin's, {list_of_not_supported}")
del list_of_not_supported

# We are adding initial support for `np.array(...)` +,-,* `BaseTracer`
# (note that this is not the proper complete handling of these functions)


def _on_numpy_add(lhs, rhs):
    return lhs.__add__(rhs)


def _on_numpy_subtract(lhs, rhs):
    return lhs.__sub__(rhs)


def _on_numpy_multiply(lhs, rhs):
    return lhs.__mul__(rhs)


NPTracer.UFUNC_ROUTING[numpy.add] = _on_numpy_add
NPTracer.UFUNC_ROUTING[numpy.subtract] = _on_numpy_subtract
NPTracer.UFUNC_ROUTING[numpy.multiply] = _on_numpy_multiply


def trace_numpy_function(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> OPGraph:
    """Trace a numpy function.

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer

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
