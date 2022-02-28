"""numpy tracing utilities."""
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy
from numpy.typing import DTypeLike

from ..common.data_types.dtypes_helpers import mix_values_determine_holding_dtype
from ..common.debugging.custom_assert import assert_true
from ..common.operator_graph import OPGraph
from ..common.representation.intermediate import Constant, Dot, GenericFunction, MatMul
from ..common.tracing import BaseTracer, make_input_tracers, prepare_function_parameters
from ..common.tracing.tracing_helpers import tracing_context
from ..common.values import BaseValue, TensorValue
from .np_dtypes_helpers import (
    SUPPORTED_NUMPY_DTYPES_CLASS_TYPES,
    convert_numpy_dtype_to_base_data_type,
    get_base_value_for_numpy_or_python_constant_data,
    get_numpy_function_output_dtype_and_shape_from_input_tracers,
)
from .np_indexing_helpers import process_indexing_element

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
            (tracing_func in [NPTracer.numpy_sum, NPTracer.numpy_concatenate]) or len(kwargs) == 0,
            f"**kwargs are currently not supported for numpy functions, func: {func}",
        )

        # Fixme: Special case to be removed once #772 is done
        if func is not numpy.reshape:
            sanitized_args = [self._sanitize(arg) for arg in args]
        else:
            # In numpy.reshape, the second argument is the new shape
            sanitized_args = [self._sanitize(args[0]), args[1]]
            return tracing_func(self, sanitized_args[0], sanitized_args[1], **kwargs)

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
        generic_function_output_value = deepcopy(self.output)
        generic_function_output_value.dtype = output_dtype
        traced_computation = GenericFunction(
            inputs=[self.output],
            arbitrary_func=lambda x, dtype: x.astype(dtype),
            output_value=generic_function_output_value,
            op_kind="TLU",
            op_kwargs={"dtype": normalized_numpy_dtype.type},
            op_name="astype",
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

        # numpy.invert is not great in term of types it supports, so we've decided not to support it
        # and to propose to the user to use numpy.bitwise_not
        if func == numpy.invert:
            raise RuntimeError(
                f"NPTracer does not manage the following func: {func.__name__}. Please replace by "
                f"calls to bitwise_xor with appropriate mask"
            )

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

    @classmethod
    def _np_operator(
        cls,
        numpy_operator,
        numpy_operator_string,
        numpy_operator_nin,
        *input_tracers: "NPTracer",
        **kwargs,
    ) -> "NPTracer":
        """Trace a numpy operator.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true(len(input_tracers) == numpy_operator_nin)

        common_output_dtypes_and_shapes = (
            get_numpy_function_output_dtype_and_shape_from_input_tracers(
                numpy_operator,
                *input_tracers,
            )
        )
        assert_true(len(common_output_dtypes_and_shapes) == 1)

        variable_input_indices = [
            idx
            for idx, pred in enumerate(input_tracers)
            if not isinstance(pred.traced_computation, Constant)
        ]
        assert_true(
            (non_constant_pred_count := len(variable_input_indices)) == 1,
            f"Can only have 1 non constant predecessor in {cls._np_operator.__name__}, "
            f"got {non_constant_pred_count} for operator {numpy_operator}",
        )

        variable_input_idx = variable_input_indices[0]
        output_dtype, output_shape = common_output_dtypes_and_shapes[0]

        generic_function_output_value = TensorValue(
            output_dtype,
            input_tracers[variable_input_idx].output.is_encrypted,
            output_shape,
        )

        op_kwargs = deepcopy(kwargs)

        traced_computation = GenericFunction(
            inputs=[input_tracer.output for input_tracer in input_tracers],
            arbitrary_func=numpy_operator,
            output_value=generic_function_output_value,
            op_kind="TLU",
            op_kwargs=op_kwargs,
            op_name=numpy_operator_string,
        )
        output_tracer = cls(
            input_tracers,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def numpy_dot(self, *args: "NPTracer", **_kwargs) -> "NPTracer":
        """Trace numpy.dot.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true((num_args := len(args)) == 2, f"dot expects 2 inputs got {num_args}")

        common_output_dtypes_and_shapes = (
            get_numpy_function_output_dtype_and_shape_from_input_tracers(numpy.dot, *args)
        )
        assert_true(len(common_output_dtypes_and_shapes) == 1)

        traced_computation = Dot(
            [input_tracer.output for input_tracer in args],
            common_output_dtypes_and_shapes[0][0],
            delegate_evaluation_function=numpy.dot,
        )

        output_tracer = self.__class__(
            args,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def clip(self, *args: Union["NPTracer", Any], **kwargs) -> "NPTracer":
        """Trace x.clip.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        sanitized_args = [cast(NPTracer, self._sanitize(arg)) for arg in args]
        return self.numpy_clip(self, *sanitized_args, **kwargs)

    def numpy_clip(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace numpy.clip.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self._np_operator(numpy.clip, "clip", 3, *args, **kwargs)

    def dot(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace x.dot.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert len(args) == 1
        arg0 = self._sanitize(args[0])
        assert_true(isinstance(arg0, NPTracer))
        arg0 = cast(NPTracer, arg0)
        return self.numpy_dot(self, arg0, **kwargs)

    def transpose(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace x.transpose.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self.numpy_transpose(self, *args, **kwargs)

    def numpy_transpose(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace numpy.transpose.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true((num_args := len(args)) == 1, f"transpose expect 1 input got {num_args}")

        first_arg_output = args[0].output
        assert_true(isinstance(first_arg_output, TensorValue))
        first_arg_output = cast(TensorValue, first_arg_output)

        transpose_is_fusable = first_arg_output.is_scalar or first_arg_output.ndim == 1

        out_dtype = first_arg_output.dtype
        out_shape = first_arg_output.shape[::-1]

        generic_function_output_value = TensorValue(
            out_dtype,
            first_arg_output.is_encrypted,
            out_shape,
        )

        traced_computation = GenericFunction(
            inputs=[first_arg_output],
            arbitrary_func=numpy.transpose,
            output_value=generic_function_output_value,
            op_kind="Memory",
            op_kwargs=deepcopy(kwargs),
            op_name="transpose",
            op_attributes={"fusable": transpose_is_fusable},
        )
        output_tracer = self.__class__(
            args,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def ravel(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace x.ravel.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self.numpy_ravel(self, *args, **kwargs)

    def numpy_ravel(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace numpy.ravel.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true((num_args := len(args)) == 1, f"ravel expect 1 input got {num_args}")

        first_arg_output = args[0].output
        assert_true(isinstance(first_arg_output, TensorValue))
        first_arg_output = cast(TensorValue, first_arg_output)

        ravel_is_fusable = first_arg_output.ndim == 1

        out_dtype = first_arg_output.dtype
        out_shape = (1,) if first_arg_output.is_scalar else (numpy.product(first_arg_output.shape),)

        generic_function_output_value = TensorValue(
            out_dtype,
            first_arg_output.is_encrypted,
            out_shape,
        )

        traced_computation = GenericFunction(
            inputs=[first_arg_output],
            arbitrary_func=numpy.ravel,
            output_value=generic_function_output_value,
            op_kind="Memory",
            op_kwargs=deepcopy(kwargs),
            op_name="ravel",
            op_attributes={"fusable": ravel_is_fusable},
        )
        output_tracer = self.__class__(
            args,
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def reshape(self, newshape: Tuple[Any, ...], **kwargs) -> "NPTracer":
        """Trace x.reshape.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        return self.numpy_reshape(self, newshape, **kwargs)

    def numpy_reshape(self, arg0: "NPTracer", arg1: Tuple[Any, ...], **kwargs) -> "NPTracer":
        """Trace numpy.reshape.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """

        # FIXME: #772, restore reshape(self, *args, **kwargs) signature when possible, with mypy
        # types

        # FIXME: #772, restore
        # assert_true((num_args := len(args)) == 2, f"reshape expect 2 input got {num_args}")
        # when possible

        assert_true((num_kwargs := len(kwargs)) == 0, f"reshape expect 0 kwargs got {num_kwargs}")

        first_arg_output = arg0.output
        assert_true(isinstance(first_arg_output, TensorValue))
        first_arg_output = cast(TensorValue, first_arg_output)

        try:
            # calculate a newshape using numpy to handle edge cases such as `-1`s within new shape
            newshape = numpy.zeros(first_arg_output.shape).reshape(arg1).shape
        except Exception as error:
            raise ValueError(
                f"shapes are not compatible (old shape {first_arg_output.shape}, new shape {arg1})"
            ) from error

        reshape_is_fusable = newshape == first_arg_output.shape

        out_dtype = first_arg_output.dtype
        out_shape = newshape

        generic_function_output_value = TensorValue(
            out_dtype,
            first_arg_output.is_encrypted,
            out_shape,
        )

        traced_computation = GenericFunction(
            inputs=[first_arg_output],
            arbitrary_func=numpy.reshape,
            output_value=generic_function_output_value,
            op_kind="Memory",
            op_kwargs={"newshape": newshape},
            op_name="reshape",
            op_attributes={"fusable": reshape_is_fusable},
        )
        output_tracer = self.__class__(
            [arg0],
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def flatten(self, *args: "NPTracer", **kwargs) -> "NPTracer":
        """Trace x.flatten.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """
        assert_true((num_args := len(args)) == 0, f"flatten expect 0 input got {num_args}")

        first_arg_output = self.output
        assert_true(isinstance(first_arg_output, TensorValue))
        first_arg_output = cast(TensorValue, first_arg_output)

        flatten_is_fusable = first_arg_output.ndim == 1

        out_dtype = first_arg_output.dtype
        out_shape = (1,) if first_arg_output.is_scalar else (numpy.product(first_arg_output.shape),)

        generic_function_output_value = TensorValue(
            out_dtype,
            first_arg_output.is_encrypted,
            out_shape,
        )

        traced_computation = GenericFunction(
            inputs=[first_arg_output],
            arbitrary_func=lambda x: x.flatten(),
            output_value=generic_function_output_value,
            op_kind="Memory",
            op_kwargs=deepcopy(kwargs),
            op_name="flatten",
            op_attributes={"fusable": flatten_is_fusable},
        )
        output_tracer = self.__class__(
            [self],
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def numpy_sum(self, inp: "NPTracer", **kwargs) -> "NPTracer":
        """Trace numpy.sum.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """

        input_value = inp.output

        def supported(value):
            if not value.is_encrypted or not isinstance(input_value, TensorValue):
                return False

            value = cast(TensorValue, value)
            if value.shape == ():
                return False

            return True

        if not supported(input_value):
            raise ValueError(
                f"only encrypted tensor sum is supported but you tried to sum {input_value}"
            )

        try:
            # calculate a newshape using numpy to handle all cases
            newshape = numpy.sum(numpy.zeros(input_value.shape), **kwargs).shape  # type: ignore
        except Exception as error:
            raise ValueError(
                f"invalid sum on {input_value} with "
                f"{', '.join('='.join([key, str(value)]) for key, value in kwargs.items())}"
            ) from error

        output_value = TensorValue(
            input_value.dtype,
            input_value.is_encrypted,
            newshape,
        )
        traced_computation = GenericFunction(
            inputs=[input_value],
            arbitrary_func=numpy.sum,
            output_value=output_value,
            op_kind="Memory",
            op_kwargs=kwargs,
            op_name="sum",
            op_attributes={"fusable": False},
        )
        output_tracer = self.__class__(
            [inp],
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def numpy_concatenate(self, inputs: Tuple["NPTracer", ...], **kwargs) -> "NPTracer":
        """Trace numpy.concatenate.

        Returns:
            NPTracer: The output NPTracer containing the traced function
        """

        input_values = [tracer.output for tracer in inputs]

        def supported(values):
            if any(
                not value.is_encrypted or not isinstance(value, TensorValue) for value in values
            ):
                return False

            values = [cast(TensorValue, value) for value in values]
            if any(value.shape == () for value in values):
                return False

            return True

        if not supported(input_values):
            raise ValueError(
                f"only encrypted tensor concatenation is supported "
                f"but you tried to concatenate "
                f"{', '.join(str(input_value) for input_value in input_values)}"
            )

        input_tensor_values = [cast(TensorValue, value) for value in input_values]

        try:
            # calculate a newshape using numpy to handle all cases
            sample = tuple(numpy.zeros(input_value.shape) for input_value in input_tensor_values)
            newshape = numpy.concatenate(sample, **kwargs).shape
        except Exception as error:
            kwarg_info = ""
            if len(kwargs) != 0:
                kwarg_info += " with "
                kwarg_info += ", ".join(
                    "=".join([key, str(value)]) for key, value in kwargs.items()
                )

            raise ValueError(
                f"invalid concatenation of "
                f"{', '.join(str(input_value) for input_value in input_values)}{kwarg_info}"
            ) from error

        output_value = TensorValue(
            input_tensor_values[0].dtype,
            input_tensor_values[0].is_encrypted,
            newshape,
        )
        traced_computation = GenericFunction(
            inputs=input_values,
            arbitrary_func=numpy.concatenate,
            output_value=output_value,
            op_kind="Memory",
            op_kwargs=kwargs,
            op_name="concat",
            op_attributes={"fusable": False},
        )
        output_tracer = self.__class__(
            list(inputs),
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = tuple(process_indexing_element(indexing_element) for indexing_element in item)
        else:
            item = process_indexing_element(item)

        return BaseTracer.__getitem__(self, item)

    def __matmul__(self, other):
        """Trace numpy.matmul."""
        return self.__array_ufunc__(numpy.matmul, "__call__", self, other)

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
    #
    # numpy.invert (as known as numpy.bitwise_not) is not here, because it has strange input type.
    # We ask the user to replace bitwise_xor instead
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
        numpy.dot: numpy_dot,
        numpy.transpose: numpy_transpose,
        numpy.reshape: numpy_reshape,
        numpy.ravel: numpy_ravel,
        numpy.clip: numpy_clip,
        numpy.sum: numpy_sum,
        numpy.concatenate: numpy_concatenate,
    }


def _get_unary_fun(function: numpy.ufunc):
    """Wrap _unary_operator in a lambda to populate NPTRACER.UFUNC_ROUTING."""

    # We have to access this method to be able to build NPTracer.UFUNC_ROUTING
    # dynamically
    # pylint: disable=protected-access
    return lambda *input_tracers, **kwargs: NPTracer._np_operator(
        function, f"{function.__name__}", 1, *input_tracers, **kwargs
    )
    # pylint: enable=protected-access


def _get_binary_fun(function: numpy.ufunc):
    """Wrap _binary_operator in a lambda to populate NPTRACER.UFUNC_ROUTING."""

    # We have to access this method to be able to build NPTracer.UFUNC_ROUTING
    # dynamically
    # pylint: disable=protected-access
    return lambda *input_tracers, **kwargs: NPTracer._np_operator(
        function, f"{function.__name__}", 2, *input_tracers, **kwargs
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


def _on_numpy_matmul(lhs: NPTracer, rhs: NPTracer):
    common_output_dtypes_and_shapes = get_numpy_function_output_dtype_and_shape_from_input_tracers(
        numpy.matmul, lhs, rhs
    )
    assert_true(len(common_output_dtypes_and_shapes) == 1)

    output_shape = common_output_dtypes_and_shapes[0][1]
    traced_computation = MatMul(
        [lhs.output, rhs.output],
        common_output_dtypes_and_shapes[0][0],
        output_shape,
    )
    return NPTracer([lhs, rhs], traced_computation, output_idx=0)


NPTracer.UFUNC_ROUTING[numpy.add] = _on_numpy_add
NPTracer.UFUNC_ROUTING[numpy.subtract] = _on_numpy_subtract
NPTracer.UFUNC_ROUTING[numpy.multiply] = _on_numpy_multiply
NPTracer.UFUNC_ROUTING[numpy.matmul] = _on_numpy_matmul


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
    with tracing_context([NPTracer]):
        output_tracers = function_to_trace(**input_tracers)

    if isinstance(output_tracers, NPTracer):
        output_tracers = (output_tracers,)

    op_graph = OPGraph.from_output_tracers(output_tracers)

    return op_graph
