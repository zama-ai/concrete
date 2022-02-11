"""This file holds the code that can be shared between tracers."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union, cast

from ..data_types import Float
from ..data_types.base import BaseDataType
from ..debugging.custom_assert import assert_true
from ..representation.intermediate import (
    IR_MIX_VALUES_FUNC_ARG_NAME,
    Add,
    Constant,
    GenericFunction,
    IndexConstant,
    IntermediateNode,
    Mul,
    Sub,
)
from ..values import BaseValue, TensorValue


class BaseTracer(ABC):
    """Base class for implementing tracers."""

    # this variable changes the behavior of __eq__ so that it can be traced but still allows to hash
    # BaseTracers when not tracing.
    _is_tracing: bool = False

    inputs: List["BaseTracer"]
    traced_computation: IntermediateNode
    output_idx: int
    output: BaseValue
    _mix_values_func: Callable[..., BaseValue]

    def __init__(
        self,
        inputs: Iterable["BaseTracer"],
        traced_computation: IntermediateNode,
        output_idx: int,
    ) -> None:
        self.inputs = list(inputs)
        self.traced_computation = traced_computation
        self.output_idx = output_idx
        self.output = traced_computation.outputs[output_idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the output of the tracer.

        Returns:
            Tuple[int, ...]: the shape of the output
        """

        if isinstance(self.output, TensorValue):
            return self.output.shape

        raise AttributeError(
            f"'{self.__class__.__name__}' object "
            f"with '{self.output}' output "
            f"has no attribute 'shape'"
        )  # pragma: no cover

        # this error cannot be covered because we only have TensorValue for now

    @abstractmethod
    def _supports_other_operand(self, other: Any) -> bool:
        """Check if the current class supports tracing with the other operand.

        Args:
            other (Any): the operand to check compatibility with.

        Returns:
            bool: True if the tracer can manage operations with the other operand.
        """
        return isinstance(other, self.__class__)

    @abstractmethod
    def _make_const_input_tracer(self, constant_data: Any) -> "BaseTracer":
        """Create a tracer for a constant input.

        Args:
            constant_data (Any): The constant to store.

        Returns:
            BaseTracer: The BaseTracer for that constant.
        """

    @classmethod
    def set_is_tracing(cls, is_tracing: bool) -> None:
        """Set whether we are in a tracing context to change __eq__ behavior.

        Args:
            is_tracing (bool): boolean to use to set whether we are tracing
        """
        cls._is_tracing = is_tracing

    @classmethod
    def _get_mix_values_func(cls):
        return cls._mix_values_func

    def _sanitize(self, inp) -> "BaseTracer":
        if not isinstance(inp, BaseTracer) and not (
            isinstance(inp, Tuple)  # type: ignore
            and all(isinstance(item, BaseTracer) for item in inp)  # type: ignore
        ):
            return self._make_const_input_tracer(inp)
        return inp

    def instantiate_output_tracers(
        self,
        inputs: Iterable[Union["BaseTracer", Any]],
        computation_to_trace: Type[IntermediateNode],
    ) -> Tuple["BaseTracer", ...]:
        """Instantiate all output BaseTracer for a given computation.

        Args:
            inputs (Iterable[Union[BaseTracer, Any]]): Previous BaseTracer or data used as inputs
                for a new node.
            computation_to_trace (Type[IntermediateNode]): The IntermediateNode class
                to instantiate for the computation being traced

        Returns:
            Tuple[BaseTracer, ...]: A tuple containing an BaseTracer per output function
        """

        # For inputs which are actually constant, first convert into a tracer
        sanitized_inputs = [self._sanitize(inp) for inp in inputs]

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
            self.__class__(sanitized_inputs, traced_computation, output_idx)
            for output_idx in range(len(traced_computation.outputs))
        )

        return output_tracers

    def _helper_for_unary_functions(self, op_lambda: Callable, op_name: str) -> "BaseTracer":
        """Trace a unary operator which maintains the shape, which will thus be replaced by a TLU.

        Returns:
            BaseTracer: The output NPTracer containing the traced function
        """
        first_arg_output = self.output
        assert_true(isinstance(first_arg_output, TensorValue))
        first_arg_output = cast(TensorValue, first_arg_output)

        out_dtype = first_arg_output.dtype
        out_shape = first_arg_output.shape

        generic_function_output_value = TensorValue(
            out_dtype,
            first_arg_output.is_encrypted,
            out_shape,
        )

        traced_computation = GenericFunction(
            inputs=[first_arg_output],
            arbitrary_func=op_lambda,
            output_value=generic_function_output_value,
            op_kind="TLU",
            op_name=f"{op_name}",
        )
        output_tracer = self.__class__(
            [self],
            traced_computation=traced_computation,
            output_idx=0,
        )
        return output_tracer

    def _helper_for_binary_functions_with_one_cst_input(
        self,
        lhs: Union["BaseTracer", Any],
        rhs: Union["BaseTracer", Any],
        op_lambda: Callable,
        op_name: str,
        output_dtype: Optional[BaseDataType] = None,
    ) -> "BaseTracer":
        """Trace a binary operator which maintains the shape, when one input is a constant.

        This function is helpful to convert an operation with two inputs, one of which being a
        constant, into a TLU, while maintaining the constant somewhere in the graph, eg to simplify
        debugging.

        Returns:
            BaseTracer: The output NPTracer containing the traced function
        """
        if isinstance(lhs, BaseTracer):
            if not self._supports_other_operand(rhs):
                return NotImplemented
        elif isinstance(rhs, BaseTracer):
            if not self._supports_other_operand(lhs):
                return NotImplemented

        sanitized_inputs = [self._sanitize(inp) for inp in [lhs, rhs]]

        # One of the inputs has to be constant
        if not (
            isinstance(sanitized_inputs[0].traced_computation, Constant)
            or isinstance(sanitized_inputs[1].traced_computation, Constant)
        ):
            raise NotImplementedError(f"Can't manage binary operator {op_name}")

        sanitized_input_values = [san_input.output for san_input in sanitized_inputs]
        output_value = self._get_mix_values_func()(*sanitized_input_values)
        if output_dtype is not None:
            output_value.dtype = deepcopy(output_dtype)

        traced_computation = GenericFunction(
            inputs=sanitized_input_values,
            arbitrary_func=op_lambda,
            output_value=output_value,
            op_kind="TLU",
            op_name=op_name,
        )

        result_tracer = self.__class__(sanitized_inputs, traced_computation, 0)

        return result_tracer

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            Add,
        )

        assert_true(len(result_tracer) == 1)
        return result_tracer[0]

    # With that is that x + 1 and 1 + x have the same graph. If we want to keep
    # the order, we need to do as in __rsub__, ie mostly a copy of __sub__ +
    # some changes
    __radd__ = __add__

    def __neg__(self) -> "BaseTracer":
        return 0 - self

    def __pos__(self) -> "BaseTracer":
        # Remark that we don't want to return 'self' since we want the result to be a copy, ie not
        # a reference to the same object
        return 0 + self

    def _lshift(self, lhs: Union["BaseTracer", Any], rhs: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._helper_for_binary_functions_with_one_cst_input(
            lhs, rhs, lambda x, y: x << y, "lshift"
        )

    def __lshift__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x << shift
        return self._lshift(self, other)

    def __rlshift__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # cst << x
        return self._lshift(other, self)

    def _rshift(self, lhs: Union["BaseTracer", Any], rhs: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._helper_for_binary_functions_with_one_cst_input(
            lhs, rhs, lambda x, y: x >> y, "rshift"
        )

    def __rshift__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x >> shift
        return self._rshift(self, other)

    def __rrshift__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # cst >> x
        return self._rshift(other, self)

    def __gt__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x > cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x > y, "gt"
        )

    def __ge__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x >= cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x >= y, "ge"
        )

    def __lt__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x < cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x < y, "lt"
        )

    def __le__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        # x <= cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x <= y, "le"
        )

    def __eq__(self, other: Union["BaseTracer", Any]):
        # x == cst
        # Return the tracer if we are tracing, else return the result of the default __eq__ function
        # allows to have hash capabilities outside of tracing
        return (
            self._helper_for_binary_functions_with_one_cst_input(
                self, other, lambda x, y: x == y, "eq"
            )
            if self._is_tracing
            else self is other
        )

    def __ne__(self, other: Union["BaseTracer", Any]):
        # x != cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x != y, "ne"
        )

    def __pow__(self, other: Union["BaseTracer", Any]):
        # x ** cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x ** y, "pow"
        )

    def __rpow__(self, other: Union["BaseTracer", Any]):
        # cst ** x
        return self._helper_for_binary_functions_with_one_cst_input(
            other, self, lambda x, y: x ** y, "pow"
        )

    def __mod__(self, other: Union["BaseTracer", Any]):
        # x % cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x % y, "mod"
        )

    def __rmod__(self, other: Union["BaseTracer", Any]):
        # cst % x
        return self._helper_for_binary_functions_with_one_cst_input(
            other, self, lambda x, y: x % y, "mod"
        )

    def __and__(self, other: Union["BaseTracer", Any]):
        # x & cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x & y, "and"
        )

    def __rand__(self, other: Union["BaseTracer", Any]):
        # cst & x
        return self._helper_for_binary_functions_with_one_cst_input(
            other, self, lambda x, y: x & y, "and"
        )

    def __or__(self, other: Union["BaseTracer", Any]):
        # x | cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x | y, "or"
        )

    def __ror__(self, other: Union["BaseTracer", Any]):
        # cst | x
        return self._helper_for_binary_functions_with_one_cst_input(
            other, self, lambda x, y: x | y, "or"
        )

    def __xor__(self, other: Union["BaseTracer", Any]):
        # x ^ cst
        return self._helper_for_binary_functions_with_one_cst_input(
            self, other, lambda x, y: x ^ y, "xor"
        )

    def __rxor__(self, other: Union["BaseTracer", Any]):
        # cst ^ x
        return self._helper_for_binary_functions_with_one_cst_input(
            other, self, lambda x, y: x ^ y, "xor"
        )

    def __sub__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            Sub,
        )

        assert_true(len(result_tracer) == 1)
        return result_tracer[0]

    def __rsub__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [other, self],
            Sub,
        )

        assert_true(len(result_tracer) == 1)
        return result_tracer[0]

    def __mul__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        if not self._supports_other_operand(other):
            return NotImplemented

        result_tracer = self.instantiate_output_tracers(
            [self, other],
            Mul,
        )

        assert_true(len(result_tracer) == 1)
        return result_tracer[0]

    # With that is that x * 3 and 3 * x have the same graph. If we want to keep
    # the order, we need to do as in __rmul__, ie mostly a copy of __mul__ +
    # some changes
    __rmul__ = __mul__

    def __abs__(self):
        return self._helper_for_unary_functions(lambda x: x.__abs__(), "__abs__")

    def __invert__(self):
        return self._helper_for_unary_functions(lambda x: x.__invert__(), "__invert__")

    def __getitem__(self, item):
        traced_computation = IndexConstant(self.output, item)
        return self.__class__([self], traced_computation, 0)

    def _truediv(
        self, lhs: Union["BaseTracer", Any], rhs: Union["BaseTracer", Any]
    ) -> "BaseTracer":
        return self._helper_for_binary_functions_with_one_cst_input(
            lhs, rhs, lambda x, y: x / y, "truediv", Float(64)
        )

    def __truediv__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._truediv(self, other)

    def __rtruediv__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._truediv(other, self)

    def _floordiv(
        self, lhs: Union["BaseTracer", Any], rhs: Union["BaseTracer", Any]
    ) -> "BaseTracer":
        return self._helper_for_binary_functions_with_one_cst_input(
            lhs, rhs, lambda x, y: x // y, "floordiv"
        )

    def __floordiv__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._floordiv(self, other)

    def __rfloordiv__(self, other: Union["BaseTracer", Any]) -> "BaseTracer":
        return self._floordiv(other, self)
