"""
Declaration of `Tracer` class.
"""

import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import DTypeLike

from ..dtypes import Float, Integer
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..representation.utils import format_indexing_element
from ..values import Value


class Tracer:
    """
    Tracer class, to create computation graphs from python functions.
    """

    computation: Node
    input_tracers: List["Tracer"]
    output: Value

    # variable to control the behavior of __eq__
    # so that it can be traced but still allow
    # using Tracers in dicts when not tracing
    _is_tracing: bool = False

    @staticmethod
    def trace(function: Callable, parameters: Dict[str, Value]) -> Graph:
        """
        Trace `function` and create the `Graph` that represents it.

        Args:
            function (Callable):
                function to trace

            parameters (Dict[str, Value]):
                parameters of function to trace
                e.g. parameter x is an EncryptedScalar holding a 7-bit UnsignedInteger

        Returns:
            Graph:
                computation graph corresponding to `function`
        """

        signature = inspect.signature(function)

        missing_args = list(signature.parameters)
        for arg in parameters.keys():
            missing_args.remove(arg)
        assert_that(len(missing_args) == 0)

        arguments = {}
        input_indices = {}

        for index, param in enumerate(signature.parameters.keys()):
            node = Node.input(param, parameters[param])
            arguments[param] = Tracer(node, [])
            input_indices[node] = index

        Tracer._is_tracing = True
        output_tracers: Any = function(**arguments)
        Tracer._is_tracing = False

        if not isinstance(output_tracers, tuple):
            output_tracers = (output_tracers,)

        sanitized_tracers = []
        for tracer in output_tracers:
            if isinstance(tracer, Tracer):
                sanitized_tracers.append(tracer)
                continue

            try:
                sanitized_tracers.append(Tracer._sanitize(tracer))
            except Exception as error:
                raise ValueError(
                    f"Function '{function.__name__}' "
                    f"returned '{tracer}', "
                    f"which is not supported"
                ) from error

        output_tracers = tuple(sanitized_tracers)

        def create_graph_from_output_tracers(output_tracers: Tuple[Tracer, ...]) -> nx.MultiDiGraph:
            graph = nx.MultiDiGraph()

            visited_tracers: Set[Tracer] = set()
            current_tracers = {tracer: None for tracer in output_tracers}

            while current_tracers:
                next_tracers: Dict[Tracer, None] = {}
                for tracer in current_tracers:
                    if tracer not in visited_tracers:
                        current_node = tracer.computation
                        graph.add_node(current_node)

                        for input_idx, input_tracer in enumerate(tracer.input_tracers):
                            pred_node = input_tracer.computation

                            graph.add_node(pred_node)
                            graph.add_edge(
                                pred_node,
                                current_node,
                                input_idx=input_idx,
                            )

                            if input_tracer not in visited_tracers:
                                next_tracers.update({input_tracer: None})

                        visited_tracers.add(tracer)

                current_tracers = next_tracers

            assert_that(nx.algorithms.dag.is_directed_acyclic_graph(graph))

            unique_edges = set(
                (pred, succ, tuple((k, v) for k, v in edge_data.items()))
                for pred, succ, edge_data in graph.edges(data=True)
            )
            assert_that(len(unique_edges) == len(graph.edges))

            return graph

        graph = create_graph_from_output_tracers(output_tracers)
        input_nodes = {
            input_indices[node]: node
            for node in graph.nodes()
            if len(graph.pred[node]) == 0 and node.operation == Operation.Input
        }
        output_nodes = {
            output_idx: tracer.computation for output_idx, tracer in enumerate(output_tracers)
        }

        return Graph(graph, input_nodes, output_nodes)

    def __init__(self, computation: Node, input_tracers: List["Tracer"]):
        self.computation = computation
        self.input_tracers = input_tracers
        self.output = computation.output

    def __hash__(self) -> int:
        return id(self)

    @staticmethod
    def _sanitize(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(Tracer._sanitize(item) for item in value)

        if isinstance(value, Tracer):
            return value

        computation = Node.constant(value)
        return Tracer(computation, [])

    SUPPORTED_NUMPY_OPERATORS: Set[Any] = {
        np.abs,
        np.absolute,
        np.add,
        np.arccos,
        np.arccosh,
        np.arcsin,
        np.arcsinh,
        np.arctan,
        np.arctan2,
        np.arctanh,
        np.around,
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.cbrt,
        np.ceil,
        np.clip,
        np.concatenate,
        np.copysign,
        np.cos,
        np.cosh,
        np.deg2rad,
        np.degrees,
        np.divide,
        np.dot,
        np.equal,
        np.exp,
        np.exp2,
        np.expm1,
        np.fabs,
        np.float_power,
        np.floor,
        np.floor_divide,
        np.fmax,
        np.fmin,
        np.fmod,
        np.gcd,
        np.greater,
        np.greater_equal,
        np.heaviside,
        np.hypot,
        np.invert,
        np.isfinite,
        np.isinf,
        np.isnan,
        np.lcm,
        np.ldexp,
        np.left_shift,
        np.less,
        np.less_equal,
        np.log,
        np.log10,
        np.log1p,
        np.log2,
        np.logaddexp,
        np.logaddexp2,
        np.logical_and,
        np.logical_not,
        np.logical_or,
        np.logical_xor,
        np.matmul,
        np.maximum,
        np.minimum,
        np.mod,
        np.multiply,
        np.negative,
        np.nextafter,
        np.not_equal,
        np.ones_like,
        np.positive,
        np.power,
        np.rad2deg,
        np.radians,
        np.reciprocal,
        np.remainder,
        np.reshape,
        np.right_shift,
        np.rint,
        np.round_,
        np.sign,
        np.signbit,
        np.sin,
        np.sinh,
        np.spacing,
        np.sqrt,
        np.square,
        np.subtract,
        np.sum,
        np.tan,
        np.tanh,
        np.transpose,
        np.true_divide,
        np.trunc,
        np.where,
        np.zeros_like,
    }

    SUPPORTED_KWARGS: Dict[Any, Set[str]] = {
        np.around: {
            "decimals",
        },
        np.concatenate: {
            "axis",
        },
        np.ones_like: {
            "dtype",
        },
        np.reshape: {
            "newshape",
        },
        np.round_: {
            "decimals",
        },
        np.sum: {
            "axis",
            "keepdims",
        },
        np.zeros_like: {
            "dtype",
        },
    }

    @staticmethod
    def _trace_numpy_operation(operation: Callable, *args, **kwargs) -> "Tracer":
        """
        Trace an arbitrary numpy operation into an Operation.Generic node.

        Args:
            operation (Callable):
                operation to trace

            args (List[Any]):
                args of the arbitrary computation

            kwargs (Dict[str, Any]):
                kwargs of the arbitrary computation

        Returns:
            Tracer:
                tracer representing the arbitrary computation
        """

        if operation not in Tracer.SUPPORTED_NUMPY_OPERATORS:
            raise RuntimeError(f"Function 'np.{operation.__name__}' is not supported")

        supported_kwargs = Tracer.SUPPORTED_KWARGS.get(operation, set())
        for kwarg in kwargs:
            if kwarg not in supported_kwargs:
                raise RuntimeError(
                    f"Function 'np.{operation.__name__}' is not supported with kwarg '{kwarg}'"
                )

        if operation == np.ones_like:  # pylint: disable=comparison-with-callable
            dtype = kwargs.get("dtype", np.int64)
            return Tracer(Node.constant(np.ones(args[0].shape, dtype=dtype)), [])

        if operation == np.zeros_like:  # pylint: disable=comparison-with-callable
            dtype = kwargs.get("dtype", np.int64)
            return Tracer(Node.constant(np.zeros(args[0].shape, dtype=dtype)), [])

        def sampler(arg: Any) -> Any:
            if isinstance(arg, tuple):
                return tuple(sampler(item) for item in arg)

            output = arg.output
            assert_that(isinstance(output.dtype, (Float, Integer)))

            dtype: Any = np.int64
            if isinstance(output.dtype, Float):
                assert_that(output.dtype.bit_width in [16, 32, 64])
                dtype = {64: np.float64, 32: np.float32, 16: np.float16}[output.dtype.bit_width]

            if output.shape == ():
                return dtype(1)

            return np.ones(output.shape, dtype=dtype)

        sample = [sampler(arg) for arg in args]
        evaluation = operation(*sample, **kwargs)

        def extract_tracers(arg: Any, tracers: List[Tracer]):
            if isinstance(arg, tuple):
                for item in arg:
                    extract_tracers(item, tracers)

            if isinstance(arg, Tracer):
                tracers.append(arg)

        tracers: List[Tracer] = []
        for arg in args:
            extract_tracers(arg, tracers)

        output_value = Value.of(evaluation)
        output_value.is_encrypted = any(tracer.output.is_encrypted for tracer in tracers)

        computation = Node.generic(
            operation.__name__,
            [tracer.output for tracer in tracers],
            output_value,
            operation,
            kwargs=kwargs,
        )
        return Tracer(computation, tracers)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Numpy ufunc hook.

        (https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch)
        """

        if method == "__call__":
            sanitized_args = [self._sanitize(arg) for arg in args]
            return Tracer._trace_numpy_operation(ufunc, *sanitized_args, **kwargs)

        raise RuntimeError("Only __call__ hook is supported for numpy ufuncs")

    def __array_function__(self, func, _types, args, kwargs):
        """
        Numpy function hook.

        (https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch)
        """

        if func is np.reshape:
            sanitized_args = [self._sanitize(args[0])]
            if len(args) > 1:
                kwargs["newshape"] = args[1]
        elif func is np.transpose:
            sanitized_args = [self._sanitize(args[0])]
            if len(args) > 1:
                kwargs["axes"] = args[1]
        else:
            sanitized_args = [self._sanitize(arg) for arg in args]

        return Tracer._trace_numpy_operation(func, *sanitized_args, **kwargs)

    def __add__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.add, self, self._sanitize(other))

    def __radd__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.add, self._sanitize(other), self)

    def __sub__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.subtract, self, self._sanitize(other))

    def __rsub__(self, other) -> "Tracer":
        return Tracer._trace_numpy_operation(np.subtract, self._sanitize(other), self)

    def __mul__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.multiply, self, self._sanitize(other))

    def __rmul__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.multiply, self._sanitize(other), self)

    def __truediv__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.true_divide, self, self._sanitize(other))

    def __rtruediv__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.true_divide, self._sanitize(other), self)

    def __floordiv__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.floor_divide, self, self._sanitize(other))

    def __rfloordiv__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.floor_divide, self._sanitize(other), self)

    def __pow__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.power, self, self._sanitize(other))

    def __rpow__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.power, self._sanitize(other), self)

    def __mod__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.mod, self, self._sanitize(other))

    def __rmod__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.mod, self._sanitize(other), self)

    def __matmul__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.matmul, self, self._sanitize(other))

    def __rmatmul__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.matmul, self._sanitize(other), self)

    def __neg__(self) -> "Tracer":
        return Tracer._trace_numpy_operation(np.negative, self)

    def __pos__(self) -> "Tracer":
        return Tracer._trace_numpy_operation(np.positive, self)

    def __abs__(self):
        return Tracer._trace_numpy_operation(np.absolute, self)

    def __round__(self, ndigits=None):
        if ndigits is None:
            return Tracer._trace_numpy_operation(np.around, self).astype(np.int64)

        return Tracer._trace_numpy_operation(np.around, self, decimals=ndigits)

    def __invert__(self):
        return Tracer._trace_numpy_operation(np.invert, self)

    def __and__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_and, self, self._sanitize(other))

    def __rand__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_and, self._sanitize(other), self)

    def __or__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_or, self, self._sanitize(other))

    def __ror__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_or, self._sanitize(other), self)

    def __xor__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_xor, self, self._sanitize(other))

    def __rxor__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.bitwise_xor, self._sanitize(other), self)

    def __lshift__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.left_shift, self, self._sanitize(other))

    def __rlshift__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.left_shift, self._sanitize(other), self)

    def __rshift__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.right_shift, self, self._sanitize(other))

    def __rrshift__(self, other: Any) -> "Tracer":
        return Tracer._trace_numpy_operation(np.right_shift, self._sanitize(other), self)

    def __gt__(self, other: Any) -> "Tracer":  # type: ignore
        return Tracer._trace_numpy_operation(np.greater, self, self._sanitize(other))

    def __ge__(self, other: Any) -> "Tracer":  # type: ignore
        return Tracer._trace_numpy_operation(np.greater_equal, self, self._sanitize(other))

    def __lt__(self, other: Any) -> "Tracer":  # type: ignore
        return Tracer._trace_numpy_operation(np.less, self, self._sanitize(other))

    def __le__(self, other: Any) -> "Tracer":  # type: ignore
        return Tracer._trace_numpy_operation(np.less_equal, self, self._sanitize(other))

    def __eq__(self, other: Any) -> Union[bool, "Tracer"]:  # type: ignore
        return (
            self is other
            if not self._is_tracing
            else Tracer._trace_numpy_operation(np.equal, self, self._sanitize(other))
        )

    def __ne__(self, other: Any) -> Union[bool, "Tracer"]:  # type: ignore
        return (
            self is not other
            if not self._is_tracing
            else Tracer._trace_numpy_operation(np.not_equal, self, self._sanitize(other))
        )

    def astype(self, dtype: DTypeLike) -> "Tracer":
        """
        Trace numpy.ndarray.astype(dtype).
        """

        normalized_dtype = np.dtype(dtype)
        if np.issubdtype(normalized_dtype, np.integer) and normalized_dtype != np.int64:
            print(
                "Warning: When using `value.astype(newtype)` "
                "with an integer newtype, "
                "only use `np.int64` as the newtype "
                "to avoid unexpected overflows "
                "during inputset evaluation"
            )

        output_value = deepcopy(self.output)
        output_value.dtype = Value.of(normalized_dtype.type(0)).dtype

        computation = Node.generic(
            "astype",
            [self.output],
            output_value,
            lambda x, dtype: x.astype(dtype),
            kwargs={"dtype": normalized_dtype.type},
        )
        return Tracer(computation, [self])

    def clip(self, minimum: Any, maximum: Any) -> "Tracer":
        """
        Trace numpy.ndarray.clip().
        """

        return Tracer._trace_numpy_operation(
            np.clip, self, self._sanitize(minimum), self._sanitize(maximum)
        )

    def dot(self, other: Any) -> "Tracer":
        """
        Trace numpy.ndarray.dot().
        """

        return Tracer._trace_numpy_operation(np.dot, self, self._sanitize(other))

    def flatten(self) -> "Tracer":
        """
        Trace numpy.ndarray.flatten().
        """

        return Tracer._trace_numpy_operation(np.reshape, self, newshape=(self.output.size,))

    def reshape(self, newshape: Tuple[Any, ...]) -> "Tracer":
        """
        Trace numpy.ndarray.reshape(newshape).
        """

        return Tracer._trace_numpy_operation(np.reshape, self, newshape=newshape)

    def round(self, decimals: int = 0) -> "Tracer":
        """
        Trace numpy.ndarray.round().
        """

        return Tracer._trace_numpy_operation(np.around, self, decimals=decimals)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tracer":
        """
        Trace numpy.ndarray.transpose().
        """

        if axes is None:
            return Tracer._trace_numpy_operation(np.transpose, self)

        return Tracer._trace_numpy_operation(np.transpose, self, axes=axes)

    def __getitem__(
        self,
        index: Union[int, np.integer, slice, Tuple[Union[int, np.integer, slice], ...]],
    ) -> "Tracer":
        if not isinstance(index, tuple):
            index = (index,)

        for indexing_element in index:
            valid = isinstance(indexing_element, (int, np.integer, slice))

            if isinstance(indexing_element, slice):
                if (
                    not (
                        indexing_element.start is None
                        or isinstance(indexing_element.start, (int, np.integer))
                    )
                    or not (
                        indexing_element.stop is None
                        or isinstance(indexing_element.stop, (int, np.integer))
                    )
                    or not (
                        indexing_element.step is None
                        or isinstance(indexing_element.step, (int, np.integer))
                    )
                ):
                    valid = False

            if not valid:
                raise ValueError(
                    f"Indexing with '{format_indexing_element(indexing_element)}' is not supported"
                )

        output_value = deepcopy(self.output)
        output_value.shape = np.zeros(output_value.shape)[index].shape

        computation = Node.generic(
            "index.static",
            [self.output],
            output_value,
            lambda x: x[index],
            attributes={"index": index},
        )
        return Tracer(computation, [self])

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Trace numpy.ndarray.shape.
        """

        return self.output.shape

    @property
    def ndim(self) -> int:
        """
        Trace numpy.ndarray.ndim.
        """

        return self.output.ndim

    @property
    def size(self) -> int:
        """
        Trace numpy.ndarray.size.
        """

        return self.output.size

    @property
    def T(self) -> "Tracer":  # pylint: disable=invalid-name
        """
        Trace numpy.ndarray.T.
        """

        return Tracer._trace_numpy_operation(np.transpose, self)
