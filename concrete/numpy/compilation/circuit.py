"""
Declaration of `Circuit` class.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from concrete.compiler import CompilerEngine

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..representation import Graph
from ..values import Value


class Circuit:
    """
    Circuit class, to combine computation graph and compiler engine into a single object.
    """

    graph: Graph
    engine: CompilerEngine

    def __init__(self, graph: Graph, engine: CompilerEngine):
        self.graph = graph
        self.engine = engine

    def __str__(self):
        return self.graph.format()

    def draw(
        self,
        show: bool = False,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
    ) -> Path:
        """
        Draw the `self.graph` and optionally save/show the drawing.

        note that this function requires the python `pygraphviz` package
        which itself requires the installation of `graphviz` packages
        see https://pygraphviz.github.io/documentation/stable/install.html

        Args:
            show (bool, default = False):
                whether to show the drawing using matplotlib or not

            horizontal (bool, default = False):
                whether to draw horizontally or not

            save_to (Optional[Path], default = None):
                path to save the drawing
                a temporary file will be used if it's None

        Returns:
            Path:
                path to the saved drawing
        """

        return self.graph.draw(show, horizontal, save_to)

    def encrypt_run_decrypt(
        self,
        *args: Union[int, np.ndarray],
    ) -> Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
        """
        Encrypt inputs, run the circuit, and decrypt the outputs in one go.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the engine

        Returns:
            Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
                result of the homomorphic evaluation
        """

        if len(args) != len(self.graph.input_nodes):
            raise ValueError(f"Expected {len(self.graph.input_nodes)} inputs but got {len(args)}")

        sanitized_args = {}

        for index, node in self.graph.input_nodes.items():
            arg = args[index]
            is_valid = isinstance(arg, (int, np.integer)) or (
                isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.integer)
            )

            expected_value = node.output

            assert_that(isinstance(expected_value.dtype, Integer))
            expected_dtype = cast(Integer, expected_value.dtype)

            if is_valid:
                expected_min = expected_dtype.min()
                expected_max = expected_dtype.max()
                expected_shape = expected_value.shape

                actual_min = arg if isinstance(arg, int) else arg.min()
                actual_max = arg if isinstance(arg, int) else arg.max()
                actual_shape = () if isinstance(arg, int) else arg.shape

                is_valid = (
                    actual_min >= expected_min
                    and actual_max <= expected_max
                    and actual_shape == expected_shape
                )

                if is_valid:
                    sanitized_args[index] = arg if isinstance(arg, int) else arg.astype(np.uint8)

            if not is_valid:
                actual_value = Value.of(arg, is_encrypted=expected_value.is_encrypted)
                raise ValueError(
                    f"Expected argument {index} to be {expected_value} but it's {actual_value}"
                )

        results = self.engine.run(*[sanitized_args[i] for i in range(len(sanitized_args))])
        if not isinstance(results, tuple):
            results = (results,)

        sanitized_results: List[Union[int, np.ndarray]] = []

        for index, node in self.graph.output_nodes.items():
            expected_value = node.output
            assert_that(isinstance(expected_value.dtype, Integer))

            expected_dtype = cast(Integer, expected_value.dtype)
            n = expected_dtype.bit_width

            result = results[index] % (2 ** n)
            if expected_dtype.is_signed:
                if isinstance(result, int):
                    sanititzed_result = result if result < (2 ** (n - 1)) else result - (2 ** n)
                    sanitized_results.append(sanititzed_result)
                else:
                    result = result.astype(np.longlong)  # to prevent overflows in numpy
                    sanititzed_result = np.where(result < (2 ** (n - 1)), result, result - (2 ** n))
                    sanitized_results.append(sanititzed_result.astype(np.int8))
            else:
                sanitized_results.append(
                    result if isinstance(result, int) else result.astype(np.uint8)
                )

        return sanitized_results[0] if len(sanitized_results) == 1 else tuple(sanitized_results)
