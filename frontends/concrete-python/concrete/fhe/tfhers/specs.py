"""TFHE-rs client specs."""

from typing import Any, Optional

from ..representation import Graph
from .dtypes import TFHERSIntegerType


class TFHERSClientSpecs:
    """TFHE-rs client specs.

    Contains info about TFHE-rs inputs and outputs.

    input_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):
        maps every input to a type for every function in the module. None means a non-tfhers type
    output_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):
        maps every output to a type for every function in the module. None means a non-tfhers type
    input_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):
        maps every input to a shape for every function in the module. None means a non-tfhers type
    output_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):
        maps every output to a shape for every function in the module. None means a non-tfhers type
    """

    input_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]]
    output_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]]
    input_shapes_per_func: dict[str, list[Optional[tuple[int, ...]]]]
    output_shapes_per_func: dict[str, list[Optional[tuple[int, ...]]]]

    def __init__(
        self,
        input_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]],
        output_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]],
        input_shapes_per_func: dict[str, list[Optional[tuple[int, ...]]]],
        output_shapes_per_func: dict[str, list[Optional[tuple[int, ...]]]],
    ):
        self.input_types_per_func = input_types_per_func
        self.output_types_per_func = output_types_per_func
        self.input_shapes_per_func = input_shapes_per_func
        self.output_shapes_per_func = output_shapes_per_func

    def __eq__(self, other: Any) -> bool:
        return (
            self.input_types_per_func == other.input_types_per_func
            and self.output_types_per_func == other.output_types_per_func
            and self.input_shapes_per_func == other.input_shapes_per_func
            and self.output_shapes_per_func == other.output_shapes_per_func
        )

    @staticmethod
    def from_graphs(graphs: dict[str, Graph]) -> "TFHERSClientSpecs":
        """Create a TFHERSClientSpecs instance from a dictionary of graphs.

        Args:
            graphs (Dict[str, Graph]): graphs to extract the specs from
        Returns:
            TFHERSClientSpecs: An instance of TFHERSClientSpecs containing the input
                and output types and shapes for each function.
        """

        input_types_per_func = {}
        output_types_per_func = {}
        input_shapes_per_func = {}
        output_shapes_per_func = {}

        for func_name, graph in graphs.items():
            input_types: list[Optional[TFHERSIntegerType]] = []
            input_shapes: list[Optional[tuple[int, ...]]] = []
            for input_node in graph.ordered_inputs():
                if isinstance(input_node.output.dtype, TFHERSIntegerType):
                    input_types.append(input_node.output.dtype)
                    input_shapes.append(input_node.output.shape)
                else:
                    input_types.append(None)
                    input_shapes.append(None)
            input_types_per_func[func_name] = input_types
            input_shapes_per_func[func_name] = input_shapes

            output_types: list[Optional[TFHERSIntegerType]] = []
            output_shapes: list[Optional[tuple[int, ...]]] = []
            for output_node in graph.ordered_outputs():
                if isinstance(output_node.output.dtype, TFHERSIntegerType):
                    output_types.append(output_node.output.dtype)
                    output_shapes.append(output_node.output.shape)
                else:  # pragma: no cover
                    output_types.append(None)
                    output_shapes.append(None)
            output_types_per_func[func_name] = output_types
            output_shapes_per_func[func_name] = output_shapes

        return TFHERSClientSpecs(
            input_types_per_func,
            output_types_per_func,
            input_shapes_per_func,
            output_shapes_per_func,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the TFHERSClientSpecs object to a dictionary representation.

        Returns:
            Dict[str, Any]: dictionary representation
        """

        return {
            "input_types_per_func": {
                func: [t.to_dict() if isinstance(t, TFHERSIntegerType) else None for t in types]
                for func, types in self.input_types_per_func.items()
            },
            "output_types_per_func": {
                func: [t.to_dict() if isinstance(t, TFHERSIntegerType) else None for t in types]
                for func, types in self.output_types_per_func.items()
            },
            "input_shapes_per_func": self.input_shapes_per_func,
            "output_shapes_per_func": self.output_shapes_per_func,
        }

    @staticmethod
    def from_dict(dict_obj: dict[str, Any]) -> "TFHERSClientSpecs":
        """Create a TFHERSClientSpecs instance from a dictionary.

        Args:
            dict_obj (Dict[str, Any]): A dictionary containing the specifications.

        Returns:
            TFHERSClientSpecs: An instance of TFHERSClientSpecs created from the dictionary.
        """

        input_types_per_func = {
            func: [TFHERSIntegerType.from_dict(t) if t is not None else None for t in types]
            for func, types in dict_obj["input_types_per_func"].items()
        }
        output_types_per_func = {
            func: [TFHERSIntegerType.from_dict(t) if t is not None else None for t in types]
            for func, types in dict_obj["output_types_per_func"].items()
        }
        input_shapes_per_func = {
            func: [tuple(shape) if shape is not None else None for shape in shapes]
            for func, shapes in dict_obj["input_shapes_per_func"].items()
        }
        output_shapes_per_func = {
            func: [tuple(shape) if shape is not None else None for shape in shapes]
            for func, shapes in dict_obj["output_shapes_per_func"].items()
        }
        return TFHERSClientSpecs(
            input_types_per_func,
            output_types_per_func,
            input_shapes_per_func,
            output_shapes_per_func,
        )
