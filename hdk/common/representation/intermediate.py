"""File containing HDK's intermdiate representation of source programs operations"""

from abc import ABC
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..data_types import BaseValue


class IntermediateNode(ABC):
    """Abstract Base Class to derive from to represent source program operations"""

    inputs: List[BaseValue]
    outputs: List[BaseValue]
    op_args: Optional[Tuple[Any, ...]]
    op_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        inputs: Iterable[BaseValue],
        op_args: Optional[Tuple[Any, ...]] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.inputs = list(inputs)
        self.op_args = op_args
        self.op_kwargs = op_kwargs


class Add(IntermediateNode):
    """Addition between two values"""

    def __init__(
        self,
        inputs: Iterable[BaseValue],
    ) -> None:
        super().__init__(inputs)
        assert len(self.inputs) == 2

        # For now copy the first input type for the output type
        # We don't perform checks or enforce consistency here for now, so this is OK
        self.outputs = [deepcopy(self.inputs[0])]


class Input(IntermediateNode):
    """Node representing an input of the numpy program"""

    def __init__(
        self,
        inputs: Iterable[BaseValue],
    ) -> None:
        super().__init__(inputs)
        assert len(self.inputs) == 1
        self.outputs = [deepcopy(self.inputs[0])]
