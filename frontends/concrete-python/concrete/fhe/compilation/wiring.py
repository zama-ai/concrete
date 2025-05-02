"""
Declaration of wiring related class.
"""

# pylint: disable=import-error,no-name-in-module

from collections.abc import Iterable
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol, runtime_checkable

from ..representation import Graph
from .composition import CompositionClause, CompositionRule

if TYPE_CHECKING:
    from .module_compiler import FunctionDef  # pragma: no cover


class NotComposable:
    """
    Composition policy that does not allow the forwarding of any output to any input.
    """

    def get_rules_iter(self, _funcs: list["FunctionDef"]) -> Iterable[CompositionRule]:
        """
        Return an iterator over composition rules.
        """
        return []  # pragma: no cover


class AllComposable:
    """
    Composition policy that allows to forward any output of the module to any of its input.
    """

    def get_rules_iter(self, funcs: list[Graph]) -> Iterable[CompositionRule]:
        """
        Return an iterator over composition rules.
        """
        outputs = []
        for f in funcs:
            for pos, node in f.output_nodes.items():
                if node.output.is_encrypted:
                    outputs.append(CompositionClause.create((f.name, pos)))
        inputs = []
        for f in funcs:
            for pos, node in f.input_nodes.items():
                if node.output.is_encrypted:
                    inputs.append(CompositionClause.create((f.name, pos)))

        return map(CompositionRule.create, product(outputs, inputs))


@runtime_checkable
class WireOutput(Protocol):
    """
    A protocol for wire outputs.
    """

    def get_outputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible outputs of the wire output.
        """


@runtime_checkable
class WireInput(Protocol):
    """
    A protocol for wire inputs.
    """

    def get_inputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible inputs of the wire input.
        """


class Output(NamedTuple):
    """
    The output of a given function of a module.
    """

    func: "FunctionDef"
    pos: int

    def get_outputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible outputs of the wire output.
        """
        return [CompositionClause(self.func.name, self.pos)]


class AllOutputs(NamedTuple):
    """
    All the encrypted outputs of a given function of a module.
    """

    func: "FunctionDef"

    def get_outputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible outputs of the wire output.
        """
        assert self.func.graph  # pragma: no cover
        # No need to filter since only encrypted outputs are valid.
        return map(  # pragma: no cover
            CompositionClause.create,
            zip(repeat(self.func.name), range(self.func.graph.outputs_count)),
        )


class Input(NamedTuple):
    """
    The input of a given function of a module.
    """

    func: "FunctionDef"
    pos: int

    def get_inputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible inputs of the wire input.
        """
        return [CompositionClause(self.func.name, self.pos)]


class AllInputs(NamedTuple):
    """
    All the encrypted inputs of a given function of a module.
    """

    func: "FunctionDef"

    def get_inputs_iter(self) -> Iterable[CompositionClause]:
        """
        Return an iterator over the possible inputs of the wire input.
        """
        assert self.func.graph  # pragma: no cover
        output = []
        for i in range(self.func.graph.inputs_count):
            if self.func.graph.input_nodes[i].output.is_encrypted:
                output.append(CompositionClause.create((self.func.name, i)))
        return output


class Wire(NamedTuple):
    """
    A forwarding rule between an output and an input.
    """

    output: WireOutput
    input: WireInput

    def get_rules_iter(self, _) -> Iterable[CompositionRule]:
        """
        Return an iterator over composition rules.
        """
        return map(
            CompositionRule.create,
            product(self.output.get_outputs_iter(), self.input.get_inputs_iter()),
        )


class Wired:
    """
    Composition policy which allows the forwarding of certain outputs to certain inputs.
    """

    wires: set[Wire]

    def __init__(self, wires: Optional[set[Wire]] = None):
        self.wires = wires if wires else set()

    def get_rules_iter(self, funcs: list[Graph]) -> Iterable[CompositionRule]:
        """
        Return an iterator over composition rules.
        """
        funcsd = {f.name: f for f in funcs}
        rules = list(chain(*[w.get_rules_iter(funcs) for w in self.wires]))

        # We check that the given rules are legit (they concern only encrypted values)
        for rule in rules:
            if (
                not funcsd[rule.from_.func].output_nodes[rule.from_.pos].output.is_encrypted
            ):  # pragma: no cover
                message = f"Invalid composition rule encountered: \
Output {rule.from_.pos} of {rule.from_.func} is not encrypted"
                raise RuntimeError(message)
            if not funcsd[rule.to.func].input_nodes[rule.to.pos].output.is_encrypted:
                message = f"Invalid composition rule encountered: \
Input {rule.from_.pos} of {rule.from_.func} is not encrypted"
                raise RuntimeError(message)

        return rules


class TracedOutput(NamedTuple):
    """
    A wrapper type used to trace wiring.

    Allows to tag an output value coming from an other module function, and binds it with
    information about its origin.
    """

    output_info: Output
    returned_value: Any


class WireTracingContextManager:
    """
    A context manager returned by the `wire_pipeline` method.

    Activates wire tracing and yields an inputset that can be iterated on for tracing.
    """

    def __init__(self, module, inputset):
        self.module = module
        self.inputset = inputset

    def __enter__(self):
        for func in self.module.functions.values():
            func._trace_wires = self.module.composition.wires
        return self.inputset

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        for func in self.module.functions.values():
            func._trace_wires = None
