"""
Declaration of `Converter` class.
"""

# pylint: disable=import-error,no-name-in-module

import math
import sys
from collections.abc import Iterable
from typing import Optional, Union

import concrete.lang
import concrete.lang.dialects.tracing
import networkx as nx
import numpy as np
from mlir.dialects import func
from mlir.ir import Context as MlirContext
from mlir.ir import InsertionPoint as MlirInsertionPoint
from mlir.ir import Location as MlirLocation
from mlir.ir import Module as MlirModule

from ..compilation.composition import CompositionRule
from ..compilation.configuration import Configuration, Exactness, ParameterSelectionStrategy
from ..representation import Graph, GraphProcessor, MultiGraphProcessor, Node, Operation
from ..tfhers.dtypes import TFHERSIntegerType
from .context import Context
from .conversion import Conversion
from .processors import *  # pylint: disable=wildcard-import
from .utils import MAXIMUM_TLU_BIT_WIDTH, construct_deduplicated_tables

# pylint: enable=import-error,no-name-in-module


class Converter:
    """
    Converter class, to convert a computation graph to MLIR.
    """

    configuration: Configuration
    composition_rules: list[CompositionRule]

    def __init__(
        self,
        configuration: Configuration,
        composition_rules: Optional[Iterable[CompositionRule]] = None,
    ):
        self.configuration = configuration
        self.composition_rules = list(composition_rules) if composition_rules else []

    def convert_many(
        self,
        graphs: dict[str, Graph],
        mlir_context: MlirContext,
    ) -> MlirModule:
        """
        Convert multiple computation graphs to an MLIR module.

        Args:
            graphs (Dict[str, Graph]):
                graphs to convert

            mlir_context (MlirContext):
                MLIR Context to use for module generation

        Return:
            MlirModule:
                In-memory MLIR module corresponding to the graph
        """
        self.process(graphs)

        with mlir_context as context, MlirLocation.unknown():
            concrete.lang.register_dialects(context)  # pylint: disable=no-member

            module = MlirModule.create()
            with MlirInsertionPoint(module.body):
                for name, graph in graphs.items():
                    # pylint: disable=cell-var-from-loop
                    # ruff: noqa: B023
                    ctx = Context(context, graph, self.configuration)

                    # if using tfhers integers, parameter selection strategy has to be
                    # multi-parameters. We try to catch this early, although the compiler
                    # will also fail without it.
                    if (
                        any(
                            isinstance(node.output.dtype, TFHERSIntegerType)
                            for node in graph.ordered_inputs()
                        )
                        and self.configuration.parameter_selection_strategy
                        != ParameterSelectionStrategy.MULTI
                    ):
                        msg = (
                            "Can't use tfhers integers with "
                            f"{self.configuration.parameter_selection_strategy} parameters. "
                            "Please use `ParameterSelectionStrategy.MULTI` as the parameter "
                            "selection strategy instead."
                        )
                        raise RuntimeError(msg)

                    input_types = [ctx.typeof(node).mlir for node in graph.ordered_inputs()]

                    location = graph.location.split(":")
                    with MlirLocation.file(
                        location[0], line=int(location[1]), col=0, context=context
                    ):

                        @func.FuncOp.from_py_func(*input_types, name=name)
                        def main(*args):
                            for index, node in enumerate(graph.ordered_inputs()):
                                conversion = Conversion(node, args[index])
                                if "original_bit_width" in node.properties:
                                    conversion.set_original_bit_width(
                                        node.properties["original_bit_width"]
                                    )
                                ctx.conversions[node] = conversion

                            ordered_nodes = [
                                node
                                for node in nx.lexicographical_topological_sort(graph.graph)
                                if node.operation != Operation.Input
                            ]

                            for progress_index, node in enumerate(ordered_nodes):
                                self.trace_progress(
                                    self.configuration, progress_index, ordered_nodes
                                )
                                preds = [
                                    ctx.conversions[pred] for pred in graph.ordered_preds_of(node)
                                ]
                                self.node(ctx, node, preds)
                            self.trace_progress(
                                self.configuration, len(ordered_nodes), ordered_nodes
                            )

                            outputs = []
                            for node in graph.ordered_outputs():
                                assert node in ctx.conversions
                                outputs.append(ctx.conversions[node].result)

                            return tuple(outputs)

        return module

    def convert(
        self,
        graph: Graph,
        mlir_context: MlirContext,
    ) -> MlirModule:
        """
        Convert a computation graph to MLIR.

        Args:
            graph (Graph):
                graph to convert

            mlir_context (MlirContext):
                MLIR Context to use for module generation


        Return:
            MlirModule:
                In-memory MLIR module corresponding to the graph
        """

        return self.convert_many({graph.name: graph}, mlir_context)

    @staticmethod
    def stdout_with_ansi_support() -> bool:
        """Detect if ansi characters can be used (e.g. not the case in notebooks)."""
        return sys.stdout.isatty()  # pragma: no cover

    @staticmethod
    def simplify_tag(configuration: Configuration, tag: str) -> str:
        """Keep only `n` higher tag parts."""
        if configuration.progress_tag is True or not tag:
            return tag
        last_dot_pos = 0
        for _ in range(configuration.progress_tag):
            last_dot_pos = tag.find(".", last_dot_pos + 1)
            if last_dot_pos == -1:
                return tag
        return tag[:last_dot_pos]

    @classmethod
    def trace_progress(cls, configuration: Configuration, progress_index: int, nodes: list[Node]):
        """
        Add a trace_message for progress.

        Args:
            configuration:
                configuration for title, tags options

            progress_index:
                index of the next node to process

            nodes:
                all nodes
        """
        if not nodes or not configuration.show_progress:
            return

        total = len(nodes)
        title = configuration.progress_title
        max_nb_steps = 50

        assert 0 <= progress_index <= total

        nb_ops_to_percent = lambda current: int(100 * current / total)
        percent = nb_ops_to_percent(progress_index)
        prev_percent = nb_ops_to_percent(progress_index - 1)
        steps_done = percent // 2
        prev_steps_done = prev_percent // 2

        step = "█"
        if not cls.stdout_with_ansi_support():
            if progress_index == 0:
                msg = f"{' ' * len(title)}{'_' * max_nb_steps}\n{title}"
            else:
                if steps_done == prev_steps_done:
                    return
                msg = step
                if percent == 100:
                    msg += " 100%\n"

        elif progress_index == 0 or percent != prev_percent:
            if configuration.progress_tag and progress_index != total:
                tag = nodes[progress_index].tag
                tag = cls.simplify_tag(configuration, tag)
                if tag:
                    tag = f" ({tag})"
            else:
                tag = ""
            cleared_line = "\033[512D\033[2K"
            full_bar = f"|{step * steps_done}{'.' * (max_nb_steps - steps_done)}|"
            msg = f"{cleared_line}{title}{percent:>3}% {full_bar} {percent:>3}%{tag}"
            if percent == 100:
                msg += "\n"
        else:
            return
        concrete.lang.dialects.tracing.TraceMessageOp(msg=msg)  # pylint: disable=no-member

    def process(self, graphs: dict[str, Graph]):
        """
        Process a computation graph for MLIR conversion.

        Args:
            graphs (Dict[str, Graph]):
                graphs to process
        """

        configuration = self.configuration
        composition_rules = self.composition_rules

        pipeline = (
            configuration.additional_pre_processors
            + [
                CheckIntegerOnly(),
                AssignBitWidths(
                    single_precision=configuration.single_precision,
                    composition_rules=composition_rules,
                    comparison_strategy_preference=configuration.comparison_strategy_preference,
                    bitwise_strategy_preference=configuration.bitwise_strategy_preference,
                    shifts_with_promotion=configuration.shifts_with_promotion,
                    multivariate_strategy_preference=configuration.multivariate_strategy_preference,
                    min_max_strategy_preference=configuration.min_max_strategy_preference,
                ),
                ProcessRounding(
                    rounding_exactness=configuration.rounding_exactness,
                ),
            ]
            + configuration.additional_post_processors
            + [
                AssignNodeIds(),
            ]
        )

        for processor in pipeline:
            assert isinstance(processor, GraphProcessor)
            if isinstance(processor, MultiGraphProcessor):
                processor.apply_many(graphs)
            else:
                for graph in graphs.values():
                    processor.apply(graph)

    def node(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        """
        Convert a computation graph node into MLIR.

        Args:
            ctx (Context):
                conversion context

            node (Node):
                node to convert

            preds (List[Conversion]):
                conversions of ordered predecessors of the node

        Return:
            Conversion:
                conversion object corresponding to node
        """

        ctx.converting = node

        assert node.operation != Operation.Input
        operation = "constant" if node.operation == Operation.Constant else node.properties["name"]
        assert operation not in ["convert", "node"]

        converter = getattr(self, operation, self.tlu)
        conversion = converter(ctx, node, preds)
        conversion.set_original_bit_width(node.properties["original_bit_width"])

        ctx.conversions[node] = conversion
        return conversion

    # The name of the remaining methods all correspond to node names.
    # And they have the same signature so that they can be called in a generic way.

    # pylint: disable=missing-function-docstring,unused-argument

    def add(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.add(ctx.typeof(node), preds[0], preds[1])

    def amax(
        self, ctx: Context, node: Node, preds: list[Conversion]
    ) -> Conversion:  # pragma: no cover
        return self.max(ctx, node, preds)

    def amin(
        self, ctx: Context, node: Node, preds: list[Conversion]
    ) -> Conversion:  # pragma: no cover
        return self.min(ctx, node, preds)

    def array(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) > 0
        return ctx.array(ctx.typeof(node), elements=preds)

    def assign_dynamic(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) >= 3

        x = preds[0]
        y = preds[-1]

        dynamic_indices = preds[1:-1]
        static_indices = node.properties["kwargs"]["static_indices"]

        indices = []

        cursor = 0
        for index in static_indices:
            if index is None:
                indices.append(dynamic_indices[cursor])
                cursor += 1
            else:
                indices.append(index)

        return ctx.assign(ctx.typeof(node), x, y, indices)

    def assign_static(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.assign(
            ctx.typeof(node),
            preds[0],
            preds[1],
            node.properties["kwargs"]["index"],
        )

    def bitwise_and(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.bitwise_and(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def bitwise_or(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.bitwise_or(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def bitwise_xor(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.bitwise_xor(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def broadcast_to(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.broadcast_to(preds[0], shape=node.output.shape)

    def concatenate(self, ctx: Context, node: Node, preds: list[Conversion]):
        return ctx.concatenate(
            ctx.typeof(node),
            preds,
            axis=node.properties["kwargs"].get("axis", 0),
        )

    def constant(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 0
        return ctx.constant(ctx.typeof(node), data=node())

    def conv1d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        ctx.error({node: "1-dimensional convolutions are not supported at the moment"})
        assert False, "unreachable"  # pragma: no cover

    def conv2d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) in [2, 3]
        return ctx.conv2d(
            ctx.typeof(node),
            preds[0],
            preds[1],
            preds[2] if len(preds) == 3 else None,
            strides=node.properties["kwargs"]["strides"],
            dilations=node.properties["kwargs"]["dilations"],
            pads=node.properties["kwargs"]["pads"],
            group=node.properties["kwargs"]["group"],
        )

    def conv3d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        ctx.error({node: "3-dimensional convolutions are not supported at the moment"})
        assert False, "unreachable"  # pragma: no cover

    def copy(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return preds[0]

    def dot(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.dot(ctx.typeof(node), preds[0], preds[1])

    def dynamic_tlu(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.dynamic_tlu(ctx.typeof(node), preds[0], preds[1])

    def equal(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.equal(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def expand_dims(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.reshape(preds[0], shape=node.output.shape)

    def extract_bit_pattern(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.extract_bits(ctx.typeof(node), preds[0], bits=node.properties["kwargs"]["bits"])

    def greater(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.greater(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def greater_equal(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.greater_equal(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def identity(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        force_noise_refresh = node.properties["kwargs"].get("force_noise_refresh", False)
        return ctx.identity(ctx.typeof(node), preds[0], force_noise_refresh)

    def index_dynamic(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) >= 2

        x = preds[0]
        dynamic_indices = preds[1:]
        static_indices = node.properties["kwargs"]["static_indices"]

        indices = []

        cursor = 0
        for index in static_indices:
            if index is None:
                indices.append(dynamic_indices[cursor])
                cursor += 1
            else:
                indices.append(index)

        return ctx.index(ctx.typeof(node), x, indices)

    def index_static(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.index(
            ctx.typeof(node),
            preds[0],
            index=node.properties["kwargs"]["index"],
        )

    def left_shift(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.shift(
                ctx.typeof(node),
                preds[0],
                preds[1],
                orientation="left",
                original_resulting_bit_width=node.properties["original_bit_width"],
            )

        return self.tlu(ctx, node, preds)

    def less(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.less(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def less_equal(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.less_equal(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def matmul(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.matmul(ctx.typeof(node), preds[0], preds[1])

    def max(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.min_max(
            ctx.typeof(node),
            preds[0],
            axes=node.properties["kwargs"].get("axis", ()),
            keep_dims=node.properties["kwargs"].get("keepdims", False),
            operation="max",
        )

    def maximum(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.maximum(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def maxpool1d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        ctx.error({node: "1-dimensional maxpooling is not supported at the moment"})
        assert False, "unreachable"  # pragma: no cover

    def maxpool2d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.maxpool2d(
            ctx.typeof(node),
            preds[0],
            kernel_shape=node.properties["kwargs"]["kernel_shape"],
            strides=node.properties["kwargs"]["strides"],
            dilations=node.properties["kwargs"]["dilations"],
        )

    def maxpool3d(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        ctx.error({node: "3-dimensional maxpooling is not supported at the moment"})
        assert False, "unreachable"  # pragma: no cover

    def min(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.min_max(
            ctx.typeof(node),
            preds[0],
            axes=node.properties["kwargs"].get("axis", ()),
            keep_dims=node.properties["kwargs"].get("keepdims", False),
            operation="min",
        )

    def minimum(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.minimum(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def multiply(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.mul(ctx.typeof(node), preds[0], preds[1])

    def negative(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.neg(ctx.typeof(node), preds[0])

    def not_equal(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.not_equal(ctx.typeof(node), preds[0], preds[1])

        return self.tlu(ctx, node, preds)

    def ones(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 0
        return ctx.ones(ctx.typeof(node))

    def relu(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.relu(ctx.typeof(node), preds[0])

    def reshape(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.reshape(preds[0], shape=node.output.shape)

    def right_shift(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2

        if all(pred.is_encrypted for pred in preds):
            return ctx.shift(
                ctx.typeof(node),
                preds[0],
                preds[1],
                orientation="right",
                original_resulting_bit_width=node.properties["original_bit_width"],
            )

        return self.tlu(ctx, node, preds)

    def round_bit_pattern(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        pred = preds[0]

        final_lsbs_to_remove = node.properties["final_lsbs_to_remove"]
        overflow_detected = node.properties["overflow_detected"]

        if pred.is_encrypted and pred.bit_width != pred.original_bit_width:
            overflow_protection = node.properties["overflow_protection"]

            shifter = 2 ** (pred.bit_width - pred.original_bit_width)
            if overflow_protection and overflow_detected:
                shifter //= 2

            if shifter != 1:
                shift_amount = int(np.log2(shifter))
                pred = ctx.reinterpret(
                    ctx.mul(
                        pred.type,
                        pred,
                        ctx.constant(ctx.i(pred.bit_width + 1), shifter),
                    ),
                    bit_width=(pred.bit_width - shift_amount),
                )
                final_lsbs_to_remove -= shift_amount

        return ctx.round_bit_pattern(
            ctx.typeof(node),
            pred,
            final_lsbs_to_remove,
            node.properties["exactness"],
            overflow_detected,
        )

    def subtract(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 2
        return ctx.sub(ctx.typeof(node), preds[0], preds[1])

    def sum(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.sum(
            ctx.typeof(node),
            preds[0],
            axes=node.properties["kwargs"].get("axis", []),
            keep_dims=node.properties["kwargs"].get("keepdims", False),
        )

    def squeeze(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        # because of the tracing logic, we have the correct output shape

        # if the output shape is (), it means (1, 1, ..., 1, 1) is squeezed
        # and the result is a scalar, so we need to do indexing, not reshape
        if node.output.shape == ():
            if preds[0].shape == ():
                return preds[0]

            assert all(size == 1 for size in preds[0].shape)
            index = (0,) * len(preds[0].shape)
            return ctx.index(ctx.typeof(node), preds[0], index)

        # otherwise, a simple reshape would work as we already have the correct shape
        return ctx.reshape(preds[0], shape=node.output.shape)

    @classmethod
    def tlu_adjust(cls, table, variable_input, target_bit_width, clipping, reduce_precision):
        target_bit_width = min(
            variable_input.bit_width, target_bit_width
        )  # inconsistency due to more precise bound vs precision
        table_bit_width = math.log2(len(table))
        assert table_bit_width.is_integer()
        table_bit_width = int(table_bit_width)
        table_has_right_size = variable_input.bit_width == table_bit_width
        if table_has_right_size and not clipping:
            return table
        half_rounded_bit_width = target_bit_width - 1
        if variable_input.is_signed:
            # upper = positive part, lower = negative part
            upper_clipping_index = 2**half_rounded_bit_width - 1
            lower_clipping_index = 2**table_bit_width - 2**half_rounded_bit_width
            positive_clipped_card = 2 ** (table_bit_width - 1) - upper_clipping_index - 1
            negative_clipped_card = 2 ** (table_bit_width - 1) - 2**half_rounded_bit_width
        else:
            upper_clipping_index = 2**target_bit_width - 1
            lower_clipping_index = 0
            positive_clipped_card = 2**table_bit_width - upper_clipping_index - 1
        lower_clipping = table[lower_clipping_index]
        upper_clipping = table[upper_clipping_index]
        if table_has_right_size:
            # value clipping
            assert clipping
            if variable_input.is_signed:
                table = (
                    list(table[: upper_clipping_index + 1])
                    + [upper_clipping] * positive_clipped_card
                    + [lower_clipping] * negative_clipped_card
                    + list(table[lower_clipping_index:])
                )
            else:
                table = (
                    list(table[lower_clipping_index : upper_clipping_index + 1])
                    + [upper_clipping] * positive_clipped_card
                )
            assert len(table) == 2**table_bit_width, (
                len(table),
                2**table_bit_width,
                table,
                upper_clipping,
                lower_clipping,
            )
            return np.array(table, dtype=np.uint64)  # negative value are in unsigned representation

        # adjust tlu size
        assert reduce_precision
        if variable_input.is_signed:
            return np.concatenate((table[: upper_clipping_index + 1], table[lower_clipping_index:]))

        return table[lower_clipping_index : upper_clipping_index + 1]

    def tlu(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert node.converted_to_table_lookup

        is_multivariate = (
            node.operation == Operation.Generic
            and node.properties["attributes"].get("is_multivariate") is True
        )

        pred_nodes = ctx.graph.ordered_preds_of(node)

        variable_input_indices = []
        for pred_index, pred_node in enumerate(pred_nodes):
            if pred_node.operation != Operation.Constant:
                variable_input_indices.append(pred_index)

        if is_multivariate:
            sum_of_bit_widths = sum(pred.original_bit_width for pred in preds)
            if sum_of_bit_widths > MAXIMUM_TLU_BIT_WIDTH:
                highlights: dict[Node, Union[str, list[str]]] = {
                    pred.origin: f"this {pred.original_bit_width}-bit value is one of the inputs"
                    for pred in preds
                }
                highlights[node] = [
                    (
                        f"which means the inputs would be packed to {sum_of_bit_widths}-bits "
                        f"for the table lookup"
                    ),
                    f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit table lookups are supported",
                ]
                ctx.error(highlights)

        else:
            assert len(variable_input_indices) == 1

            variable_input_index = variable_input_indices[0]
            variable_input = preds[variable_input_index]

            if variable_input.bit_width > MAXIMUM_TLU_BIT_WIDTH:
                variable_input_messages = [
                    f"this {variable_input.bit_width}-bit value "
                    f"is used as an input to a table lookup"
                ]
                if variable_input.bit_width != variable_input.original_bit_width:
                    variable_input_messages.append(
                        "("
                        f"note that it's assigned {variable_input.bit_width}-bits "
                        f"during compilation because of its relation with other operations"
                        ")"
                    )

                highlights = {
                    variable_input.origin: variable_input_messages,
                    node: f"but only up to {MAXIMUM_TLU_BIT_WIDTH}-bit table lookups are supported",
                }
                ctx.error(highlights)

        tables = construct_deduplicated_tables(node, pred_nodes, self.configuration)

        assert len(tables) > 0

        lut_shape: tuple[int, ...] = ()
        map_shape: tuple[int, ...] = ()

        if len(tables) == 1:
            table = tables[0][0]
            assert tables[0][1] is None

            lut_shape = (len(table),)
            lut_values = np.array(table, dtype=np.int64)

            map_shape = ()
            map_values = None
        else:
            individual_table_size = len(tables[0][0])

            lut_shape = (len(tables), individual_table_size)
            map_shape = node.output.shape

            lut_values = np.zeros(lut_shape, dtype=np.uint64)
            map_values = np.zeros(map_shape, dtype=np.intp)

            for i, (table, indices) in enumerate(tables):
                assert len(table) == individual_table_size
                lut_values[i, :] = table

                assert indices is not None
                for index in indices:
                    map_values[index] = i

        if is_multivariate:
            if len(tables) == 1:
                return ctx.multivariate_tlu(ctx.typeof(node), preds, table=lut_values.tolist())

            assert map_values is not None
            return ctx.multivariate_multi_tlu(
                ctx.typeof(node),
                xs=preds,
                tables=lut_values.tolist(),
                mapping=map_values.tolist(),
            )

        assert len(variable_input_indices) == 1

        variable_input_index = variable_input_indices[0]
        variable_input = preds[variable_input_index]

        if variable_input.origin.properties.get("name") == "truncate_bit_pattern":
            original_bit_width = variable_input.origin.properties["original_bit_width"]
            lsbs_to_remove = variable_input.origin.properties["kwargs"]["lsbs_to_remove"]
            truncated_bit_width = original_bit_width - lsbs_to_remove

            if variable_input.bit_width > original_bit_width:
                bit_width_difference = variable_input.bit_width - original_bit_width
                shifter = ctx.constant(
                    ctx.i(variable_input.bit_width + 1),
                    2**bit_width_difference,
                )
                variable_input = ctx.mul(variable_input.type, variable_input, shifter)

            variable_input = ctx.reinterpret(variable_input, bit_width=truncated_bit_width)
        elif variable_input.origin.properties.get("name") == "round_bit_pattern":
            exactness = (
                variable_input.origin.properties["exactness"]
                or ctx.configuration.rounding_exactness
            )
            if exactness is Exactness.APPROXIMATE:
                # we clip values to enforce input precision exactly as queried
                original_bit_width = variable_input.origin.properties["original_bit_width"]
                lsbs_to_remove = variable_input.origin.properties["kwargs"]["lsbs_to_remove"]
                overflow = variable_input.origin.properties["overflow_detected"]
                rounded_bit_width = original_bit_width - lsbs_to_remove - overflow
                approx_config = ctx.configuration.approximate_rounding_config
                clipping = approx_config.logical_clipping
                reduce_precision = approx_config.reduce_precision_after_approximate_clipping
                if len(tables) == 1:
                    lut_values = self.tlu_adjust(
                        lut_values,
                        variable_input,
                        rounded_bit_width,
                        clipping,
                        reduce_precision,
                    )
                else:  # pragma: no cover
                    for sub_i, sub_lut_values in enumerate(lut_values):
                        lut_values[sub_i] = self.tlu_adjust(
                            sub_lut_values,
                            variable_input,
                            rounded_bit_width,
                            clipping,
                            reduce_precision,
                        )

        if len(tables) == 1:
            return ctx.tlu(ctx.typeof(node), on=variable_input, table=lut_values.tolist())

        assert map_values is not None
        return ctx.multi_tlu(
            ctx.typeof(node),
            on=variable_input,
            tables=lut_values.tolist(),
            mapping=map_values.tolist(),
        )

    def transpose(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.transpose(
            ctx.typeof(node),
            preds[0],
            axes=node.properties["kwargs"].get("axes", []),
        )

    def truncate_bit_pattern(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        return ctx.truncate_bit_pattern(preds[0], node.properties["kwargs"]["lsbs_to_remove"])

    def where(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 3
        return ctx.where(ctx.typeof(node), *preds)

    def zeros(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 0
        return ctx.zeros(ctx.typeof(node))

    def tfhers_to_native(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        tfhers_int = preds[0]
        dtype: TFHERSIntegerType = node.properties["attributes"]["type"]
        result_bit_width, carry_width, msg_width = (
            # even if TFHE-rs value are using non-variable bit-width, we want the output
            # to be pluggable into the rest of the computation. For example, two 8bits TFHE-rs
            # integers could be used in a 9bits addition.
            ctx.typeof(node).bit_width,
            dtype.carry_width,
            dtype.msg_width,
        )

        tfhers_int = ctx.change_partition(tfhers_int, src_partition=dtype.params)

        # number of ciphertexts representing a single integer
        num_cts = tfhers_int.shape[-1]
        # first table maps to the lsb, and last one maps to the msb
        tables = [
            # we consider carry bits for all ciphertexts.
            # This means that overflow can happen, and it's undefined behavior
            [value << (msg_width * i) for value in range(2 ** (msg_width + carry_width))]
            for i in range(num_cts)
        ]
        mapping = np.broadcast_to(np.array(range(num_cts)), tfhers_int.shape)

        # intermediate type increase bit_width via TLU but keep the same shape
        interm_type = ctx.tensor(ctx.eint(result_bit_width), tfhers_int.shape)
        mapped = ctx.multi_tlu(interm_type, tfhers_int, tables, mapping)

        # sum will remove the last dim which is the dim of ciphertexts
        result_shape = tfhers_int.shape[:-1]
        # if result_shape is () then ctx.tensor would return a scalar type
        result_type = ctx.tensor(
            ctx.esint(result_bit_width) if dtype.is_signed else ctx.eint(result_bit_width),
            result_shape,
        )
        sum_result = ctx.sum(result_type, mapped, axes=-1)

        # we want to set the padding bit if the native type is signed
        # and the ciphertext is negative (sign bit set to 1)
        # we also want to extend the sign bit if we are going for a larger bitwidth
        if dtype.is_signed:
            # select MSBs of all tfhers ciphetexts
            index = [slice(0, dim_size) for dim_size in tfhers_int.shape[:-1]] + [
                -1,
            ]
            msbs = ctx.index(
                ctx.tensor(ctx.eint(msg_width + carry_width), tfhers_int.shape[:-1]),
                tfhers_int,
                index=index,
            )
            # extend sign bit up to the padding bit (carry would be considered negative)
            sign_ext_bit_table = [
                0,
            ] * 2 ** (msg_width - 1) + [
                # we fill all new msbs (including the padding bit) with the sign bit
                2**result_bit_width  # padding bit
                + (  # sign bit extension
                    2**result_bit_width - 2**dtype.bit_width
                    if result_bit_width > dtype.bit_width
                    else 0
                ),
            ] * (2 ** (carry_width + msg_width) - 2 ** (msg_width - 1))
            sign_ext_bits = ctx.tlu(result_type, msbs, sign_ext_bit_table)
            # set padding bits (where necessary) in the final result
            result = ctx.add(result_type, sum_result, sign_ext_bits)
        else:
            result = sum_result

        return result

    def tfhers_from_native(self, ctx: Context, node: Node, preds: list[Conversion]) -> Conversion:
        assert len(preds) == 1
        dtype: TFHERSIntegerType = node.properties["attributes"]["type"]
        output_tfhers_bit_width, carry_width, msg_width = (
            dtype.bit_width,
            dtype.carry_width,
            dtype.msg_width,
        )
        native_int = preds[0]

        # output_tfhers_bit_width is the target bitwidth while native_int.bit_width is the number
        # of bits containing the actual message
        assert output_tfhers_bit_width >= native_int.bit_width, (
            f"output_tfhers_bit_width: {output_tfhers_bit_width}, "
            f"native_int.bit_width: {native_int.bit_width}"
        )
        assert (
            output_tfhers_bit_width % msg_width == 0
        ), f"output_tfhers_bit_width: {output_tfhers_bit_width}, msg_width: {msg_width}"

        # number of ciphertexts representing a single integer
        num_cts = output_tfhers_bit_width // msg_width
        # number of ciphertexts representing the actual part of the message
        num_full_cts_msg = native_int.bit_width // msg_width
        remaining_bits = native_int.bit_width % msg_width
        num_half_cts_msg = 0 if remaining_bits == 0 else 1
        num_cts_msg = num_full_cts_msg + num_half_cts_msg

        # adds a dimension of ciphertexts for the result
        result_shape_msg = native_int.shape + (num_cts_msg,)

        # we reshape so that we can concatenate later over the last dim (ciphertext dim)
        reshaped_native_int = ctx.reshape(native_int, native_int.shape + (1,))

        # we want to extract `msg_width` bits at a time, and store them
        # in a `msg_width + carry_width` bits eint
        bits_type = ctx.tensor(ctx.eint(msg_width + carry_width), reshaped_native_int.shape)
        # we extract lsb first
        extracted_bits = [
            ctx.extract_bits(
                bits_type,
                reshaped_native_int,
                bits=slice(i * msg_width, (i + 1) * msg_width, 1),
            )
            for i in range(num_full_cts_msg)
        ]
        # we need to extract the remaining bits if any
        if num_half_cts_msg:
            # in the unsigned case, we won't do anything to this half full ciphertext: so we target
            # the final bitwidth directly
            target_bitwidth = remaining_bits if dtype.is_signed else (msg_width + carry_width)
            remaining_bits_type = ctx.tensor(
                ctx.eint(target_bitwidth),
                reshaped_native_int.shape,
            )
            extracted_bits.append(
                ctx.extract_bits(
                    remaining_bits_type,
                    reshaped_native_int,
                    bits=slice(
                        num_full_cts_msg * msg_width,
                        num_full_cts_msg * msg_width + remaining_bits,
                        1,
                    ),
                )
            )

        # if we expect more ciphertexts than we have in the message: we pad the result
        # if we have one non-full ct (when signed): we extend the sign bit
        if num_cts_msg != num_cts or (num_half_cts_msg and dtype.is_signed):
            # if it's signed then we have to move the sign bit to the msb
            if dtype.is_signed:
                # special case where the sign_bit is already isolated into a single ciphertext
                if remaining_bits == 1:
                    sign_bits = extracted_bits.pop()
                else:  # remaining_bits either > 1 or == 0
                    msbs = extracted_bits.pop()
                    sign_bit_idx = (remaining_bits - 1) % msg_width
                    sign_bits = ctx.extract_bits(
                        ctx.tensor(ctx.eint(1), reshaped_native_int.shape),
                        msbs,
                        bits=sign_bit_idx,
                    )
                    if remaining_bits > 1:
                        # extend sign bit in msbs
                        # For example: we have this message |0000|0110| where sign_bit_idx is 2
                        # we want the result to be |0000|1110| extending the sign bit (1) to the
                        # remaining MSBs
                        sign_extension = 2**msg_width - 1 - (2 ** (sign_bit_idx + 1) - 1)
                        extend_sign_bit_table = [
                            i + sign_extension if i & 2**sign_bit_idx else i
                            for i in range(2**remaining_bits)
                        ]
                        msbs_with_extended_sign = ctx.tlu(bits_type, msbs, extend_sign_bit_table)
                        extracted_bits.append(msbs_with_extended_sign)
                    else:  # remaining_bits == 0
                        # no need to extend the sign bit
                        extracted_bits.append(msbs)

                # padding will contain the sign
                padding_length = num_cts - len(extracted_bits)
                if padding_length:
                    extend_sign_bit_table = [0, (2**msg_width) - 1]
                    extended_sign_bits = ctx.tlu(bits_type, sign_bits, extend_sign_bit_table)
                    padding_type = ctx.tensor(
                        ctx.eint(msg_width + carry_width),
                        result_shape_msg[:-1] + (padding_length,),
                    )
                    padding = ctx.zeros(padding_type)
                    padding = ctx.add(padding_type, extended_sign_bits, padding)
                    to_concat = extracted_bits + [
                        padding,
                    ]
                else:
                    # no need to pad: we only extended the sign bit
                    to_concat = extracted_bits
            else:
                padding_length = num_cts - len(extracted_bits)
                assert padding_length > 0
                padding = ctx.zeros(
                    ctx.tensor(
                        ctx.eint(msg_width + carry_width),
                        result_shape_msg[:-1] + (padding_length,),
                    )
                )
                to_concat = extracted_bits + [
                    padding,
                ]
        else:
            # no need to pad: we have the right number of ciphertexts
            to_concat = extracted_bits

        result_shape = native_int.shape + (num_cts,)
        result_type = ctx.tensor(ctx.eint(msg_width + carry_width), result_shape)
        result = ctx.concatenate(
            result_type,
            to_concat,
            axis=-1,
        )
        return ctx.change_partition(result, dest_partition=dtype.params)

    # pylint: enable=missing-function-docstring,unused-argument
