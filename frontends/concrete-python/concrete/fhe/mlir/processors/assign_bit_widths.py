"""
Declaration of `AssignBitWidths` graph processor.
"""

from typing import Dict, List

import z3

from ...dtypes import Integer
from ...representation import Graph, Node, Operation
from . import GraphProcessor


class AssignBitWidths(GraphProcessor):
    """
    AssignBitWidths graph processor, to assign proper bit-widths to be compatible with FHE.

    There are two modes:
    - Single Precision, where all encrypted values have the same precision.
    - Multi Precision, where encrypted values can have different precisions.
    """

    def __init__(self, single_precision=False):
        self.single_precision = single_precision

    def apply(self, graph: Graph):
        optimizer = z3.Optimize()

        max_bit_width: z3.Int = z3.Int("max")
        bit_widths: Dict[Node, z3.Int] = {}

        additional_constraints = AdditionalConstraints(optimizer, graph, bit_widths)

        nodes = graph.query_nodes(ordered=True)
        for i, node in enumerate(nodes):
            assert isinstance(node.output.dtype, Integer)
            required_bit_width = node.output.dtype.bit_width

            bit_width_hint = node.properties.get("bit_width_hint")
            if bit_width_hint is not None:
                required_bit_width = max(required_bit_width, bit_width_hint)

            bit_width = z3.Int(f"%{i}")
            bit_widths[node] = bit_width

            optimizer.add(max_bit_width >= bit_width)
            optimizer.add(bit_width >= required_bit_width)

            additional_constraints.generate_for(node, bit_width)

        if self.single_precision:
            for bit_width in bit_widths.values():
                optimizer.add(bit_width == max_bit_width)

        optimizer.minimize(sum(bit_width**2 for bit_width in bit_widths.values()))

        assert optimizer.check() == z3.sat
        model = optimizer.model()

        for node, bit_width in bit_widths.items():
            assert isinstance(node.output.dtype, Integer)
            new_bit_width = model[bit_width].as_long()

            if node.output.is_clear:
                new_bit_width += 1

            node.properties["original_bit_width"] = node.output.dtype.bit_width
            node.output.dtype.bit_width = new_bit_width


class AdditionalConstraints:
    """
    AdditionalConstraints class to customize bit-width assignment step easily.
    """

    optimizer: z3.Optimize
    graph: Graph
    bit_widths: Dict[Node, z3.Int]

    node: Node
    bit_width: z3.Int

    # pylint: disable=missing-function-docstring,unused-argument

    def __init__(self, optimizer: z3.Optimize, graph: Graph, bit_widths: Dict[Node, z3.Int]):
        self.optimizer = optimizer
        self.graph = graph
        self.bit_widths = bit_widths

    def generate_for(self, node: Node, bit_width: z3.Int):
        """
        Generate additional constraints for a node.

        Args:
            node (Node):
                node to generate constraints for

            bit_width (z3.Int):
                symbolic bit-width which will be assigned to node once constraints are solved
        """

        assert node.operation in {Operation.Generic, Operation.Constant, Operation.Input}
        operation_name = (
            node.properties["name"]
            if node.operation == Operation.Generic
            else ("constant" if node.operation == Operation.Constant else "input")
        )

        if hasattr(self, operation_name):
            constraints = getattr(self, operation_name)
            preds = self.graph.ordered_preds_of(node)

            if isinstance(constraints, set):
                for add_constraint in constraints:
                    add_constraint(self, node, preds)

            elif isinstance(constraints, dict):
                for condition, conditional_constraints in constraints.items():
                    if condition(self, node, preds):
                        for add_constraint in conditional_constraints:
                            add_constraint(self, node, preds)

            else:  # pragma: no cover
                message = (
                    f"Expected a set or a dict "
                    f"for additional constraints of '{operation_name}' operation"
                    f"but got {type(constraints).__name__} instead"
                )
                raise ValueError(message)

    # ==========
    # Conditions
    # ==========

    def all_inputs_are_encrypted(self, node: Node, preds: List[Node]) -> bool:
        return all(pred.output.is_encrypted for pred in preds)

    def some_inputs_are_clear(self, node: Node, preds: List[Node]) -> bool:
        return any(pred.output.is_clear for pred in preds)

    def has_overflow_protection(self, node: Node, preds: List[Node]) -> bool:
        return node.properties["attributes"]["overflow_protection"] is True

    # ===========
    # Constraints
    # ===========

    def inputs_share_precision(self, node: Node, preds: List[Node]):
        for i in range(len(preds) - 1):
            self.optimizer.add(self.bit_widths[preds[i]] == self.bit_widths[preds[i + 1]])

    def inputs_and_output_share_precision(self, node: Node, preds: List[Node]):
        self.inputs_share_precision(node, preds)
        if len(preds) != 0:
            self.optimizer.add(self.bit_widths[preds[-1]] == self.bit_widths[node])

    def inputs_require_one_more_bit(self, node: Node, preds: List[Node]):
        for pred in preds:
            assert isinstance(pred.output.dtype, Integer)

            actual_bit_width = pred.output.dtype.bit_width
            required_bit_width = actual_bit_width + 1

            self.optimizer.add(self.bit_widths[pred] >= required_bit_width)

    def inputs_require_at_least_four_bits(self, node: Node, preds: List[Node]):
        for pred in preds:
            self.optimizer.add(self.bit_widths[pred] >= 4)

    # ==========
    # Operations
    # ==========

    add = {
        inputs_and_output_share_precision,
    }

    array = {
        inputs_and_output_share_precision,
    }

    assign_static = {
        inputs_and_output_share_precision,
    }

    bitwise_and = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    bitwise_or = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    bitwise_xor = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    broadcast_to = {
        inputs_and_output_share_precision,
    }

    concatenate = {
        inputs_and_output_share_precision,
    }

    conv1d = {
        inputs_and_output_share_precision,
    }

    conv2d = {
        inputs_and_output_share_precision,
    }

    conv3d = {
        inputs_and_output_share_precision,
    }

    copy = {
        inputs_and_output_share_precision,
    }

    dot = {
        all_inputs_are_encrypted: {
            inputs_share_precision,
            inputs_require_one_more_bit,
        },
        some_inputs_are_clear: {
            inputs_and_output_share_precision,
        },
    }

    equal = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    expand_dims = {
        inputs_and_output_share_precision,
    }

    greater = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
            inputs_require_at_least_four_bits,
        },
    }

    greater_equal = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
            inputs_require_at_least_four_bits,
        },
    }

    index_static = {
        inputs_and_output_share_precision,
    }

    left_shift = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    less = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
            inputs_require_at_least_four_bits,
        },
    }

    less_equal = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
            inputs_require_at_least_four_bits,
        },
    }

    matmul = {
        all_inputs_are_encrypted: {
            inputs_share_precision,
            inputs_require_one_more_bit,
        },
        some_inputs_are_clear: {
            inputs_and_output_share_precision,
        },
    }

    maxpool1d = {
        inputs_and_output_share_precision,
        inputs_require_one_more_bit,
    }

    maxpool2d = {
        inputs_and_output_share_precision,
        inputs_require_one_more_bit,
    }

    maxpool3d = {
        inputs_and_output_share_precision,
        inputs_require_one_more_bit,
    }

    multiply = {
        all_inputs_are_encrypted: {
            inputs_share_precision,
            inputs_require_one_more_bit,
        },
        some_inputs_are_clear: {
            inputs_and_output_share_precision,
        },
    }

    negative = {
        inputs_and_output_share_precision,
    }

    not_equal = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    reshape = {
        inputs_and_output_share_precision,
    }

    right_shift = {
        all_inputs_are_encrypted: {
            inputs_and_output_share_precision,
        },
    }

    round_bit_pattern = {
        has_overflow_protection: {
            inputs_and_output_share_precision,
        },
    }

    subtract = {
        inputs_and_output_share_precision,
    }

    sum = {
        inputs_and_output_share_precision,
    }

    squeeze = {
        inputs_and_output_share_precision,
    }

    transpose = {
        inputs_and_output_share_precision,
    }
