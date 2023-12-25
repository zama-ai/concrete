"""
Declaration of `AssignBitWidths` graph processor.
"""

from typing import Dict, List

import z3

from ...compilation.configuration import (
    BitwiseStrategy,
    ComparisonStrategy,
    MinMaxStrategy,
    MultivariateStrategy,
)
from ...dtypes import Integer
from ...representation import Graph, Node, Operation
from . import GraphProcessor


class AssignBitWidths(GraphProcessor):
    """
    AssignBitWidths graph processor, to assign proper bit-widths to be compatible with FHE.

    There are two modes:
    - Single Precision, where all encrypted values have the same precision.
    - Multi Precision, where encrypted values can have different precisions.

    There is preference list for comparison strategies.
    - Strategies will be traversed in order and bit-widths
      will be assigned according to the first available strategy.
    """

    single_precision: bool
    comparison_strategy_preference: List[ComparisonStrategy]
    bitwise_strategy_preference: List[BitwiseStrategy]
    shifts_with_promotion: bool
    multivariate_strategy_preference: List[MultivariateStrategy]
    min_max_strategy_preference: List[MinMaxStrategy]

    def __init__(
        self,
        single_precision: bool,
        comparison_strategy_preference: List[ComparisonStrategy],
        bitwise_strategy_preference: List[BitwiseStrategy],
        shifts_with_promotion: bool,
        multivariate_strategy_preference: List[MultivariateStrategy],
        min_max_strategy_preference: List[MinMaxStrategy],
    ):
        self.single_precision = single_precision
        self.comparison_strategy_preference = comparison_strategy_preference
        self.bitwise_strategy_preference = bitwise_strategy_preference
        self.shifts_with_promotion = shifts_with_promotion
        self.multivariate_strategy_preference = multivariate_strategy_preference
        self.min_max_strategy_preference = min_max_strategy_preference

    def apply(self, graph: Graph):
        optimizer = z3.Optimize()

        max_bit_width: z3.Int = z3.Int("max")
        bit_widths: Dict[Node, z3.Int] = {}

        additional_constraints = AdditionalConstraints(
            optimizer,
            graph,
            bit_widths,
            self.comparison_strategy_preference,
            self.bitwise_strategy_preference,
            self.shifts_with_promotion,
            self.multivariate_strategy_preference,
            self.min_max_strategy_preference,
        )

        nodes = graph.query_nodes(ordered=True)
        for i, node in enumerate(nodes):
            assert isinstance(node.output.dtype, Integer)
            required_bit_width = node.output.dtype.bit_width

            bit_width_hint = node.properties.get("bit_width_hint")
            if bit_width_hint is not None:
                required_bit_width = max(required_bit_width, bit_width_hint)

            bit_width = z3.Int(f"%{i}")
            bit_widths[node] = bit_width

            base_constraint = bit_width >= required_bit_width
            node.bit_width_constraints.append(base_constraint)

            optimizer.add(base_constraint)
            optimizer.add(max_bit_width >= bit_width)

            additional_constraints.generate_for(node, bit_width)

        if self.single_precision:
            for bit_width in bit_widths.values():
                optimizer.add(bit_width == max_bit_width)

        optimizer.minimize(sum(bit_width for bit_width in bit_widths.values()))

        assert optimizer.check() == z3.sat
        model = optimizer.model()

        for node, bit_width in bit_widths.items():
            assert isinstance(node.output.dtype, Integer)
            new_bit_width = model[bit_width].as_long()

            if node.output.is_clear:
                new_bit_width += 1

            node.properties["original_bit_width"] = node.output.dtype.bit_width
            node.output.dtype.bit_width = new_bit_width

        graph.bit_width_constraints = optimizer
        graph.bit_width_assignments = model


class AdditionalConstraints:
    """
    AdditionalConstraints class to customize bit-width assignment step easily.
    """

    optimizer: z3.Optimize
    graph: Graph
    bit_widths: Dict[Node, z3.Int]

    comparison_strategy_preference: List[ComparisonStrategy]
    bitwise_strategy_preference: List[BitwiseStrategy]
    shifts_with_promotion: bool
    multivariate_strategy_preference: List[MultivariateStrategy]
    min_max_strategy_preference: List[MinMaxStrategy]

    node: Node
    bit_width: z3.Int

    # pylint: disable=missing-function-docstring,unused-argument

    def __init__(
        self,
        optimizer: z3.Optimize,
        graph: Graph,
        bit_widths: Dict[Node, z3.Int],
        comparison_strategy_preference: List[ComparisonStrategy],
        bitwise_strategy_preference: List[BitwiseStrategy],
        shifts_with_promotion: bool,
        multivariate_strategy_preference: List[MultivariateStrategy],
        min_max_strategy_preference: List[MinMaxStrategy],
    ):
        self.optimizer = optimizer
        self.graph = graph
        self.bit_widths = bit_widths

        self.comparison_strategy_preference = comparison_strategy_preference
        self.bitwise_strategy_preference = bitwise_strategy_preference
        self.shifts_with_promotion = shifts_with_promotion
        self.multivariate_strategy_preference = multivariate_strategy_preference
        self.min_max_strategy_preference = min_max_strategy_preference

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

        if node.operation == Operation.Generic and node.properties["attributes"].get(
            "is_multivariate"
        ):
            operation_name = "multivariate"

        if hasattr(self, operation_name):
            constraints = getattr(self, operation_name)
            preds = self.graph.ordered_preds_of(node)

            if callable(constraints):
                constraints(node, preds)

            elif isinstance(constraints, set):
                for add_constraint in constraints:
                    add_constraint(self, node, preds)

            elif isinstance(constraints, dict):
                for conditions, conditional_constraints in constraints.items():
                    if not isinstance(conditions, tuple):
                        conditions = (conditions,)

                    if all(condition(self, node, preds) for condition in conditions):
                        for add_constraint in conditional_constraints:
                            add_constraint(self, node, preds)

            else:  # pragma: no cover
                message = (
                    f"Expected a set or a dict "
                    f"for additional constraints of '{operation_name}' operation "
                    f"but got {type(constraints).__name__} instead"
                )
                raise ValueError(message)

    def constraint(self, node: Node, constraint: z3.BoolRef):
        node.bit_width_constraints.append(constraint)
        self.optimizer.add(constraint)

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
            self.constraint(node, self.bit_widths[preds[i]] == self.bit_widths[preds[i + 1]])

    def inputs_and_output_share_precision(self, node: Node, preds: List[Node]):
        self.inputs_share_precision(node, preds)
        if len(preds) != 0:
            self.constraint(node, self.bit_widths[preds[-1]] == self.bit_widths[node])

    def inputs_require_one_more_bit(self, node: Node, preds: List[Node]):
        for pred in preds:
            assert isinstance(pred.output.dtype, Integer)

            actual_bit_width = pred.output.dtype.bit_width
            required_bit_width = actual_bit_width + 1

            self.constraint(node, self.bit_widths[pred] >= required_bit_width)

    def comparison(self, node: Node, preds: List[Node]):
        assert len(preds) == 2

        x = preds[0]
        y = preds[1]

        assert x.output.is_encrypted
        assert y.output.is_encrypted

        strategies = self.comparison_strategy_preference
        fallback = [
            ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED,
            ComparisonStrategy.CHUNKED,
        ]

        for strategy in strategies + fallback:
            if strategy.can_be_used(x.output, y.output):
                new_x_bit_width, new_y_bit_width = strategy.promotions(x.output, y.output)
                self.constraint(node, self.bit_widths[x] >= new_x_bit_width)
                self.constraint(node, self.bit_widths[y] >= new_y_bit_width)

                if strategy == ComparisonStrategy.ONE_TLU_PROMOTED:
                    self.constraint(node, self.bit_widths[x] == self.bit_widths[y])

                node.properties["strategy"] = strategy
                break

    def bitwise(self, node: Node, preds: List[Node]):
        assert len(preds) == 2

        x = preds[0]
        y = preds[1]

        assert x.output.is_encrypted
        assert y.output.is_encrypted

        strategies = self.bitwise_strategy_preference
        fallback = [
            BitwiseStrategy.CHUNKED,
        ]

        for strategy in strategies + fallback:
            if strategy.can_be_used(x.output, y.output):
                new_x_bit_width, new_y_bit_width = strategy.promotions(x.output, y.output)
                self.constraint(node, self.bit_widths[x] >= new_x_bit_width)
                self.constraint(node, self.bit_widths[y] >= new_y_bit_width)

                if strategy == BitwiseStrategy.ONE_TLU_PROMOTED:
                    self.constraint(node, self.bit_widths[x] == self.bit_widths[y])

                node.properties["strategy"] = strategy
                break

        if (
            node.properties.get("name", None) in {"left_shift", "right_shift"}
            and node.properties["strategy"] == BitwiseStrategy.CHUNKED
            and self.shifts_with_promotion
        ):
            self.constraint(node, self.bit_widths[x] == self.bit_widths[node])

    def multivariate(self, node: Node, preds: List[Node]):
        assert all(
            pred.output.is_encrypted and pred.properties.get("name") != "round_bit_pattern"
            for pred in preds
        )

        strategies = self.multivariate_strategy_preference
        fallback = [
            MultivariateStrategy.CASTED,
        ]

        for strategy in strategies + fallback:
            if strategy.can_be_used(*(pred.output for pred in preds)):
                promotions = strategy.promotions(*(pred.output for pred in preds))
                for pred, promotion in zip(preds, promotions):
                    self.constraint(node, self.bit_widths[pred] >= promotion)

                if strategy == MultivariateStrategy.PROMOTED:
                    for i in range(len(preds) - 1):
                        self.constraint(
                            node,
                            self.bit_widths[preds[i]] == self.bit_widths[preds[i + 1]],
                        )

                node.properties["strategy"] = strategy
                break

    def min_max(self, node: Node, preds: List[Node]):
        assert len(preds) == 2

        x = preds[0]
        y = preds[1]

        assert x.output.is_encrypted
        assert y.output.is_encrypted

        assert isinstance(x.output.dtype, Integer)
        assert isinstance(y.output.dtype, Integer)
        assert isinstance(node.output.dtype, Integer)

        x_bit_width = x.output.dtype.bit_width
        y_bit_width = y.output.dtype.bit_width
        result_bit_width = node.output.dtype.bit_width

        if y_bit_width != result_bit_width:
            x_can_be_added_directly = result_bit_width == x_bit_width
            x_has_smaller_bit_width = x_bit_width < y_bit_width
            if x_can_be_added_directly or x_has_smaller_bit_width:
                x, y = y, x

        strategies = self.min_max_strategy_preference
        fallback = [
            MinMaxStrategy.CHUNKED,
        ]

        for strategy in strategies + fallback:
            if strategy.can_be_used(x.output, y.output):
                new_x_bit_width, new_y_bit_width = strategy.promotions(x.output, y.output)
                self.constraint(node, self.bit_widths[x] >= new_x_bit_width)
                self.constraint(node, self.bit_widths[y] >= new_y_bit_width)

                if strategy == MinMaxStrategy.ONE_TLU_PROMOTED:
                    self.constraint(node, self.bit_widths[x] == self.bit_widths[y])
                    self.constraint(node, self.bit_widths[y] == self.bit_widths[node])

                node.properties["strategy"] = strategy
                break

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
            bitwise,
        },
    }

    bitwise_or = {
        all_inputs_are_encrypted: {
            bitwise,
        },
    }

    bitwise_xor = {
        all_inputs_are_encrypted: {
            bitwise,
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
            comparison,
        },
    }

    expand_dims = {
        inputs_and_output_share_precision,
    }

    greater = {
        all_inputs_are_encrypted: {
            comparison,
        },
    }

    greater_equal = {
        all_inputs_are_encrypted: {
            comparison,
        },
    }

    index_static = {
        inputs_and_output_share_precision,
    }

    left_shift = {
        all_inputs_are_encrypted: {
            bitwise,
        },
    }

    less = {
        all_inputs_are_encrypted: {
            comparison,
        },
    }

    less_equal = {
        all_inputs_are_encrypted: {
            comparison,
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

    maximum = {
        all_inputs_are_encrypted: {
            min_max,
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

    minimum = {
        all_inputs_are_encrypted: {
            min_max,
        },
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
            comparison,
        },
    }

    reshape = {
        inputs_and_output_share_precision,
    }

    right_shift = {
        all_inputs_are_encrypted: {
            bitwise,
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

    truncate_bit_pattern = {
        inputs_and_output_share_precision,
    }
