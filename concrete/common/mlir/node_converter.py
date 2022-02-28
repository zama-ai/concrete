"""Module that provides IntermediateNode conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from typing import Any, Dict, List, Tuple, cast

import numpy
from concrete.lang.dialects import fhe, fhelinalg
from mlir.dialects import arith, linalg, tensor
from mlir.ir import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    Context,
    DenseElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    OpResult,
    RankedTensorType,
)

from ..data_types import Integer
from ..debugging import assert_true
from ..helpers.indexing_helpers import determine_new_dimension_size
from ..operator_graph import OPGraph
from ..representation.intermediate import (
    Add,
    Constant,
    Conv2D,
    Dot,
    GenericFunction,
    IndexConstant,
    IntermediateNode,
    MatMul,
    Mul,
    Sub,
)
from ..values import TensorValue
from .conversion_helpers import integer_to_mlir_type, value_to_mlir_type

# pylint: enable=no-name-in-module


class IntermediateNodeConverter:
    """Converter of IntermediateNode to MLIR."""

    ctx: Context
    op_graph: OPGraph
    node: IntermediateNode
    preds: List[OpResult]

    all_of_the_inputs_are_encrypted: bool
    all_of_the_inputs_are_tensors: bool
    one_of_the_inputs_is_a_tensor: bool

    nodes_to_mlir_names: Dict[IntermediateNode, str]
    mlir_names_to_mlir_types: Dict[str, str]
    scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]]

    def __init__(
        self,
        ctx: Context,
        op_graph: OPGraph,
        node: IntermediateNode,
        preds: List[OpResult],
        nodes_to_mlir_names: Dict[OpResult, str],
        mlir_names_to_mlir_types: Dict[str, str],
        scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]],
    ):
        self.ctx = ctx
        self.op_graph = op_graph
        self.node = node
        self.preds = preds

        self.all_of_the_inputs_are_encrypted = True
        self.all_of_the_inputs_are_tensors = True
        self.one_of_the_inputs_is_a_tensor = False

        for inp in node.inputs:
            if inp.is_clear:
                self.all_of_the_inputs_are_encrypted = False

            if isinstance(inp, TensorValue):
                if inp.is_scalar:
                    self.all_of_the_inputs_are_tensors = False
                else:
                    self.one_of_the_inputs_is_a_tensor = True
            else:  # pragma: no cover
                # this branch is not covered as there are only TensorValues for now
                self.all_of_the_inputs_are_tensors = False

        self.nodes_to_mlir_names = nodes_to_mlir_names
        self.mlir_names_to_mlir_types = mlir_names_to_mlir_types
        self.scalar_to_1d_tensor_conversion_hacks = scalar_to_1d_tensor_conversion_hacks

    def convert(self, additional_conversion_info: Dict[str, Any]) -> OpResult:
        """Convert an intermediate node to its corresponding MLIR representation.

        Args:
            additional_conversion_info (Dict[str, Any]):
                external info that the converted node might need

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        # pylint: disable=too-many-branches

        if isinstance(self.node, Add):
            result = self.convert_add()

        elif isinstance(self.node, Constant):
            result = self.convert_constant()

        elif isinstance(self.node, Dot):
            result = self.convert_dot()

        elif isinstance(self.node, GenericFunction):
            if self.node.op_name in ["flatten", "reshape"]:
                # notice flatten() == reshape(-1) and convert_reshape can handle that
                result = self.convert_reshape()
            elif self.node.op_name == "sum":
                result = self.convert_sum()
            elif self.node.op_name == "concat":
                result = self.convert_concat()
            else:
                result = self.convert_generic_function(additional_conversion_info)

        elif isinstance(self.node, IndexConstant):
            result = self.convert_index_constant()

        elif isinstance(self.node, MatMul):
            result = self.convert_matmul()

        elif isinstance(self.node, Mul):
            result = self.convert_mul()

        elif isinstance(self.node, Sub):
            result = self.convert_sub()

        elif isinstance(self.node, Conv2D):
            result = self.convert_conv2d()

        else:  # pragma: no cover
            # this branch is not covered as unsupported opeations fail on check mlir compatibility
            raise NotImplementedError(f"{type(self.node)} nodes cannot be converted to MLIR yet")

        # pylint: enable=too-many-branches

        mlir_name = str(result).replace("Value(", "").split("=", maxsplit=1)[0].strip()

        self.nodes_to_mlir_names[self.node] = mlir_name
        self.mlir_names_to_mlir_types[mlir_name] = str(result.type)

        if isinstance(self.node, (Add, Mul, Sub, Dot)):
            if self.one_of_the_inputs_is_a_tensor and not self.all_of_the_inputs_are_tensors:
                to_be_converted = []
                for (pred, output) in self.op_graph.get_ordered_preds_and_inputs_of(self.node):
                    inp = pred.outputs[output]
                    if isinstance(inp, TensorValue) and inp.is_scalar:
                        to_be_converted.append(self.nodes_to_mlir_names[pred])
                self.scalar_to_1d_tensor_conversion_hacks[mlir_name] = to_be_converted

        return result

    def convert_add(self) -> OpResult:
        """Convert an Add node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.all_of_the_inputs_are_encrypted:
            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.AddEintOp(resulting_type, *preds).result
            else:
                result = fhe.AddEintOp(resulting_type, *preds).result
        else:
            if self.node.inputs[0].is_clear:  # pragma: no cover
                # this branch is not covered as it's impossible to get into due to how tracing works
                # however, it doesn't hurt to keep it as an extra measure
                preds = preds[::-1]

            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.AddEintIntOp(resulting_type, *preds).result
            else:
                result = fhe.AddEintIntOp(resulting_type, *preds).result

        return result

    def convert_concat(self) -> OpResult:
        """Convert a "concat" node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) >= 2)
        assert_true(len(self.node.outputs) == 1)

        node = cast(GenericFunction, self.node)
        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])

        axis = node.op_kwargs.get("axis", 0)
        if axis is not None:
            if axis < 0:
                axis += len(cast(TensorValue, self.node.inputs[0]).shape)
            return fhelinalg.ConcatOp(
                resulting_type,
                self.preds,
                IntegerAttr.get(IntegerType.get_signless(64), axis),
            ).result

        flattened_preds = []
        for pred, input_value in zip(self.preds, self.node.inputs):
            input_shape = cast(TensorValue, input_value).shape
            input_size = numpy.prod(input_shape)
            input_dtype = cast(Integer, input_value.dtype)

            flattened_pred_type = RankedTensorType.get(
                [input_size],
                integer_to_mlir_type(self.ctx, input_dtype, input_value.is_encrypted),
            )
            flattened_pred = linalg.TensorCollapseShapeOp(
                flattened_pred_type,
                pred,
                ArrayAttr.get(
                    [
                        ArrayAttr.get(
                            [
                                IntegerAttr.get(IndexType.parse("index"), i)
                                for i in range(len(input_shape))
                            ]
                        )
                    ]
                ),
            ).result
            flattened_preds.append(flattened_pred)

        return fhelinalg.ConcatOp(
            resulting_type,
            flattened_preds,
            IntegerAttr.get(IntegerType.get_signless(64), 0),
        ).result

    def convert_constant(self) -> OpResult:
        """Convert a Constant node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 0)
        assert_true(len(self.node.outputs) == 1)

        value = self.node.outputs[0]
        if not isinstance(value, TensorValue):  # pragma: no cover
            # this branch is not covered as there are only TensorValues for now
            raise NotImplementedError(f"{value} constants cannot be converted to MLIR yet")

        resulting_type = value_to_mlir_type(self.ctx, value)
        data = cast(Constant, self.node).constant_data

        if value.is_scalar:
            attr = IntegerAttr.get(resulting_type, data)
        else:
            # usage of `Attribute.parse` is the result of some limitations in the MLIR module
            # provided by LLVM

            # what should have been used is `DenseElementsAttr` but it's impossible to assign
            # custom bit-widths using it (e.g., uint5)

            # since we coudn't create a `DenseElementsAttr` with a custom bit width using python api
            # we use `Attribute.parse` to let the underlying library do it by itself

            attr = Attribute.parse(f"dense<{str(data.tolist())}> : {resulting_type}")

        return arith.ConstantOp(resulting_type, attr).result

    def convert_conv2d(self) -> OpResult:
        """Convert a Conv2D node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2 or len(self.node.inputs) == 3)
        assert_true(len(self.node.outputs) == 1)
        has_bias = len(self.node.inputs) == 3

        x = self.node.inputs[0]
        weight = self.node.inputs[1]
        if not (x.is_encrypted and weight.is_clear):  # pragma: no cover
            raise NotImplementedError(
                f"Conv2D with input {x} and weight {weight} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        node = cast(Conv2D, self.node)
        integer_type = IntegerType.get_signless(64, context=self.ctx)
        strides = DenseElementsAttr.get(
            numpy.array(list(node.strides), dtype=numpy.uint64),
            context=self.ctx,
            type=integer_type,
        )
        dilations = DenseElementsAttr.get(
            numpy.array(list(node.dilations), dtype=numpy.uint64),
            context=self.ctx,
            type=integer_type,
        )
        pads = DenseElementsAttr.get(
            numpy.array(list(node.pads), dtype=numpy.uint64), context=self.ctx, type=integer_type
        )
        if has_bias:
            result = fhelinalg.Conv2dOp(resulting_type, *preds, pads, strides, dilations).result
        else:
            result = fhelinalg.Conv2dOp(
                resulting_type, *preds, None, pads, strides, dilations
            ).result

        return result

    def convert_dot(self) -> OpResult:
        """Convert a Dot node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]
            raise NotImplementedError(
                f"Dot product between {lhs} and {rhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.node.inputs[0].is_clear:
            preds = preds[::-1]

        if self.all_of_the_inputs_are_tensors:
            # numpy.dot(x, y) where x and y are both vectors = regular dot product
            result = fhelinalg.Dot(resulting_type, *preds).result

        elif not self.one_of_the_inputs_is_a_tensor:
            # numpy.dot(x, y) where x and y are both scalars = x * y
            result = fhe.MulEintIntOp(resulting_type, *preds).result

        else:
            # numpy.dot(x, y) where one of x or y is a scalar and the other one is a vector = x * y
            result = fhelinalg.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_generic_function(self, additional_conversion_info: Dict[str, Any]) -> OpResult:
        """Convert a GenericFunction node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        variable_input_indices = [
            idx
            for idx, inp in enumerate(self.op_graph.get_ordered_preds(self.node))
            if not isinstance(inp, Constant)
        ]
        if len(variable_input_indices) != 1:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how tracing works
            # however, it doesn't hurt to keep it as an extra measure
            raise NotImplementedError(
                "Table lookups with more than one variable input cannot be converted to MLIR yet"
            )
        variable_input_index = variable_input_indices[0]

        assert_true(len(self.node.outputs) == 1)
        output = self.node.outputs[0]

        value = self.node.inputs[variable_input_index]
        assert_true(value.is_encrypted)

        if not isinstance(value.dtype, Integer):  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how compilation works
            # however, it doesn't hurt to keep it as an extra measure
            raise NotImplementedError(f"Table lookup on {value} cannot be converted to MLIR yet")

        tables = additional_conversion_info["tables"][self.node]
        assert_true(len(tables) > 0)

        lut_shape: Tuple[int, ...] = ()
        map_shape: Tuple[int, ...] = ()

        if len(tables) == 1:
            table = tables[0][0]

            # The reduction on 63b is to avoid problems like doing a TLU of
            # the form T[j] = 2<<j, for j which is supposed to be 7b as per
            # constraint of the compiler, while in practice, it is a small
            # value. Reducing on 64b was not ok for some reason
            lut_shape = (len(table),)
            lut_values = numpy.array(table % (2 << 63), dtype=numpy.uint64)

            map_shape = ()
            map_values = None
        else:
            assert_true(isinstance(output, TensorValue))
            assert isinstance(output, TensorValue)

            individual_table_size = len(tables[0][0])

            lut_shape = (len(tables), individual_table_size)
            map_shape = output.shape

            lut_values = numpy.zeros(lut_shape, dtype=numpy.uint64)
            map_values = numpy.zeros(map_shape, dtype=numpy.intp)

            for i, (table, indices) in enumerate(tables):
                assert_true(len(table) == individual_table_size)
                lut_values[i, :] = table
                for index in indices:
                    map_values[index] = i

        lut_type = RankedTensorType.get(lut_shape, IntegerType.get_signless(64, context=self.ctx))
        lut_attr = DenseElementsAttr.get(lut_values, context=self.ctx)
        lut = arith.ConstantOp(lut_type, lut_attr).result

        resulting_type = value_to_mlir_type(self.ctx, output)
        pred = self.preds[variable_input_index]

        if self.one_of_the_inputs_is_a_tensor:
            if len(tables) == 1:
                result = fhelinalg.ApplyLookupTableEintOp(resulting_type, pred, lut).result
            else:
                assert_true(map_shape != ())
                assert_true(map_values is not None)

                index_type = IndexType.parse("index")
                map_type = RankedTensorType.get(map_shape, index_type)
                map_attr = DenseElementsAttr.get(map_values, context=self.ctx, type=index_type)

                result = fhelinalg.ApplyMappedLookupTableEintOp(
                    resulting_type,
                    pred,
                    lut,
                    arith.ConstantOp(map_type, map_attr).result,
                ).result
        else:
            result = fhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result

        return result

    def convert_index_constant(self) -> OpResult:
        """Convert a IndexConstant node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        # pylint: disable=too-many-locals

        assert_true(len(self.node.inputs) == 1)
        assert_true(len(self.node.outputs) == 1)

        tensor_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        pred = self.preds[0]

        input_value = cast(TensorValue, self.node.inputs[0])
        input_shape = input_value.shape

        index = cast(IndexConstant, self.node).index
        index_str = self.node.text_for_formatting([""], 0)

        index_type = IndexType.parse("index")

        if len(index) == len(input_shape) and all(isinstance(i, int) for i in index):
            indices = []
            for value, dimension_size in zip(index, input_shape):
                assert isinstance(value, int)  # mypy
                attr = IntegerAttr.get(index_type, value if value >= 0 else value + dimension_size)
                indices.append(arith.ConstantOp(index_type, attr).result)
            return tensor.ExtractOp(tensor_type, pred, indices).result

        offsets = []
        sizes = []
        strides = []

        destroyed_dimensions = []
        for dimension, (indexing_element, dimension_size) in enumerate(zip(index, input_shape)):

            if isinstance(indexing_element, int):
                destroyed_dimensions.append(dimension)
                size = 1
                stride = 1
                offset = (
                    indexing_element if indexing_element >= 0 else indexing_element + dimension_size
                )

            elif isinstance(indexing_element, slice):
                size = determine_new_dimension_size(
                    indexing_element,
                    dimension_size,
                    dimension,
                    input_shape,
                    index_str,
                )
                stride = indexing_element.step if isinstance(indexing_element.step, int) else 1
                offset = (
                    (
                        indexing_element.start
                        if indexing_element.start >= 0
                        else indexing_element.start + dimension_size
                    )
                    if isinstance(indexing_element.start, int)
                    else (0 if stride > 0 else dimension_size - 1)
                )

            else:  # pragma: no cover
                # this branch is impossible to reach with all the previous checks
                # but let's keep it as an extra measure
                raise NotImplementedError(
                    f"Indexing of {input_value} with {index_str} cannot be converted to MLIR",
                )

            offsets.append(offset)
            sizes.append(size)
            strides.append(stride)

        if len(destroyed_dimensions) == 0:
            return tensor.ExtractSliceOp(
                tensor_type,
                pred,
                [],
                [],
                [],
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in offsets]),
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in sizes]),
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in strides]),
            ).result

        output_value = cast(TensorValue, self.node.outputs[0])

        intermediate_shape = list(output_value.shape)
        for dimension in destroyed_dimensions:
            intermediate_shape.insert(dimension, 1)

        intermediate_type = RankedTensorType.get(
            intermediate_shape,
            integer_to_mlir_type(
                self.ctx,
                cast(Integer, output_value.dtype),
                output_value.is_encrypted,
            ),
        )

        intermediate = tensor.ExtractSliceOp(
            intermediate_type,
            pred,
            [],
            [],
            [],
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in offsets]),
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in sizes]),
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in strides]),
        ).result

        reassociaton = []

        current_intermediate_dimension = 0
        for _ in range(len(output_value.shape)):
            indices = [current_intermediate_dimension]
            while current_intermediate_dimension in destroyed_dimensions:
                current_intermediate_dimension += 1
                indices.append(current_intermediate_dimension)

            reassociaton.append(indices)
            current_intermediate_dimension += 1
        while current_intermediate_dimension < len(intermediate_shape):
            reassociaton[-1].append(current_intermediate_dimension)
            current_intermediate_dimension += 1

        return linalg.TensorCollapseShapeOp(
            tensor_type,
            intermediate,
            ArrayAttr.get(
                [
                    ArrayAttr.get(
                        [IntegerAttr.get(index_type, index) for index in indices],
                    )
                    for indices in reassociaton
                ],
            ),
        ).result

        # pylint: enable=too-many-locals

    def convert_matmul(self) -> OpResult:
        """Convert a MatMul node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]
            raise NotImplementedError(
                f"Matrix multiplication between {lhs} and {rhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        assert isinstance(self.node.outputs[0], TensorValue)
        if self.node.outputs[0].shape == ():
            if self.node.inputs[0].is_clear:
                preds = preds[::-1]
            result = fhelinalg.Dot(resulting_type, *preds).result

        elif self.node.inputs[0].is_clear:
            result = fhelinalg.MatMulIntEintOp(resulting_type, *preds).result
        else:
            result = fhelinalg.MatMulEintIntOp(resulting_type, *preds).result

        return result

    def convert_mul(self) -> OpResult:
        """Convert a Mul node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]
            raise NotImplementedError(
                f"Multiplication between {lhs} and {rhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.node.inputs[0].is_clear:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how tracing works
            # however, it doesn't hurt to keep it as an extra measure
            preds = preds[::-1]

        if self.one_of_the_inputs_is_a_tensor:
            result = fhelinalg.MulEintIntOp(resulting_type, *preds).result
        else:
            result = fhe.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_reshape(self) -> OpResult:
        """Convert a "reshape" node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 1)
        assert_true(len(self.node.outputs) == 1)

        assert_true(isinstance(self.node.inputs[0], TensorValue))
        input_shape = cast(TensorValue, self.node.inputs[0]).shape

        assert_true(isinstance(self.node.outputs[0], TensorValue))
        output_shape = cast(TensorValue, self.node.outputs[0]).shape

        pred = self.preds[0]
        if input_shape == output_shape:
            return pred

        # we can either collapse or expand, which changes the number of dimensions
        # this is a limitation of the current compiler and it will be improved in the future (#1060)
        can_be_converted_directly = len(input_shape) != len(output_shape)

        reassociation: List[List[int]] = []
        if can_be_converted_directly:
            if len(output_shape) == 1:
                # output is 1 dimensional so collapse every dimension into the same dimension
                reassociation.append(list(range(len(input_shape))))
            else:
                # input is m dimensional
                # output is n dimensional
                # and m is different than n

                # we don't want to duplicate code so we forget about input and output
                # and we focus on smaller shape and bigger shape

                smaller_shape, bigger_shape = (
                    (output_shape, input_shape)
                    if len(output_shape) < len(input_shape)
                    else (input_shape, output_shape)
                )
                s_index, b_index = 0, 0

                # now we will figure out how to group the bigger shape to get the smaller shape
                # think of the algorithm below as
                #     keep merging the dimensions of the bigger shape
                #     until we have a match on the smaller shape
                #     then try to match the next dimension of the smaller shape
                #     if all dimensions of the smaller shape is matched
                #     we can convert it

                group = []
                size = 1
                while s_index < len(smaller_shape) and b_index < len(bigger_shape):
                    # dimension `b_index` of `bigger_shape` belongs to current group
                    group.append(b_index)

                    # and current group has `size * bigger_shape[b_index]` elements now
                    size *= bigger_shape[b_index]

                    # if current group size matches the dimension `s_index` of `smaller_shape`
                    if size == smaller_shape[s_index]:
                        # we finalize this group and reset everything
                        size = 1
                        reassociation.append(group)
                        group = []

                        # now try to match the next dimension of `smaller_shape`
                        s_index += 1

                    # now process the next dimension of `bigger_shape`
                    b_index += 1

                # handle the case where bigger shape has proceeding 1s
                # e.g., (5,) -> (5, 1)
                while b_index < len(bigger_shape) and bigger_shape[b_index] == 1:
                    reassociation[-1].append(b_index)
                    b_index += 1

                # if not all dimensions of both shapes are processed exactly
                if s_index != len(smaller_shape) or b_index != len(bigger_shape):
                    # we cannot convert
                    can_be_converted_directly = False

        index_type = IndexType.parse("index")
        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])

        if can_be_converted_directly:
            reassociation_attr = ArrayAttr.get(
                [
                    ArrayAttr.get([IntegerAttr.get(index_type, dimension) for dimension in group])
                    for group in reassociation
                ]
            )
            if len(output_shape) < len(input_shape):
                return linalg.TensorCollapseShapeOp(resulting_type, pred, reassociation_attr).result
            return linalg.TensorExpandShapeOp(resulting_type, pred, reassociation_attr).result

        flattened_type = value_to_mlir_type(
            self.ctx,
            TensorValue(
                self.node.inputs[0].dtype,
                self.node.inputs[0].is_encrypted,
                (numpy.prod(input_shape),),
            ),
        )
        flattened_result = linalg.TensorCollapseShapeOp(
            flattened_type,
            pred,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(index_type, i) for i in range(len(input_shape))])]
            ),
        ).result

        return linalg.TensorExpandShapeOp(
            resulting_type,
            flattened_result,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(index_type, i) for i in range(len(output_shape))])]
            ),
        ).result

    def convert_sub(self) -> OpResult:
        """Convert a Sub node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        lhs = self.node.inputs[0]
        rhs = self.node.inputs[1]
        if not (lhs.is_clear and rhs.is_encrypted):
            raise NotImplementedError(
                f"Subtraction of {rhs} from {lhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.one_of_the_inputs_is_a_tensor:
            result = fhelinalg.SubIntEintOp(resulting_type, *preds).result
        else:
            result = fhe.SubIntEintOp(resulting_type, *preds).result

        return result

    def convert_sum(self) -> OpResult:
        """Convert a "sum" node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 1)
        assert_true(len(self.node.outputs) == 1)

        node = cast(GenericFunction, self.node)
        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])

        axes = node.op_kwargs.get("axis", [])
        keep_dims = node.op_kwargs.get("keepdims", False)

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, tuple):
            axes = list(axes)

        input_dimensions = len(cast(TensorValue, self.node.inputs[0]).shape)
        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] += input_dimensions

        return fhelinalg.SumOp(
            resulting_type,
            self.preds[0],
            ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]),
            BoolAttr.get(keep_dims),
        ).result
