"""
Convert the simple Tlu Dag to a concrete-python Tracer function.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
from mlir.dialects import arith

from concrete.fhe.dtypes.integer import Integer
from concrete.fhe.extensions.synthesis.verilog_to_luts import TluCircuit, TluNode
from concrete.fhe.mlir import context as mlir_context
from concrete.fhe.mlir.conversion import Conversion
from concrete.fhe.values.value_description import ValueDescription

WEIGHT_TO_TLU = True
RESCALE_BEFORE_TLU = True


def power_of_2_scale(v):
    """Compute the exact log2 of v."""
    assert v > 0
    if v % 2 == 0:
        return 1 + power_of_2_scale(v // 2)
    else:
        assert v == 1
        return 0


def compute_max_arity_use(
    circuit: TluCircuit,
    params: 'dict[str, ValueDescription | list[ValueDescription]]',
    scheduled_nodes,
    bit_location,
    used_count,
):
    """Collect max arity use for all values to later compute all bitwidth."""
    max_arity_use = {}
    for bits in circuit.parameters.values():
        for value in bits:
            max_arity_use[value.name] = 1
    for tlu in scheduled_nodes:
        for value in tlu.arguments:
            max_arity_use[value.name] = max(max_arity_use[value.name], tlu.arity)
        for value in tlu.results:
            assert value.name not in max_arity_use
            max_arity_use[value.name] = 1

    # Result can be either precision correct on single use or < 7 or converted
    for result in circuit.results:
        for res_bit in result:
            base_name = result[0].origin.base_name
            if isinstance(params[base_name], list):
                word_i = bit_location[base_name][res_bit.name].word
                dtype = params[base_name][word_i].dtype  # type: ignore
            else:
                dtype = params[base_name].dtype  # type: ignore
            bit_width = dtype.bit_width  # type: ignore
            use_bit_width = (
                # direct to result = exact
                used_count.get(res_bit.name, 1) == 1
                # force result precision everywhere if small
                or max_arity_use[res_bit.name] < bit_width < 7
            )
            if use_bit_width:
                max_arity_use[res_bit.name] = bit_width
            else:
                # do not let big precision results for big precision everywhere
                pass
    return max_arity_use


def compute_inferred_bit_width(
    circuit: TluCircuit,
    params: 'dict[str, ValueDescription | list[ValueDescription]]',
    scheduled_nodes,
    bit_location,
    used_count,
):
    """Compute the bitwidth of all values."""
    max_arity_use = compute_max_arity_use(
        circuit, params, scheduled_nodes, bit_location, used_count
    )
    bit_width_unify = set()
    unified = []
    for tlu in scheduled_nodes:
        constant = len(tlu.arguments) == 0
        if constant:
            continue
        no_tlu = len(tlu.arguments) == 1 and tlu.content in ([0, 1], [1, 0])
        if no_tlu:
            input_name = tlu.arguments[0].name
            output_name = tlu.results[0].name
            unified.append((input_name, output_name))
            bit_width_unify.add(tuple(sorted((input_name, output_name))))
        else:
            bit_width_unify.add(tuple(sorted([bit.name for bit in tlu.arguments])))

    changed = True
    inferred_bit_width = dict(max_arity_use)
    while changed:
        changed = False
        for bits_to_unify in bit_width_unify:
            max_arity_uses = [inferred_bit_width[bit] for bit in bits_to_unify]
            bit_width_max = max(max_arity_uses)
            for bit, v in zip(bits_to_unify, max_arity_uses):
                if v != bit_width_max:
                    inferred_bit_width[bit] = bit_width_max
                    changed = True
    for input_name, output_name in unified:
        max_max = max(inferred_bit_width[input_name], inferred_bit_width[output_name])
        inferred_bit_width[input_name] = max_max
        inferred_bit_width[output_name] = max_max
    return inferred_bit_width, unified


@dataclass
class BitLocation:
    """Express a bit position in a multi-word circuit input/output."""

    word: int
    """Word position."""
    local_bit_index: int
    """Bit position in the word."""
    global_bit_index: int
    """Bit position in the global value."""


def compute_bit_location(circuit: TluCircuit, params) -> 'dict[str, dict[str, BitLocation]]':
    """Compute the bit location of all inputs/outputs."""
    bit_location: dict[str, dict[str, BitLocation]] = {}
    for name in circuit.parameters:
        ty_l: list = params[name]  # type: ignore
        if not isinstance(params[name], list):
            ty_l = [params[name]]  # type: ignore
        bit_location[name] = {}
        g_index = 0
        for i_word, type_ in enumerate(ty_l):
            for bit_index in range(type_.dtype.bit_width):  # type: ignore
                bit = circuit.parameters[name][g_index]
                assert bit.origin.bit_index == g_index
                bit_location[name][bit.name] = BitLocation(
                    word=i_word,
                    local_bit_index=bit_index,
                    global_bit_index=g_index,
                )
                g_index += 1
    for result in circuit.results:
        name = result[0].origin.base_name
        ty_l = params[name]
        if not isinstance(ty_l, list):
            ty_l = [ty_l]
        bit_location[name] = {}
        g_index = 0
        for i_word, type_ in enumerate(ty_l):
            for bit_index in range(type_.dtype.bit_width):
                bit = result[g_index]
                assert bit.origin.bit_index == g_index
                bit_location[name][bit.name] = BitLocation(
                    word=i_word,
                    local_bit_index=bit_index,
                    global_bit_index=g_index,
                )
                g_index += 1

    return bit_location


def compute_min_scale(
    circuit: TluCircuit,
    scheduled_nodes,
    bit_location: 'dict[str, dict[str, BitLocation]]',
    unified: 'list[tuple[str, str]]',
    inferred_bit_width: 'dict[str, int]',
):
    """Compute the minimum weight for all value."""
    min_scale = {}
    for result in circuit.results:
        for res_bit in result:
            base_name = res_bit.origin.base_name
            local_bit_index = bit_location[base_name][res_bit.name].local_bit_index
            min_scale[res_bit.name] = 2**local_bit_index
    for tlu in scheduled_nodes:
        for i, value in enumerate(tlu.arguments):
            if all(value.name != name for name, _ in unified) or len(tlu.arguments) > 1:
                min_scale[value.name] = min(min_scale.get(value.name, 2**i), 2**i)
        for value in tlu.results:
            assert value.name not in min_scale or value.origin.is_result
            if not value.origin.is_result:
                min_scale[value.name] = 2 ** (inferred_bit_width[value.name] - 1)
    if not WEIGHT_TO_TLU:
        for n in min_scale:
            min_scale[n] = 1

    for input_name, output_name in unified:
        min1 = min_scale.get(input_name)
        min2 = min_scale[output_name]
        if min1 is None:
            min1 = min2
        min_min = min(min1, min2)
        min_scale[input_name] = min_min
        min_scale[output_name] = min_min

    return min_scale


def repack_scaled_bits(
    context: 'mlir_context.Context',
    scaled_bit_values: 'Iterable[tuple[int, Conversion]]',
    before_tlu: bool = True,
):
    """Recombine all binary value to an integer value."""
    max_weight = 0
    scaled_bit_values = list(scaled_bit_values)
    assert scaled_bit_values
    bit_width = len(scaled_bit_values)
    assert bit_width > 0
    repacked_bits = None
    arg_max_bit_width = 0
    constants_sum = 0
    for i, (scale, value) in enumerate(scaled_bit_values):
        if i == 0:
            assert scale == 1
        if value is None:
            assert scale == 0
            continue
        assert scale >= 1
        assert scale <= 2**i
        assert scale & (scale - 1) == 0  # is power of 2
        weight = 2**i // scale
        max_weight = max(max_weight, weight)
        if isinstance(value, Conversion) and isinstance(value.result, arith.ConstantOp):
            # clear mul-mul is not supported
            constants_sum += int(str(value.result.attributes["value"]).split(":")[0]) * weight
            continue
        if weight == 1:
            add = value
        else:
            weight = context.constant(context.i(value.type.bit_width + 1), weight)
            add = context.mul(value.type, value, weight)
        if add.type.bit_width > bit_width:
            add = context.safe_reduce_precision(add, bit_width)
        elif add.type.bit_width < bit_width:
            add_type = context.fork_type(add.type, bit_width=bit_width)
            add = context.tlu(
                add_type,
                add,
                [min(v, 2**bit_width - 1) for v in range(2**add.type.bit_width)],
            )
        assert add.type.bit_width == bit_width
        if repacked_bits is None:
            repacked_bits = add
        else:
            assert repacked_bits.type.bit_width == add.type.bit_width, (
                repacked_bits.type.bit_width,
                add.type.bit_width,
                scaled_bit_values,
            )
            repacked_bits = context.add(add.type, repacked_bits, add)
    assert repacked_bits is not None
    if constants_sum != 0:
        constant = context.constant(context.i(repacked_bits.type.bit_width + 1), constants_sum)
        repacked_bits = context.add(repacked_bits.type, repacked_bits, constant)
    extra_bits = arg_max_bit_width - bit_width
    if extra_bits > 0 and before_tlu and RESCALE_BEFORE_TLU:
        repacked_bits = context.safe_reduce_precision(repacked_bits, bit_width)
    return repacked_bits


def layered_nodes(nodes: 'list[TluNode]'):
    """
    Group nodes in layers by readyness (ready to be computed).
    """
    waiting: 'dict[str, list[int]]' = {}
    for i, tlu_node in enumerate(nodes):
        for arg in tlu_node.arguments:
            waiting.setdefault(arg.name, []).append(i)
    readyness = [
        sum(not (node.origin.is_parameter) for node in tlu_node.arguments) for tlu_node in nodes
    ]
    nb_nodes = len(nodes)
    ready_nodes = [i for i, count in enumerate(readyness) if count == 0]
    layers = []
    while nb_nodes > 0:
        assert ready_nodes, (nb_nodes, readyness)
        new_ready_nodes = []
        for i_node in ready_nodes:
            for result in nodes[i_node].results:
                for index in waiting.get(result.name, ()):
                    readyness[index] -= 1
                    assert readyness[index] >= 0
                    if readyness[index] == 0:
                        new_ready_nodes.append(index)
        layers += [ready_nodes]
        nb_nodes -= len(ready_nodes)
        ready_nodes = new_ready_nodes
    return layers


def tlu_circuit_to_mlir(
    circuit: TluCircuit,
    params: 'dict[str, ValueDescription | list[ValueDescription]]',
    result_name: str,
    verbose: bool,
):
    """Convert the simple TLU dag to a Tracer function."""
    layers = layered_nodes(circuit.nodes)
    scheduled_nodes = [circuit.nodes[index_node] for layer in layers for index_node in layer]

    if verbose:
        print("Layers")
        for i, layer in enumerate(layers):
            arities = [len(circuit.nodes[node_index].arguments) for node_index in layer]
            print(f"Layer {i}")
            print(f" {arities}")
            print(f" nb luts: {len(layer)}")

    # positions of bits, useful for multi-word inputs/outputs
    bit_location = compute_bit_location(circuit, params)

    # detect single use case
    used_count: Counter = Counter()
    used_count.update(tlu_arg.name for tlu in scheduled_nodes for tlu_arg in tlu.arguments)
    used_count.update(res_bit.name for result in circuit.results for res_bit in result)

    # inferred bit width
    inferred_bit_width, unified = compute_inferred_bit_width(
        circuit, params, scheduled_nodes, bit_location, used_count
    )

    # collect min scale use for all values
    min_scale = compute_min_scale(
        circuit, scheduled_nodes, bit_location, unified, inferred_bit_width
    )

    def mlir(context: mlir_context.Context, _resulting_type, args: 'list[Conversion]'):
        if len(circuit.parameters) != len(args):
            msg = "Invalid number of args"
            raise ValueError(msg)

        kwargs = {name: arg for name, arg in zip(circuit.parameters, args)}

        for name, value in kwargs.items():
            if name in params:
                if isinstance(params[name], list):
                    if not value.type.is_tensor:
                        msg = f"`{name}` should be a tensor"
                        raise TypeError(msg)

        # decompose parameters into bits
        parameters = {}
        for name, value in kwargs.items():
            unsigned = context.to_unsigned(value)
            for bit, bit_loc in zip(circuit.parameters[name], bit_location[name].values()):
                word_i = bit_loc.word
                word_bit_i = bit_loc.local_bit_index
                assert bit_loc.global_bit_index == bit.origin.bit_index
                if isinstance(params[name], list):
                    word_value_type = context.fork_type(
                        value.type, shape=tuple(value.type.shape)[:-1]
                    )
                    word_value = context.index(word_value_type, value, [word_i])
                else:
                    assert word_i == 0
                    word_value = unsigned
                min_scale_bit = min_scale.get(bit.name)
                if min_scale_bit is None:
                    # bit is unused
                    continue
                shift_by = power_of_2_scale(min_scale[bit.name])
                # Take into account word_bit_i
                bit_width = inferred_bit_width[bit.name] - shift_by
                # shift_by
                bit_value_type = context.fork_type(word_value.type, bit_width=bit_width)
                bit_value = context.extract_bits(
                    bit_value_type,
                    word_value,
                    word_bit_i,
                    assume_many_extract=True,
                    refresh_all_bits=True,
                )
                if shift_by > 0:
                    bit_value_type = context.fork_type(
                        bit_value_type, bit_width=inferred_bit_width[bit.name]
                    )
                    bit_value = context.reinterpret(
                        bit_value, bit_width=inferred_bit_width[bit.name]
                    )
                parameters[bit.name] = bit_value

        # will contains all intermediate ciphertext
        intermediate_values = dict(parameters)

        # handle special case first
        # constant and identity tlu
        for tlu_node in scheduled_nodes:
            output_name = tlu_node.results[0].name
            assert len(tlu_node.results) == 1
            rescale = min_scale[output_name]
            if len(tlu_node.results) == 1 and tlu_node.content == [0, 1]:
                assert len(tlu_node.arguments) == 1
                assert any(tlu_node.arguments[0].name == name for name, _ in unified)
                assert any(tlu_node.results[0].name == name for _, name in unified)
                intermediate_values[output_name] = intermediate_values[tlu_node.arguments[0].name]
                assert (
                    intermediate_values[tlu_node.arguments[0].name].type.bit_width
                    == inferred_bit_width[output_name]
                )
            elif len(tlu_node.results) == 1 and tlu_node.content == [1, 0]:
                conv_arg = intermediate_values[tlu_node.arguments[0].name]
                c_1_type = context.i(conv_arg.type.bit_width)
                c_1 = context.constant(c_1_type, 1)
                rev_conv_arg = context.sub(conv_arg.type, c_1, conv_arg)
                intermediate_values[output_name] = rev_conv_arg
            elif len(tlu_node.arguments) == 0:
                bit_type = context.i(inferred_bit_width[output_name])  # TODO: tensor input
                if tlu_node.content == ["0"]:
                    intermediate_values[output_name] = context.constant(bit_type, 0)
                elif tlu_node.content == ["1"]:
                    intermediate_values[output_name] = context.constant(bit_type, rescale)
                else:
                    msg = "Unknown Constant TLU content"
                    raise ValueError(msg)
            else:
                continue

        # apply all tlus
        for tlu_node in scheduled_nodes:
            output_name = tlu_node.results[0].name
            if output_name in intermediate_values:
                continue
            assert len(tlu_node.arguments) > 1
            repacked_bits = repack_scaled_bits(
                context,
                (
                    (min_scale[arg.name], intermediate_values[arg.name])
                    for arg in tlu_node.arguments
                ),
            )
            rescale = min_scale[output_name]
            flat_content = np.array(tlu_node.content).reshape(-1)
            rescaled_content = [v * rescale for v in flat_content]
            max_precision = inferred_bit_width[output_name]
            assert max_precision
            result_type = context.fork_type(repacked_bits.type, bit_width=max_precision)
            result = context.tlu(result_type, repacked_bits, rescaled_content, no_synth=True)
            intermediate_values[output_name] = result

        # recompose bits into result
        results = []
        for result in circuit.results:
            bits = [
                (min_scale[res_bit.name], intermediate_values[res_bit.name]) for res_bit in result
            ]
            ty_result = params[result_name]
            if ty_result and isinstance(ty_result, list):
                words = []
                for word_ty in ty_result:
                    word_bits = bits[: word_ty.dtype.bit_width]  # type: ignore
                    bits = bits[word_ty.dtype.bit_width :]  # type: ignore
                    word_result = repack_scaled_bits(context, word_bits, before_tlu=False)
                    if word_ty.dtype.is_signed:  # type: ignore
                        word_result = context.to_signed(word_result)
                    words += [word_result]
                results += [words]
            else:
                ty_result = cast(ValueDescription, ty_result)
                result = repack_scaled_bits(context, bits, before_tlu=False)
                assert isinstance(ty_result.dtype, Integer)
                if ty_result.dtype.is_signed:  # type: ignore
                    result = context.to_signed(result)
                results += [result]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    return mlir
