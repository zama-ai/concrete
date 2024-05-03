"""
Convert the simple Tlu Dag to a concrete-python Tracer function.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List

import numpy as np

from concrete import fhe
from concrete.fhe.extensions.synthesis.verilog_to_luts import TluNode
from concrete.fhe.tracing.tracer import Tracer

WEIGHT_TO_TLU = True
ENFORCE_BITWDTH = True
RESCALE_BEFORE_TLU = True

@dataclass
class BitLocation:
    word: int
    local_bit_index: int
    global_bit_index: int

def layered_nodes(nodes: List[TluNode]):
    """
    Group nodes in layers by readyness (ready to be computed).
    """
    waiting: Dict[str, List[int]] = {}
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


def tlu_circuit_to_fhe(circuit, params, verbose):
    """Convert the simple TLU dag to a Tracer function."""
    layers = layered_nodes(circuit.nodes)
    scheduled_nodes = [circuit.nodes[index_node] for layer in layers for index_node in layer]

    max_weight = 0
    if verbose:
        print("Layers")
        for i, layer in enumerate(layers):
            arities = [len(circuit.nodes[node_index].arguments) for node_index in layer]
            print(f"Layer {i}")
            print(f" {arities}")
            print(f" nb luts: {len(layer)}")

    used_count = {}
    for tlu in scheduled_nodes:
        for value in tlu.arguments:
            try:
                used_count[value.name] += 1
            except KeyError:
                used_count[value.name] = 1
    for result in circuit.results:
        for res_bit in result:
            try:
                used_count[res_bit.name] += 1
            except KeyError:
                used_count[res_bit.name] = 1

    bit_location = {}
    for name in circuit.parameters:
        ty_l = params[name]
        if not isinstance(ty_l, list):
            ty_l = [ty_l]
        bit_location[name] = {}
        g_index = 0
        for i_word, ty in enumerate(ty_l):
            for bit_index in range(ty.dtype.bit_width):
                bit = circuit.parameters[name][g_index]
                assert bit.origin.bit_index == g_index
                bit_location[name][bit.name] = BitLocation(
                    word=i_word,
                    local_bit_index=bit_index,
                    global_bit_index=g_index,
                )
                g_index += 1

    for result in circuit.results:
        name = result[0].base_name
        ty_l = params[name]
        if not isinstance(ty_l, list):
            ty_l = [ty_l]
        bit_location[name] = {}
        g_index = 0
        for i_word, ty in enumerate(ty_l):
            for bit_index in range(ty.dtype.bit_width):
                bit = result[g_index]
                assert bit.origin.bit_index == g_index
                bit_location[name][bit.name] = BitLocation(
                    word=i_word,
                    local_bit_index=bit_index,
                    global_bit_index=g_index,
                )
                g_index += 1

    # collect max arity use for all values
    max_arity_use = {}
    for bits in circuit.parameters.values():
        for value in bits:
            max_arity_use[value.name] = 0
    for tlu in scheduled_nodes:
        for value in tlu.arguments:
            max_arity_use[value.name] = max(max_arity_use[value.name], tlu.arity)
        for value in tlu.results:
            assert value.name not in max_arity_use
            max_arity_use[value.name] = 0

    # Result can be either precision correct on single use or < 7 or converted
    for result in circuit.results:
        for res_bit in result:
            if isinstance(params[res_bit.base_name], list):
                word_i = bit_location[res_bit.base_name][res_bit.name].word
                bit_width = params[res_bit.base_name][word_i].dtype.bit_width
            else:
                bit_width = params[res_bit.base_name].dtype.bit_width
            if used_count.get(res_bit.name, 1) == 1:
                # print("solo use", res_bit.name, bitwidth)
                max_arity_use[res_bit.name] = bit_width
            elif bit_width < 7 and bit_width > max_arity_use[res_bit.name]:
                # print("dominated use", res_bit.name, bitwidth)
                max_arity_use[res_bit.name] = bit_width
            else:
                # print("Warning: force conversion on", res_bit.name)
                pass

    # collect min scale use for all values
    min_scale = {}
    for bits in circuit.parameters.values():
        for value in bits:
            min_scale[value.name] = 1
    for result in circuit.results:
        for i, res_bit in enumerate(result):
            local_bit_index = bit_location[res_bit.base_name][res_bit.name].local_bit_index
            min_scale[res_bit.name] = 2 ** local_bit_index
    for tlu in scheduled_nodes:
        for i, value in enumerate(tlu.arguments):
            min_scale[value.name] = min(min_scale[value.name], 2**i)
        for value in tlu.results:
            assert value.name not in min_scale or value.origin.is_result
            if not value.origin.is_result:
                min_scale[value.name] = 2 ** (max_arity_use[value.name] - 1)
    if not WEIGHT_TO_TLU:
        for n in min_scale:
            min_scale[n] = 1

    # pylint: disable=protected-access
    skip_force_bit_witdth = not (Tracer._is_direct or ENFORCE_BITWDTH)

    def with_unsigned(tracer):
        """Transform a tracer to an unsigned one.

        Note: only works because thanks to bit extraction which it's annotate.
        """
        if not isinstance(tracer, Tracer):
            return tracer
        tracer.output = deepcopy(tracer.output)
        tracer.output.dtype.is_signed = False
        return tracer

    # stronger than hints
    def with_bit_width(tracer, bit_width):
        assert tracer is not None
        if not isinstance(tracer, Tracer):
            return tracer
        if skip_force_bit_witdth:
            return tracer
        tracer.output = deepcopy(tracer.output)
        tracer.output.dtype.bit_width = bit_width
        return tracer

    def with_result_type(tracer, out_type):
        assert isinstance(tracer, (int, Tracer)), type(tracer)
        if isinstance(tracer, int):
            if out_type.dtype.is_signed:
                if tracer >= 2**(out_type.dtype.bit_width-1):
                    return tracer - 2**out_type.dtype.bit_width
            return tracer
        if not isinstance(tracer, Tracer):
            return tracer
        assert not tracer.output.dtype.is_signed
        # a workaround for the missing to signed operator
        if out_type.dtype.is_signed:
            tracer = -(-tracer)
        tracer.output.dtype.is_signed = out_type.dtype.is_signed
        if skip_force_bit_witdth:
            return tracer
        tracer.output = deepcopy(tracer.output)
        tracer.output.dtype.bit_width = out_type.dtype.bit_width
        return tracer

    def reduce_precision_before_tlu(v, previous_bit_width, new_bit_width):
        # TODO: to be replaced by fhe.reduce_precision
        assert previous_bit_width > new_bit_width
        lsbs_to_remove = previous_bit_width - new_bit_width
        overflow_protection = False
        exactness = fhe.Exactness.APPROXIMATE
        v *= 2 ** lsbs_to_remove
        v = fhe.round_bit_pattern(v, lsbs_to_remove, overflow_protection, exactness)
        tbl = fhe.LookupTable([
            i // 2**lsbs_to_remove
            for i in range(2**previous_bit_width)
        ])
        v = with_bit_width(tbl[v], new_bit_width) # required, hint is not sufficient
        # we assume the tlu will be fuzed so it's free
        return v

    def repack_scaled_bits(scaled_bit_values, before_tlu=True):
        nonlocal max_weight
        scaled_bit_values = list(scaled_bit_values)
        assert scaled_bit_values
        bit_width = len(scaled_bit_values)
        assert bit_width > 0
        repacked_bits = None
        arg_max_bit_width = 0
        for i, (scale, value) in enumerate(scaled_bit_values):
            if i == 0:
                assert scale == 1
            if value is None:
                assert scale == 0
                continue
            if isinstance(value, int) and value == 0:
                continue
            assert scale >= 1
            assert scale <= 2**i
            assert scale & (scale - 1) == 0  # is power of 2
            weight = 2**i // scale
            max_weight = max(max_weight, weight)
            if isinstance(value, Tracer):
                arg_max_bit_width = max(arg_max_bit_width, value.output.dtype.bit_width)
                if value.output.dtype.bit_width < bit_width:
                    # print("oversizing", value.output.dtype.bit_width, bit_width)
                    value = fhe.hint(fhe.LookupTable(list(range(2**value.output.dtype.bit_width)))[value], bit_width=bit_width)
            add = value if weight == 1 else value * weight
            if repacked_bits is None:
                repacked_bits = add
            else:
                repacked_bits += add

        extra_bits = arg_max_bit_width - bit_width
        if repacked_bits is None:
            repacked_bits = 0
        elif extra_bits > 0 and before_tlu and RESCALE_BEFORE_TLU:
            repacked_bits = reduce_precision_before_tlu(repacked_bits, arg_max_bit_width, bit_width)
        return with_bit_width(repacked_bits, bit_width)


    def tracer(**kwargs):
        for name in circuit.parameters:
            if name not in kwargs:
                msg = f"{circuit.name}() has a missing keyword argument '{name}'"
                raise TypeError(msg)
        for name in kwargs:
            if name not in circuit.parameters:
                msg = f"{circuit.name}() got an unexpected keyword argument '{name}'"
                raise TypeError(msg)

        # TODO: uniformize to list
        # TODO: mapping i => (word_j, k)

        for name, value in kwargs.items():
            if name in params:
                if isinstance(params[name], list):
                    if isinstance(value, int):
                        msg = f"`{name}` should be a list or a tensor or an iterable"
                        raise TypeError(msg)
                else:
                    kwargs[name] = [value]

        all_input_in_clear = True
        # decompose parameters into bits
        with fhe.tag("bit_extractions"):
            # TODO: direct result, that are also reused should be shielded
            parameters = {}
            for name, value in kwargs.items():
                for bit, bit_loc in zip(circuit.parameters[name], bit_location[name].values()):
                    word_i = bit_loc.word
                    word_bit_i = bit_loc.local_bit_index
                    assert bit_loc.global_bit_index == bit.origin.bit_index
                    try:
                        word_value = value[word_i]
                    except IndexError:
                        bit_value = 0
                    else:
                        if not isinstance(word_value, int):
                            all_input_in_clear = False
                        bit_value = with_bit_width(
                            fhe.bits(with_unsigned(word_value))[word_bit_i],
                            bit_width=max_arity_use[bit.name],
                        )
                    parameters[bit.name] = bit_value
        # contains all intermediate tracer
        intermediate_values = dict(parameters)

        display = verbose and all_input_in_clear
        if display:
            print("input values:", intermediate_values)

        # handle special case first
        # constant tlu and identity
        for tlu_node in scheduled_nodes:
            assert len(tlu_node.results) == 1
            output_name = tlu_node.results[0].name
            rescale = min_scale[output_name]
            if len(tlu_node.results) == 1 and tlu_node.content == [0, 1]:
                assert len(tlu_node.arguments) == 1
                intermediate_values[output_name] = rescale * intermediate_values[tlu_node.arguments[0].name]
            elif len(tlu_node.arguments) == 0:
                if tlu_node.content == ["0"]:
                    intermediate_values[output_name] = 0
                elif tlu_node.content == ["1"]:
                    intermediate_values[output_name] = 1 * rescale
                else:
                    msg = "Unknown Constant TLU content"
                    raise ValueError(msg)
            else:
                continue
            if display:
                print(f"{output_name} = {intermediate_values[output_name]}")

        with fhe.tag("synthesis"):
            # apply all tlus
            for tlu_node in scheduled_nodes:
                output_name = tlu_node.results[0].name
                if output_name in intermediate_values:
                    continue
                assert tlu_node.arguments
                repacked_bits = repack_scaled_bits(
                    (min_scale[arg.name], intermediate_values[arg.name])
                    for arg in tlu_node.arguments
                )
                for arg in tlu_node.arguments:
                    assert max_arity_use[arg.name] < 7, (arg.name, max_arity_use[arg.name])
                rescale = min_scale[output_name]
                flat_content = np.array(tlu_node.content).reshape(-1)
                tlu = fhe.LookupTable([v * rescale for v in flat_content])
                max_precision = max_arity_use[output_name]
                result = tlu[repacked_bits]
                if not isinstance(result, Tracer):
                    result = int(result)
                if max_precision:
                    result = with_bit_width(result, bit_width=max_precision)
                else:
                    print("UNKOWN P FOR ", output_name)
                intermediate_values[output_name] = result
                if display:
                    print(f"{output_name} = {intermediate_values[output_name]}")

        with fhe.tag("bit_assemble"):
            # recompose bits into result
            results = []
            for result in circuit.results:
                bits = [
                    (min_scale[res_bit.name], intermediate_values[res_bit.name])
                    for res_bit in result
                ] # TODO: rely on order, add check
                ty_result = params.get("result")
                if ty_result and isinstance(ty_result, list):
                    words = []
                    for word_ty in ty_result:
                        word_bits = bits[:word_ty.dtype.bit_width]
                        bits = bits[word_ty.dtype.bit_width:]
                        word_result = repack_scaled_bits(word_bits, before_tlu=False)
                        word_result = with_result_type(word_result, word_ty)
                        words += [word_result]
                    results += [words]
                else:
                    result = repack_scaled_bits(bits, before_tlu=False)
                    result = with_result_type(result, ty_result)
                    results += [result]
            if len(results) == 1:
                return results[0]
            return tuple(results)

    return tracer
