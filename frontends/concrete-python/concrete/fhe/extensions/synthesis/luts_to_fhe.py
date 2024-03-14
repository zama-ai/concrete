"""
Convert the simple Tlu Dag to a concrete-python Tracer function.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np

from concrete import fhe
from concrete.fhe.extensions.synthesis.verilog_to_luts import TluNode
from concrete.fhe.tracing.tracer import Tracer

WEIGHT_TO_TLU = True
ENFORCE_BITWDTH = True


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

    if verbose:
        print("Layers")
        for i, layer in enumerate(layers):
            arities = [len(circuit.nodes[node_index].arguments) for node_index in layer]
            print(f"Layer {i}")
            print(f" {arities}")
            print(f" nb luts: {len(layer)}")

    results_type = [params[result[0].base_name] for result in circuit.results]

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
    for result in circuit.results:
        for res_bit in result:
            max_arity_use[res_bit.name] = len(result)

    # collect min scale use for all values
    min_scale = {}
    for bits in circuit.parameters.values():
        for value in bits:
            min_scale[value.name] = 1
    for result in circuit.results:
        for i, res_bit in enumerate(result):
            min_scale[res_bit.name] = 2**i
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

    # stonger than hints
    def with_bit_width(tracer, bit_width):
        if not isinstance(tracer, Tracer):
            return tracer
        if skip_force_bit_witdth:
            return tracer
        tracer.output = deepcopy(tracer.output)
        tracer.output.dtype.bit_width = bit_width
        return tracer

    def with_result_type(tracer, out_type):
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

    def repack_scaled_bits(scaled_bit_values):
        scaled_bit_values = list(scaled_bit_values)
        bit_width = len(scaled_bit_values)
        assert bit_width > 0
        repacked_bits = None
        for i, (scale, value) in enumerate(scaled_bit_values):
            if value is None:
                assert scale == 0
                continue
            if isinstance(value, int) and value == 0:
                continue
            assert scale >= 1
            if repacked_bits is None:
                assert scale == 1
                repacked_bits = value
            else:
                assert scale <= 2**i
                assert scale & (scale - 1) == 0  # is power of 2
                weight = 2**i // scale
                repacked_bits += value if weight == 1 else value * weight
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

        # decompose parameters into bits
        with fhe.tag("bit_extractions"):
            parameters = {
                bit.name: with_bit_width(
                    fhe.bits(with_unsigned(value))[bit.origin.bit_index],
                    bit_width=max_arity_use[bit.name],
                )
                for name, value in kwargs.items()
                for bit in circuit.parameters[name]
            }
        # contains all intermediate tracer
        intermediate_values = dict(parameters)

        # handle special case first
        # constant tlu and identity
        for tlu_node in scheduled_nodes:
            assert len(tlu_node.results) == 1
            output_name = tlu_node.results[0].name
            if len(tlu_node.results) == 1 and tlu_node.content == [0, 1]:
                assert len(tlu_node.arguments) == 1
                intermediate_values[output_name] = intermediate_values[tlu_node.arguments[0].name]
            elif len(tlu_node.arguments) == 0:
                if tlu_node.content == ["0"]:
                    intermediate_values[output_name] = 0
                elif tlu_node.content == ["1"]:
                    intermediate_values[output_name] = 1
                else:
                    msg = "Unknown Constant TLU content"
                    raise ValueError(msg)

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
                rescale = min_scale[output_name]
                flat_content = np.array(tlu_node.content).reshape(-1)
                tlu = fhe.LookupTable([v * rescale for v in flat_content])
                max_precision = max_arity_use[output_name]
                if max_precision:
                    result = with_bit_width(tlu[repacked_bits], bit_width=max_precision)
                else:
                    result = tlu[repacked_bits]
                intermediate_values[output_name] = result

        with fhe.tag("bit_assemble"):
            # recompose bits into result
            results = tuple(
                repack_scaled_bits(
                    (min_scale[res_bit.name], intermediate_values[res_bit.name])
                    for res_bit in result
                )
                for result in circuit.results
            )
            for r_type in results_type:
                assert not isinstance(r_type, list)
            # Provide the right result type
            results = tuple(
                with_result_type(result, r_type) for result, r_type in zip(results, results_type)
            )
            if len(results) == 1:
                return results[0]
            return results

    return tracer
