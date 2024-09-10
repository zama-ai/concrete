"""
INTERNAL extension to synthesize a fhe compatible function from verilog code.
"""

from __future__ import annotations
from collections import Counter
from typing import Optional

import concrete.fhe.dtypes as fhe_dtypes
import concrete.fhe.tracing.typing as fhe_typing
from concrete.fhe.dtypes.integer import Integer
from concrete.fhe.extensions.synthesis.eval_context import EvalContext
from concrete.fhe.extensions.synthesis.luts_to_fhe import tlu_circuit_to_mlir
from concrete.fhe.extensions.synthesis.luts_to_graph import to_graph
from concrete.fhe.extensions.synthesis.verilog_source import (
    Ty,
    verilog_from_expression,
    verilog_from_tlu,
)
from concrete.fhe.extensions.synthesis.verilog_to_luts import yosys_lut_synthesis
from concrete.fhe.values.value_description import ValueDescription


class FheFunction:
    """Main class to synthesize verilog."""

    def __init__(
        self,
        *,
        verilog,
        name,
        params=None,
        result_name="result",
        yosys_dot_file=False,
        verbose=False,
    ):
        assert params
        self.name = name
        self.verilog = verilog
        if verbose:
            print()
            print(f"Verilog, {name}:")
            print(verilog)
            print()
        if verbose:
            print("Synthesis")
        self.circuit = yosys_lut_synthesis(
            verilog, yosys_dot_file=yosys_dot_file, circuit_name=name
        )
        if verbose:
            print()
            print(f"TLUs counts, {self.tlu_counts()}:")
            print()
        self.params = params
        self.result_name = result_name

        self.mlir = tlu_circuit_to_mlir(self.circuit, self.params, result_name, verbose)

    def __call__(self, **kwargs):
        """
        Evaluate using mlir generation with a direct evaluation context.

        This is useful for testing purpose.
        """
        args = []
        for name, type_ in self.params.items():
            if name == "result":
                continue
            if isinstance(type_, list):
                val = EvalContext.Val(
                    kwargs[name], EvalContext.Ty(type_[0].dtype.bit_width, is_tensor=True)
                )
            else:
                val = EvalContext.Val(kwargs[name], EvalContext.Ty(type_.dtype.bit_width))
            args.append(val)
        result_ty = self.params["result"]
        if isinstance(result_ty, list):
            eval_ty = EvalContext.Ty(result_ty, is_tensor=True)
        else:
            eval_ty = EvalContext.Ty(result_ty, is_tensor=False)
        result = self.mlir(EvalContext(), eval_ty, args)
        if isinstance(result_ty, list):
            return [r.value for r in result]
        else:
            return result.value

    def tlu_counts(self):
        """Count the number of tlus in the synthesized tracer keyed by input precision."""
        counter = Counter()
        for node in self.circuit.nodes:
            if len(node.arguments) == 1:
                print(node)
            counter.update({len(node.arguments): 1})
        return dict(sorted(counter.items()))

    def is_faster_than_1_tlu(self, reference_costs):
        """Verify that synthesis is faster than the original tlu."""
        costs = 0
        for node in self.circuit.nodes:
            zero_cost = len(node.arguments) <= 1
            if zero_cost:
                # constant or inversion (converted to substraction)
                continue
            else:
                costs += reference_costs[len(node.arguments)]
        try:
            return costs <= reference_costs[self.params["a"].dtype.bit_width]
        except KeyError:
            return True

    def graph(self, *, filename=None, view=True, **kwargs):
        """Render the synthesized tracer as a graph."""
        graph = to_graph(self.name, self.circuit.nodes)
        graph.render(filename=filename, view=view, cleanup=filename is None, **kwargs)


def lut(table: 'list[int]', out_type: Optional[ValueDescription] = None, **kwargs):
    """Synthesize a lookup function from a table."""
    # assert not signed # TODO signed case
    if isinstance(out_type, list):
        msg = "Multi-message output is not supported"
        raise TypeError(msg)
    if out_type:
        assert isinstance(out_type.dtype, Integer)
        v_out_type = Ty(
            bit_width=out_type.dtype.bit_width,
            is_signed=out_type.dtype.is_signed,
        )
    verilog, v_out_type = verilog_from_tlu(table, signed_input=False, out_type=v_out_type)
    if "name" not in kwargs:
        kwargs.setdefault("name", "lut")
    if "params" not in kwargs:
        dtype = fhe_dtypes.Integer.that_can_represent(len(table) - 1)
        a_ty = getattr(fhe_typing, f"uint{dtype.bit_width}")
        assert a_ty
        kwargs["params"] = {"a": a_ty, "result": out_type}
    return FheFunction(verilog=verilog, **kwargs)


def _uniformize_as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]


def verilog_expression(
    in_params: 'dict[str, ValueDescription]', expression: str, out_type: ValueDescription, **kwargs
):
    """Synthesize a lookup function from a verilog function."""
    result_name = "result"
    if result_name in in_params:
        result_name = f"{result_name}_{hash(expression)}"
    in_params = dict(in_params)
    in_params[result_name] = out_type
    verilog_params = {
        name: Ty(
            bit_width=sum(ty.dtype.bit_width for ty in _uniformize_as_list(type_list)),
            is_signed=any(ty.dtype.is_signed for ty in _uniformize_as_list(type_list)),
        )
        for name, type_list in in_params.items()
    }
    verilog = verilog_from_expression(verilog_params, expression, result_name)
    if "name" not in kwargs:
        kwargs.setdefault("name", expression)
    return FheFunction(verilog=verilog, params=in_params, result_name=result_name, **kwargs)
