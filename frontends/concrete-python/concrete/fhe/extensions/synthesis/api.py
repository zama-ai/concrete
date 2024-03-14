"""
EXPERIMENTAL extension to synthesize a fhe compatible function from verilog code.

The resulting object can be used directly as a python function.
For instance you can write a relu function using:

    out = fhe.int5
    params = {"a": out}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, out)

    @fhe.circuit({"a": "encrypted"})
    def circuit(a: out):
        return relu(a=a)

"""
from collections import Counter

from concrete.fhe.extensions.synthesis.luts_to_fhe import tlu_circuit_to_fhe
from concrete.fhe.extensions.synthesis.luts_to_graph import to_graph
from concrete.fhe.extensions.synthesis.verilog_source import (
    Ty,
    verilog_from_expression,
    verilog_from_tlu,
)
from concrete.fhe.extensions.synthesis.verilog_to_luts import yosys_lut_synthesis


class FheFunction:
    """Main class to synthesize verilog to tracer function."""

    def __init__(
        self,
        *,
        verilog,
        name,
        params=None,
        yosys_dot_file=False,
        verbose=False,
    ):
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
        self.tracer = tlu_circuit_to_fhe(self.circuit, self.params, verbose)

    def __call__(self, **kwargs):
        """Call the tracer function."""
        return self.tracer(**kwargs)

    def tlu_counts(self):
        """Count the number of tlus in the synthesized tracer keyed by input precision."""
        counter = Counter()
        for node in self.circuit.nodes:
            if len(node.arguments) == 1:
                print(node)
            counter.update({len(node.arguments): 1})
        return dict(sorted(counter.items()))

    def graph(self, *, filename=None, view=True, **kwargs):
        """Render the synthesized tracer as a graph."""
        graph = to_graph(self.name, self.circuit.nodes)
        graph.render(filename=filename, view=view, cleanup=filename is None, **kwargs)


def lut(table, out_type=None, **kwargs):
    """Synthesize a lookup function from a table."""
    # assert not signed # TODO signed case
    if isinstance(out_type, list):
        msg = "Multi-message output is not supported"
        raise TypeError(msg)
    if out_type:
        out_type = Ty(
            bit_width=out_type.dtype.bit_width,
            is_signed=out_type.dtype.is_signed,
        )
    verilog, out_type = verilog_from_tlu(table, signed_input=False, out_type=out_type)
    if "name" not in kwargs:
        kwargs.setdefault("name", "lut")
    return FheFunction(verilog=verilog, **kwargs)


def _uniformize_as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]


def verilog_expression(params, expression, out_type, **kwargs):
    """Synthesize a lookup function from a verilog function."""
    result_name = "result"
    if result_name in params:
        result_name = f"{result_name}_{hash(expression)}"
    params = dict(params)
    params[result_name] = out_type
    verilog_params = {
        name: Ty(
            bit_width=sum(ty.dtype.bit_width for ty in _uniformize_as_list(type_list)),
            is_signed=_uniformize_as_list(type_list)[0].dtype.is_signed,
        )
        for name, type_list in params.items()
    }
    verilog = verilog_from_expression(verilog_params, expression, result_name)
    if "name" not in kwargs:
        kwargs.setdefault("name", expression)
    return FheFunction(verilog=verilog, params=params, **kwargs)


def verilog_module(verilog, **kwargs):
    """Synthesize a lookup function from a verilog module."""
    if "name" not in kwargs:
        kwargs.setdefault("name", "main")
    return FheFunction(verilog=verilog, **kwargs)
