"""Provide helper function to generate verilog source code."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Ty:
    """Type of arguments and result of a verilog module."""

    bit_width: int
    is_signed: bool


def signed(b):
    """Textual signed attribute."""
    return "signed" if b else ""


def verilog_from_expression(params: Dict[str, Ty], operation: str, result_name: str) -> str:
    """Create a verilog module source from params specification (including result), operation."""

    out_bit_width = params[result_name].bit_width
    out_signed = params[result_name].is_signed
    params = ", ".join(
        f"input {signed(ty.is_signed)} [0:{ty.bit_width - 1}] {name}"
        for name, ty in params.items()
        if name != result_name
    )
    return f"""\
module main({params}, output {signed(out_signed)} [0:{out_bit_width-1}] {result_name});
  assign {result_name} = {operation};
endmodule
"""


def verilog_from_tlu(table: List[int], signed_input=False, out_type=None) -> Tuple[str, Ty]:
    """Create a verilog module source doing the table lookup in table."""
    assert not signed_input
    table = list(np.array(table).reshape(-1))
    max_table = max(table)
    precision_a = math.ceil(math.log2(len(table)))
    if out_type is None:
        out_type = Ty(
            bit_width=max(1, math.ceil(math.log2(max(1, max_table)))),
            is_signed=False,
        )
    expr = "         " + "        :".join(
        [f"(a == {i}) ? {v}\n" for i, v in enumerate(table)] + [f"{max_table}"]
    )
    return (
        f"""\
module main(a, result);
    input[0:{precision_a-1}] a;
    output {signed(out_type.is_signed)} [0:{out_type.bit_width-1}] result;
    assign result = (\n{expr}
    );
endmodule\
""",
        out_type,
    )
