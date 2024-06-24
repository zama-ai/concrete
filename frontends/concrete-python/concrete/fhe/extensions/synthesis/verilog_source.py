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
    expr = "                  " + "        :".join(
        [f"(a == {i}) ? {v}\n" for i, v in enumerate(table)] + [f"{max_table}"]
    )
    BLOCK_SIZE = 256
    blocks = []
    for i_block in range(0, 1 + len(table) // BLOCK_SIZE):
        sub_table = table[i_block*BLOCK_SIZE:(i_block+1)*BLOCK_SIZE]
        if not sub_table:
            break
        # TODO: compress table
        expr = "                 " + "                :".join(
            [f"(a == {i}) ? {v}\n" for i, v in enumerate(sub_table[:-1], i_block*BLOCK_SIZE)] + [str(sub_table[-1])]
        )
        blocks.append(expr)
    if len(blocks) == 1:
        blocks = blocks[0]
    else:
        blocks = "        " + ":\n        ".join([
            f"(a <= {(i+1)*BLOCK_SIZE-1}) ? (\n{block}\n        )"
            for i, block in enumerate(blocks[:-1])
        ] + ["\n" + blocks[-1]])
    return (
        f"""\
module main(a, result);
    input[0:{precision_a-1}] a;
    output {signed(out_type.is_signed)} [0:{out_type.bit_width-1}] result;
    assign result = (\n{blocks}
    );
endmodule\
""",
        out_type,
    )

# def verilog_from_tlu(table: List[int], signed_input=False, out_type=None) -> Tuple[str, Ty]:
#     """Create a verilog module source doing the table lookup in table."""
#     assert not signed_input
#     table = list(np.array(table).reshape(-1))
#     max_table = max(table)
#     precision_a = math.ceil(math.log2(len(table)))
#     if out_type is None:
#         out_type = Ty(
#             bit_width=max(1, math.ceil(math.log2(max(1, max_table)))),
#             is_signed=False,
#         )
#     expr = "         " + "        :".join(
#         [f"(a == {i}) ? {v}\n" for i, v in enumerate(table)] + [f"{max_table}"]
#     )
#     src = f"""\
# module main(a, result);
#     input[0:{precision_a-1}] a;
#     output {signed(out_type.is_signed)} [0:{out_type.bit_width-1}] result;
#     case (a)
# """
#     zero_pad = "0" * precision_a
#     for i, v in enumerate(table):
#         src += f"""\
#         {precision_a}b'{(zero_pad + bin(i)[2:])[-precision_a:]} :  assign result = ;
# """
#     # {precision_a}b'{(zero_pad + bin(i)[2:])[-precision_a:]}    : result <= 1b'1;
#     src += f"""\
#         // default: result = 0;
#     endcase
# endmodule\
# """
#     print(src)
#     return src, out_type