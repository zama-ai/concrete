"""Provide helper function to generate verilog source code."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


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
    # note: verilog indexing is done with upper:lower (vs lower:upper) to avoid a bug in yosys
    assert not signed_input
    table = list(table)
    max_table = max(table)
    precision_a = math.ceil(math.log2(len(table)))
    if out_type is None:
        out_type = Ty(
            bit_width=max(1, math.ceil(math.log2(max(1, max_table)))),
            is_signed=False,
        )

    branching_bits = 2
    max_block_bits = 2

    def gen_radix_tree(table, depth=1, remaining_bits=None, bits_len=None):
        assert table
        if bits_len is None:
            bits_len = int(math.ceil(math.log2(len(table))))
        if remaining_bits is None:
            remaining_bits = bits_len
        if all(v == table[0] for v in table):
            return str(table[0])
        start_str = " " * (4 * depth)
        join_str = ":\n" + start_str
        if remaining_bits > max_block_bits:
            block_count = 2**branching_bits
            block_size = 2 ** (remaining_bits - branching_bits)
            blocks = []
            bits_checked = f"a[{remaining_bits-1}:{remaining_bits-branching_bits}]"
            for i_block in range(block_count):
                sub_table = table[i_block * block_size : (i_block + 1) * block_size]
                blocks.append(
                    gen_radix_tree(
                        sub_table,
                        depth=depth + 1,
                        remaining_bits=remaining_bits - branching_bits,
                        bits_len=bits_len,
                    )
                )
            return start_str + join_str.join(
                [
                    f"({bits_checked} == {bits_cond}) ? \n({block})\n"
                    for bits_cond, block in enumerate(blocks[:-1])
                ]
                + ["\n" + blocks[-1]]
            )

        # TODO: could compress a bit here, like simple linear cases
        count = Counter()
        count.update(table)
        # the most common result is put in a last else so no conditions need to be checked
        by_count_and_value = lambda t: (t[1] << branching_bits) + t[0]
        most_common_result = sorted(count.items(), key=by_count_and_value)[0][0]
        bits_checked = f"a[{remaining_bits-1}:0]"
        return start_str + join_str.join(
            [
                f"({bits_checked} == {bits_cond}) ? {value}"
                for bits_cond, value in enumerate(table)
                if value != most_common_result
            ]
            + [str(most_common_result)]
        )

    blocks = gen_radix_tree(table)
    return (
        f"""\
module main(a, result);
    input[{precision_a-1}:0] a;
    output {signed(out_type.is_signed)} [{out_type.bit_width-1}:0] result;
    assign result = (\n{blocks}
    );
endmodule\
""",
        out_type,
    )
