"""
Tests of 'synthesis' extension.
"""

import itertools
import time

import pytest

import concrete.fhe.extensions.synthesis as synth
from concrete import fhe
from concrete.fhe.compilation.configuration import SynthesisConfig


def signed_range(p: int):
    """Range of value for a signed of precision `p`."""
    return list(range(0, 2 ** (p - 1))) + list(range(-(2 ** (p - 1)), 0))


def signed_modulo(v, p):
    """Modulo for signed value."""
    if v >= 2 ** (p - 1):
        return signed_modulo(v - 2**p, p)
    elif v < -(2 ** (p - 1)):
        return signed_modulo(v + 2**p, p)
    return v


def test_relu_correctness():
    """
    Check relu.
    """
    ty_in_out = fhe.int5
    params = {"a": ty_in_out}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, ty_in_out)

    def ref_relu(v):
        return v if v >= 0 else 0

    for a in signed_range(5):
        assert relu(a=a) == ref_relu(a), a


def test_signed_add_correctness():
    """
    Check signed add.
    """
    ty_in_out = fhe.int5
    params = {"a": ty_in_out, "b": ty_in_out}
    expression = "a + b"

    oper = synth.verilog_expression(params, expression, ty_in_out)
    r = signed_range(5)
    for a in r:
        for b in r:
            expected = signed_modulo(a + b, 5)
            assert oper(a=a, b=b) == expected


def test_unsigned_add_correctness():
    """
    Check unsigned add.
    """
    out = fhe.uint5
    params = {"a": out, "b": out}
    expression = "a + b"

    oper = synth.verilog_expression(params, expression, out)

    r = list(range(0, 2**out.dtype.bit_width))
    for a in r:
        for b in r:
            assert oper(a=a, b=b) == (a + b) % 2**5


def test_signed_mul_correctness():
    """
    Check signed mul.
    """
    out = fhe.int5
    params = {"a": out, "b": out}
    expression = "a * b"

    oper = synth.verilog_expression(params, expression, out)

    r = signed_range(5)
    for a in r:
        for b in r:
            expected = signed_modulo(a * b, 5)
            assert oper(a=a, b=b) == expected


def test_unsigned_mul_correctness():
    """
    Check unsigned mul.
    """
    out = fhe.uint5
    params = {"a": out, "b": out}
    expression = "a * b"

    oper = synth.verilog_expression(params, expression, out)

    r = list(range(0, 2**out.dtype.bit_width))
    for a in r:
        for b in r:
            expected = (a * b) % 2**out.dtype.bit_width
            assert oper(a=a, b=b) == expected


def test_unsigned_div():
    """
    Check unsigned div.
    """
    ty_in_out = fhe.uint4
    params = {"a": ty_in_out, "b": ty_in_out}
    expression = "a / b"

    inputset = list(itertools.product(list(range(2**ty_in_out.dtype.bit_width)), repeat=2))

    oper = synth.verilog_expression(params, expression, ty_in_out)

    for a, b in inputset:
        if b == 0:
            expected = 15  # infinity
        else:
            expected = a // b
        assert oper(a=a, b=b) == expected, (a, b)


def convert_to_radix(v, tys):
    """Convert integer to multi-word integer."""
    if not (isinstance(tys, list)):
        return v
    shift_left = 0
    signed = any(ty.dtype.is_signed for ty in tys)
    words = []
    for i, type_ in enumerate(tys):
        last = (i + 1) == len(tys)
        v_word = v >> shift_left
        if signed and last:
            assert -(2 ** (type_.dtype.bit_width - 1)) <= v_word < 2 ** (type_.dtype.bit_width - 1)
        else:
            if last:
                assert v_word < 2**type_.dtype.bit_width
            v_word = v_word % (2**type_.dtype.bit_width)
        shift_left += type_.dtype.bit_width
        words.append(v_word)
    return words


def test_input_radix_relu_correctness():
    """
    Check relu with input in radix encoding.
    """
    ty_out = fhe.int5
    ty_in = [fhe.uint3, fhe.int2]
    params = {"a": ty_in}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, ty_out)

    def ref_relu(v):
        return v if v >= 0 else 0

    r = signed_range(5)
    for a in r:
        a_words = convert_to_radix(a, ty_in)
        assert relu(a=a_words) == convert_to_radix(ref_relu(a), ty_out), a


def test_radix_signed_mul_correctness():
    """
    Check signed mul with radix encoding.
    """
    ty_out = [fhe.uint1, fhe.uint1, fhe.int3]
    a_ty_in = [fhe.uint3, fhe.int2]
    b_ty_in = [fhe.uint2, fhe.int3]
    params = {"a": a_ty_in, "b": b_ty_in}
    expression = "a * b"

    oper = synth.verilog_expression(params, expression, ty_out)

    r = signed_range(5)
    for a in r:
        for b in r:
            a_words = convert_to_radix(a, a_ty_in)
            b_words = convert_to_radix(b, b_ty_in)
            expected = convert_to_radix(signed_modulo(a * b, 5), ty_out)
            assert list(oper(a=a_words, b=b_words)) == expected
            # assert list(circuit.simulate(*a_words, *b_words)) == expected


def to_bits(v, size):
    """Integer to list of bits."""
    return [v >> i & 1 for i in range(size)]


def from_bits(v):
    """List of bits to integer."""
    return sum(b << i for i, b in enumerate(v))


@pytest.mark.parametrize(
    "bit_width,reverse_bits",
    [
        pytest.param(4, False),  # generate a ternary sequence
        pytest.param(8, True),  # generate a ternary sequence
        pytest.param(10, False),  # generate a ternary tree
        pytest.param(10, True),  # generate a ternary tree
    ],
)
def test_identity_tlu(bit_width, reverse_bits):
    """
    Check the simplest TLU, this gives synthesize to a wire circuit.

    Bits are reversed to ensure we workaround correctly a bug in yosys.
    """
    inputset = list(range(2**bit_width))

    tlu_content = list(range(2**bit_width))
    if reverse_bits:
        tlu_content = [from_bits(reversed(to_bits(v, bit_width))) for v in tlu_content]

    @fhe.compiler({"a": "encrypted"})
    def tlu(a):
        return fhe.LookupTable(tlu_content)[a]

    time_0 = time.time()
    conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
    circuit = tlu.compile(inputset, conf)
    time_1 = time.time()
    assert time_1 - time_0 < 20

    for a in inputset:
        expected = tlu_content[a]
        assert circuit.simulate(a) == expected

    if bit_width < 7:
        assert circuit.mlir.count("lsb") == bit_width, circuit.mlir
    assert circuit.mlir.count("table") == 0


def test_bit_const_tlu4():
    """
    Check a more complex TLU, 4bits.
    """

    bit_width = 4
    inputset = list(range(2**bit_width))

    @fhe.compiler({"a": "encrypted"})
    def tlu(a):
        return fhe.LookupTable([v - v % 2 for v in range(2**bit_width)])[a]

    conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
    circuit = tlu.compile(inputset)

    # it's not fast enough so synthesis is not kept
    assert circuit.mlir.count("lsb") == 0
    assert circuit.mlir.count("table") == 1

    # forcing to keep the synthesis
    conf = fhe.Configuration(
        synthesis_config=SynthesisConfig(start_tlu_at_precision=0, force_tlu_at_precision=0)
    )
    circuit = tlu.compile(inputset, conf)

    assert circuit.mlir.count("lsb") >= 4
    assert circuit.mlir.count("table") == 0

    for a in inputset:
        expected = a - (a % 2)
        assert circuit.simulate(a) == expected


def test_add_1_tlu4():
    """
    Check a more complex TLU, 4bits.
    """

    bit_width = 4
    inputset = list(range(2**bit_width))

    @fhe.compiler({"a": "encrypted"})
    def tlu(a):
        return fhe.LookupTable([v + 1 for v in range(2**bit_width)])[a]

    conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
    circuit = tlu.compile(inputset)

    # it's not fast enough so synthesis is not kept
    assert circuit.mlir.count("lsb") == 0
    assert circuit.mlir.count("table") == 1

    # forcing to keep the synthesis
    conf = fhe.Configuration(
        synthesis_config=SynthesisConfig(start_tlu_at_precision=0, force_tlu_at_precision=0)
    )
    circuit = tlu.compile(inputset, conf)

    assert circuit.mlir.count("lsb") >= 4
    assert circuit.mlir.count("table") == 4

    for a in inputset:
        expected = a + 1
        assert circuit.simulate(a) == expected
        # assert circuit.encrypt_run_decrypt(a) == expected


def test_div_tlu10():
    """
    Check a more complex tlu, 10bits.
    """

    bit_width = 5
    inputset = list(itertools.product(list(range(2**bit_width)), repeat=2))

    def div(a, b):
        return a // b if a and b else 0

    div_table = [div(a, b) for a in range(2**bit_width) for b in range(2**bit_width)]

    @fhe.compiler({"a": "encrypted", "b": "encrypted"})
    def tlu(a, b):
        return fhe.LookupTable(div_table)[a * 2**bit_width + b]

    # forcing to keep the synthesis
    conf = fhe.Configuration(
        synthesis_config=SynthesisConfig(start_tlu_at_precision=0, force_tlu_at_precision=0)
    )
    time_0 = time.time()
    circuit = tlu.compile(inputset, conf)
    time_1 = time.time()
    assert time_1 - time_0 < 120

    assert circuit.mlir.count("lsb") >= 2 * bit_width
    assert circuit.mlir.count("table") > 1
    for line in circuit.mlir.splitlines():
        if "table" not in line:
            continue
        tlu_bit_width = line.rsplit(":")[1].split("->")[0].split(",")[0].split("<")[1].strip(">")
        assert 1 < int(tlu_bit_width) <= 7

    testset = inputset
    for a, b in testset:
        expected = div(a, b)
        simu = circuit.simulate(a, b)
        assert simu == expected, (a, b, expected, simu)
