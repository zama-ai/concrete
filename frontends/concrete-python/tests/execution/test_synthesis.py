"""
Tests of 'synthesis' extension.
"""
import itertools
import pytest

import numpy as np

from concrete.fhe.compilation.configuration import SynthesisConfig
import concrete.fhe.extensions.synthesis as synth
from concrete import fhe


# def signed_range(p):
#     return list(range(0, 2**(p-1))) + list(range(-(2**(p-1)), 0))

# def signed_modulo(v, p):
#     if v >= 2**(p-1):
#         return signed_modulo(v - 2**p, p)
#     elif v < -2**(p-1):
#         return signed_modulo(v + 2**p, p)
#     return v

# def test_relu_correctness():
#     """
#     Check relu.
#     """
#     out = fhe.int5
#     params = {"a": out}
#     expression = "(a >= 0) ? a : 0"

#     relu = synth.verilog_expression(params, expression, out)

#     @fhe.circuit({"a": "encrypted"})
#     def circuit(a: out):
#         return relu(a=a)

#     def ref_relu(v):
#         return v if v >= 0 else 0

#     for a in signed_range(5):
#         assert relu(a=a) == ref_relu(a), a
#         assert circuit.simulate(a) == ref_relu(a), a

# def test_signed_add_correctness():
#     """
#     Check add.
#     """
#     out = fhe.int5
#     params = {"a": out, "b": out}
#     expression = "a + b"

#     op = synth.verilog_expression(params, expression, out)
#     @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#     def circuit(a: out, b: out):
#         return op(a=a, b=b)

#     r = signed_range(5)
#     for a in r:
#         for b in r:
#             expected = signed_modulo(a + b, 5)
#             assert op(a=a, b=b) == expected
#             assert circuit.simulate(a, b) == expected


# def test_unsigned_add_correctness():
#     """
#     Check add.
#     """
#     out = fhe.uint5
#     params = {"a": out, "b": out}
#     expression = "a + b"

#     op = synth.verilog_expression(params, expression, out)

#     @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#     def circuit(a: out, b: out):
#         return op(a=a, b=b)

#     r = list(range(0, 2**5))
#     for a in r:
#         for b in r:
#             assert op(a=a, b=b) == (a + b) % 2**5
#             assert circuit.simulate(a, b) == (a + b) % 2**5

# def test_signed_mul_correctness():
#     """
#     Check add.
#     """
#     out = fhe.int5
#     params = {"a": out, "b": out}
#     expression = "a * b"

#     op = synth.verilog_expression(params, expression, out)

#     @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#     def circuit(a: out, b: out):
#         return op(a=a, b=b)

#     r = signed_range(5)
#     for a in r:
#         for b in r:
#             expected = signed_modulo(a * b, 5)
#             assert op(a=a,b=b) == expected
#             assert circuit.simulate(a, b) == expected


# def test_unsigned_mul_correctness():
#     """
#     Check add.
#     """
#     out = fhe.uint5
#     params = {"a": out, "b": out}
#     expression = "a * b"

#     op = synth.verilog_expression(params, expression, out)

#     @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#     def circuit(a: out, b: out):
#         return op(a=a, b=b)

#     r = list(range(0, 2**5))
#     for a in r:
#         for b in r:
#             expected = (a * b) % 2**5
#             assert op(a=a, b=b) == expected
#             assert circuit.simulate(a, b) == expected

# def test_relu_limit():
#     """
#     Check the limit for relu precision.
#     28bits is only attainable only with weight in tlu fuzing.
#     """
#     out = fhe.int28
#     params = {"a": out}
#     expression = "(a >= 0) ? a : 0"

#     relu = synth.verilog_expression(params, expression, out)

#     @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#     def _(a: out, b: out):
#         v = relu(a=a) - relu(a=b)
#         return relu(a=v)

#     out = fhe.int29
#     params = {"a": out}
#     expression = "(a >= 0) ? a : 0"

#     relu = synth.verilog_expression(params, expression, out)

#     with pytest.raises(RuntimeError) as err:

#         @fhe.circuit({"a": "encrypted", "b": "encrypted"})
#         def _(a: out, b: out):
#             v = relu(a=a) - relu(a=b)
#             return relu(a=v)

#     assert str(err.value) == "NoParametersFound"


# def convert_to_radix(v, tys):
#     if not(isinstance(tys, list)):
#         return v
#     shift_left = 0
#     signed = any(ty.dtype.is_signed for ty in tys)
#     words = []
#     for i, ty in enumerate(tys):
#         last = (i + 1) == len(tys)
#         v_word = (v >> shift_left)
#         if signed and last:
#             assert -(2**(ty.dtype.bit_width - 1)) <= v_word < 2**(ty.dtype.bit_width - 1)
#         else:
#             if last:
#                 assert v_word < 2 ** ty.dtype.bit_width
#             v_word = v_word % (2 ** ty.dtype.bit_width)
#         shift_left += ty.dtype.bit_width
#         words.append(v_word)
#     return words

# def test_input_radix_relu_correctness():
#     """
#     Check relu with input in radix encoding.
#     """
#     ty_out = fhe.int5
#     ty_in = [fhe.uint3, fhe.int2]
#     params = {"a": ty_in}
#     expression = "(a >= 0) ? a : 0"

#     relu = synth.verilog_expression(params, expression, ty_out)

#     @fhe.circuit({"a0": "encrypted", "a1": "encrypted"})
#     def circuit(a0: ty_in[0], a1:ty_in[1]):
#         return relu(a=[a0, a1])

#     def ref_relu(v):
#         return v if v >= 0 else 0

#     r = reversed(list(range(0, 2**4)) + list(range(-(2**4), 0)))
#     for a in r:
#         a_words = convert_to_radix(a, ty_in)
#         assert relu(a=a_words) == convert_to_radix(ref_relu(a), ty_out), a
#         assert circuit.simulate(*a_words) == convert_to_radix(ref_relu(a), ty_out), a


# def test_radix_signed_mul_correctness():
#     """
#     Check signed mul with radix encoding.
#     """
#     ty_out = [fhe.uint1, fhe.uint1, fhe.int3]
#     a_ty_in = [fhe.uint3, fhe.int2]
#     b_ty_in = [fhe.uint2, fhe.int3]
#     params = {"a": a_ty_in, "b": b_ty_in}
#     expression = "a * b"

#     op = synth.verilog_expression(params, expression, ty_out)

#     conf = fhe.Configuration(approximate_rounding_config=fhe.ApproximateRoundingConfig(symetrize_deltas=False))

#     @fhe.circuit({"a0": "encrypted", "a1": "encrypted", "b0": "encrypted", "b1": "encrypted"}, conf)
#     def circuit(a0: a_ty_in[0], a1: a_ty_in[1], b0: b_ty_in[0], b1: b_ty_in[1]):
#         return tuple(op(a=[a0, a1], b=[b0, b1]))

#     r = signed_range(5)
#     for a in r:
#         for b in r:
#             a_words = convert_to_radix(a, a_ty_in)
#             b_words = convert_to_radix(b, b_ty_in)
#             expected = convert_to_radix(signed_modulo(a * b, 5), ty_out)
#             assert list(op(a=a_words,b=b_words)) == expected
#             assert list(circuit.simulate(*a_words, *b_words)) == expected


# def test_identity_tlu8():
#     """
#     Check the simplest TLU.
#     """
#     BIT_WIDTH = 8
#     inputset = list(range(2**BIT_WIDTH))

#     @fhe.compiler({"a": "encrypted"})
#     def tlu(a):
#         return fhe.LookupTable(list(range(2**BIT_WIDTH)))[a]

#     circuit = tlu.compile(inputset)

#     assert circuit.mlir.count("lsb") == 8
#     assert circuit.mlir.count("table") == 0

#     for a in inputset:
#         expected = a
#         assert circuit.simulate(a) == expected

# def test_identity_tlu4():
#     """
#     Check the simplest TLU.
#     """

#     BIT_WIDTH = 4
#     inputset = list(range(2**BIT_WIDTH))

#     @fhe.compiler({"a": "encrypted"})
#     def tlu(a):
#         return fhe.LookupTable(list(range(2**BIT_WIDTH)))[a]

#     circuit = tlu.compile(inputset)

#     assert circuit.mlir.count("lsb") == 0
#     assert circuit.mlir.count("table") == 1

#     conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
#     circuit = tlu.compile(inputset, conf)

#     assert circuit.mlir.count("lsb") == 4
#     assert circuit.mlir.count("table") == 0

#     for a in inputset:
#         expected = a
#         assert circuit.simulate(a) == expected
#         assert circuit.encrypt_run_decrypt(a) == expected

# def test_identity_tlu4_tensor():
#     """
#     Check the simplest TLU.
#     """

#     BIT_WIDTH = 4
#     inputset = [
#         np.array([v] * (2**BIT_WIDTH))
#         for v in range(2**BIT_WIDTH)
#     ]

#     @fhe.compiler({"a": "encrypted"})
#     def tlu(a):
#         return fhe.LookupTable(list(range(2**BIT_WIDTH)))[a]

#     circuit = tlu.compile(inputset)

#     assert circuit.mlir.count("lsb") == 0
#     assert circuit.mlir.count("table") == 1

#     conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
#     circuit = tlu.compile(inputset, conf)

#     assert circuit.mlir.count("lsb") == 4
#     assert circuit.mlir.count("table") == 0

#     a = np.array(list(range(2**BIT_WIDTH)))
#     expected = a
#     assert list(circuit.simulate(a)) == list(expected)
#     assert list(circuit.encrypt_run_decrypt(a)) == list(expected)


# def test_add_1_tlu4():
#     """
#     Check the simplest TLU.
#     """

#     BIT_WIDTH = 4
#     inputset = list(range(2**BIT_WIDTH))

#     @fhe.compiler({"a": "encrypted"})
#     def tlu(a):
#         return fhe.LookupTable([v +  1 for v in range(2**BIT_WIDTH)])[a]

#     conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0))
#     circuit = tlu.compile(inputset)

#     # it's not fast enough so synthesis is not kept
#     assert circuit.mlir.count("lsb") == 0
#     assert circuit.mlir.count("table") == 1

#     # forcing to keep it
#     conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0, force_tlu_at_precision=0))
#     circuit = tlu.compile(inputset, conf)

#     assert circuit.mlir.count("lsb") >= 4
#     assert circuit.mlir.count("table") == 4

#     for a in inputset:
#         expected = a + 1
#         assert circuit.simulate(a) == expected
#         assert circuit.encrypt_run_decrypt(a) == expected

# def test_div_1_tlu8():
#     """
#     Check the simplest TLU.
#     """

#     BIT_WIDTH = 4
#     inputset = list(itertools.product(list(range(2**BIT_WIDTH)), repeat=2))

#     def div(a, b):
#         return a // b if a and b else 0

#     @fhe.compiler({"a": "encrypted", "b": "encrypted"})
#     def tlu(a, b):
#         return fhe.LookupTable([
#             div(a, b)
#             for a in range(2**BIT_WIDTH)
#             for b in range(2**BIT_WIDTH)
#         ])[a * 2**BIT_WIDTH + b]


#     # forcing to keep the synthesis
#     conf = fhe.Configuration(synthesis_config=SynthesisConfig(start_tlu_at_precision=0, force_tlu_at_precision=0))
#     circuit = tlu.compile(inputset, conf)
#     print("NTABLE", circuit.mlir.count("table"))

#     assert circuit.mlir.count("lsb") >= 2 * BIT_WIDTH
#     # assert circuit.mlir.count("table") == 218
#     for l in circuit.mlir.splitlines():
#         if "table" in l:
#             tlu_bit_width = l.rsplit(":")[1].split("->")[0].split(",")[0].split("<")[1].strip(">")
#             assert 1 < int(tlu_bit_width) <= BIT_WIDTH
#     print(circuit.mlir)
#     import time

#     t0 = time.time()
#     for a, b in inputset:
#         expected = div(a, b)
#         assert circuit.simulate(a, b) == expected
#         t0b = time.time()
#         assert circuit.encrypt_run_decrypt(a, b) == expected
#         t1b = time.time()
#         print(t1b - t0b)
#     t1 = time.time()
#     print((t1-t0) / len(inputset), "div time")
#     print(len(inputset) / (t1-t0), "div/s")

def test_div():
    """
    Check add.
    """
    out = fhe.int4
    params = {"a": out, "b": out}
    expression = "a / b"

    inputset = list(itertools.product(list(range(2**out.dtype.bit_width)), repeat=2))

    op = synth.verilog_expression(params, expression, out)

    import time

    @fhe.compiler({"a": "encrypted", "b": "encrypted"})
    def circuit(a: out, b: out):
        return op(a=a, b=b)

    circuit = circuit.compile(inputset)

    print("NTABLE", circuit.mlir.count("table"))

    t0 = time.time()
    for a, b in inputset:
        if b == 0:
            continue
        if b == 15:
            continue
        expected = op(a=a, b=b)
        if expected == -1:
            continue
        simu = circuit.simulate(a, b)
        real = circuit.encrypt_run_decrypt(a, b)
        assert simu == real
        assert real == expected, (a,b)
        t0b = time.time()
        t1b = time.time()
        print(t1b - t0b)
    t1 = time.time()
    print((t1-t0) / len(inputset), "div time")
    print(len(inputset) / (t1-t0), "div/s")
