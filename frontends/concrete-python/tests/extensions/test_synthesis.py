"""
Tests of 'synthesis' extension.
"""

import pytest

import concrete.fhe.extensions.synthesis as synth
from concrete import fhe


def test_relu_correctness():
    """
    Check relu.
    """
    out = fhe.int5
    params = {"a": out}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, out)

    @fhe.circuit({"a": "encrypted"})
    def circuit(a: out):
        return relu(a=a)

    def ref_relu(v):
        return v if v >= 0 else 0

    r = list(range(0, 2**4)) + list(range(-(2**4), 0))
    for a in r:
        assert circuit.simulate(a) == ref_relu(a)


def test_signed_add_correctness():
    """
    Check add.
    """
    out = fhe.int5
    params = {"a": out, "b": out}
    expression = "a + b"

    @fhe.circuit({"a": "encrypted", "b": "encrypted"})
    def circuit(a: out, b: out):
        return synth.verilog_expression(params, expression, out)(a=a, b=b)

    print(circuit.mlir)

    r = list(range(0, 2**4)) + list(range(-(2**4), 0))
    for a in r:
        for b in r:
            if a + b in r:
                assert circuit.simulate(a, b) == a + b


def test_unsigned_add_correctness():
    """
    Check add.
    """
    out = fhe.uint5
    params = {"a": out, "b": out}
    expression = "a + b"

    @fhe.circuit({"a": "encrypted", "b": "encrypted"})
    def circuit(a: out, b: out):
        return synth.verilog_expression(params, expression, out)(a=a, b=b)

    print(circuit.mlir)

    r = list(range(0, 2**5))
    for a in r:
        for b in r:
            if a + b in r:
                assert circuit.simulate(a, b) == a + b


def test_signed_mul_correctness():
    """
    Check add.
    """
    out = fhe.int5
    params = {"a": out, "b": out}
    expression = "a * b"

    @fhe.circuit({"a": "encrypted", "b": "encrypted"})
    def circuit(a: out, b: out):
        return synth.verilog_expression(params, expression, out)(a=a, b=b)

    r = list(range(0, 2**4)) + list(range(-(2**4), 0))
    for a in r:
        for b in r:
            if a * b in r:
                assert circuit.simulate(a, b) == a * b


def test_unsigned_mul_correctness():
    """
    Check add.
    """
    out = fhe.uint5
    params = {"a": out, "b": out}
    expression = "a * b"

    @fhe.circuit({"a": "encrypted", "b": "encrypted"})
    def circuit(a: out, b: out):
        return synth.verilog_expression(params, expression, out)(a=a, b=b)

    r = list(range(0, 2**5))
    for a in r:
        for b in r:
            if a * b in r:
                assert circuit.simulate(a, b) == a * b


def test_relu_limit():
    """
    Check the limit for relu precision.
    28bits is only attainable only with weight in tlu fuzing.
    """
    out = fhe.int28
    params = {"a": out}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, out)

    @fhe.circuit({"a": "encrypted", "b": "encrypted"})
    def _(a: out, b: out):
        v = relu(a=a) - relu(a=b)
        return relu(a=v)

    out = fhe.int29
    params = {"a": out}
    expression = "(a >= 0) ? a : 0"

    relu = synth.verilog_expression(params, expression, out)

    with pytest.raises(RuntimeError) as err:

        @fhe.circuit({"a": "encrypted", "b": "encrypted"})
        def _(a: out, b: out):
            v = relu(a=a) - relu(a=b)
            return relu(a=v)

    assert str(err.value) == "NoParametersFound"
