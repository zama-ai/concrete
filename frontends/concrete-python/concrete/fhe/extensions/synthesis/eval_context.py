# pylint: disable=missing-module-docstring,missing-function-docstring

from dataclasses import dataclass

import numpy as np

from concrete.fhe.extensions.synthesis.verilog_source import Ty


class EvalContext:
    """
    This is a reduced context with similar method as `concrete.fhe.mlir.Context`.

    It provides a clear evaluation backend for tlu_circuit_to_mlir.
    Until the all internal_api are used directly by concrete-python,
    this helps to keep all tests for previous backend and api.
    For now only synthesis of TLU is supported by concrete-python.
    """

    # For all `EvalContext` method look at `Context` documentation.

    @dataclass
    class Ty:
        """Equivalent for `ConversionType`."""

        bit_width: int
        is_tensor: bool = False
        shape: tuple = ()  # not used

    @dataclass
    class Val:
        """Equivalent for `Conversion`. Contains the evaluation result."""

        value: 'int | np.ndarray'
        type: Ty

        def __init__(self, value, type_):
            try:
                value = int(value)
            except TypeError:
                pass
            self.value = value
            self.type = type_

    def fork_type(self, type_, bit_width=None, shape=None):
        return self.Ty(bit_width=bit_width or type_.bit_width, shape=shape or type_.shape)

    def i(self, size):
        return self.Ty(size)

    def constant(self, type_: Ty, value: int):
        return self.Val(value, type_)

    def mul(self, type_: Ty, a: Val, b: Val):
        assert isinstance(b.value, int)
        assert a.type == type_
        return self.Val(a.value * b.value, type_)

    def add(self, type_: Ty, a: Val, b: Val):
        assert a.type == b.type == type_
        return self.Val(a.value + b.value, type_)

    def sub(self, type_: Ty, a: Val, b: Val):
        assert isinstance(b.value, int)
        assert a.type == type_
        return self.Val(a.value - b.value, type_)

    def tlu(self, type_: Ty, arg: Val, tlu_content, **_kwargs):
        if isinstance(arg, int):
            v = self.Val(tlu_content[arg.value], type_)
        else:
            v = np.vectorize(lambda v: int(tlu_content[v]))(arg.value)
        return self.Val(v, type_)

    def extract_bits(self, type_: Ty, arg: Val, bit_index, **_kwargs):
        return self.Val((arg.value >> bit_index) & 1, type_)

    def to_unsigned(self, arg: Val):
        def aux(value):
            if value < 0:
                return 2**arg.type.bit_width + value
            return value

        if isinstance(arg.value, int):
            v = aux(arg.value)
        else:
            v = np.vectorize(aux)(arg.value)
        return self.Val(v, arg.type)

    def to_signed(self, arg: Val):
        def aux(value):
            assert value >= 0
            negative = value >= 2 ** (arg.type.bit_width - 1)
            if negative:
                return -(2**arg.type.bit_width - arg.value)
            return value

        if isinstance(arg.value, int):
            v = aux(arg.value)
        else:
            v = np.vectorize(aux)(arg.value)
        return self.Val(v, arg.type)

    def index(self, type_: Ty, tensor: Val, index):
        assert isinstance(tensor.value, list), type(tensor.value)
        assert len(index) == 1
        (index,) = index
        return self.Val(tensor.value[index], self.Ty(type_.bit_width, is_tensor=False))

    def reinterpret(self, arg, bit_width=None):
        arg_bit_width = arg.type.bit_width
        if bit_width is None:
            bit_width = arg_bit_width
        if bit_width == arg_bit_width:
            return arg
        shift = 2 ** (bit_width - arg_bit_width)
        if isinstance(arg, int):
            v = arg.value * shift
        else:
            v = np.vectorize(lambda v: v * shift)(arg.value)
        return self.Val(v, self.Ty(bit_width=bit_width))

    def safe_reduce_precision(self, arg, bit_width):
        if arg.type.bit_width == bit_width:
            return arg
        assert arg.type.bit_width > bit_width
        shift = arg.type.bit_width - bit_width
        shifted = self.mul(arg.type, arg, self.constant(self.i(bit_width + 1), 2**shift))
        return self.reinterpret(shifted, bit_width)
