"""
Tests of 'tag' extension.
"""

import numpy as np

from concrete import fhe


def test_tag(helpers):
    """
    Test tag extension.
    """

    def g(z):
        with fhe.tag("def"):
            a = 120 - z
            b = a // 4
        return b

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        with fhe.tag("abc"):
            x = x * 2
            with fhe.tag("foo"):
                y = x + 42
            z = np.sqrt(y).astype(np.int64)

        return g(z + 3) * 2

    inputset = range(10)
    circuit = f.trace(inputset, configuration=helpers.configuration())

    helpers.check_str(
        """

 %0 = x                            # EncryptedScalar<uint4>
 %1 = 2                            # ClearScalar<uint2>            @ abc
 %2 = multiply(%0, %1)             # EncryptedScalar<uint5>        @ abc
 %3 = 42                           # ClearScalar<uint6>            @ abc.foo
 %4 = add(%2, %3)                  # EncryptedScalar<uint6>        @ abc.foo
 %5 = subgraph(%4)                 # EncryptedScalar<uint3>        @ abc
 %6 = 3                            # ClearScalar<uint2>
 %7 = add(%5, %6)                  # EncryptedScalar<uint4>
 %8 = 120                          # ClearScalar<uint7>            @ def
 %9 = subtract(%8, %7)             # EncryptedScalar<uint7>        @ def
%10 = 4                            # ClearScalar<uint3>            @ def
%11 = floor_divide(%9, %10)        # EncryptedScalar<uint5>        @ def
%12 = 2                            # ClearScalar<uint2>
%13 = multiply(%11, %12)           # EncryptedScalar<uint6>
return %13

Subgraphs:

    %5 = subgraph(%4):

        %0 = input                         # EncryptedScalar<uint2>          @ abc.foo
        %1 = sqrt(%0)                      # EncryptedScalar<float64>        @ abc
        %2 = astype(%1, dtype=int_)        # EncryptedScalar<uint1>          @ abc
        return %2

        """.strip(),
        circuit.format(show_bounds=False),
    )
