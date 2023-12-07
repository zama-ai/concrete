"""
Tests of execution of add operation.
"""

import numpy as np

from concrete import fhe


def test_composed_inc(helpers):
    """
    Test add where one of the operators is a constant.
    """

    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        # Only valid with multi
        return

    lut = fhe.LookupTable(list(range(32)))

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return lut[x + 1]

    inputset = range(30)
    circuit = function.compile(inputset, helpers.configuration())

    samples = [
        [
            np.random.randint(
                0,
                31 - 6,
            )
        ]
        for _ in range(5)
    ]

    for sample in samples:
        helpers.check_composition(circuit, function, sample, composed=6)
