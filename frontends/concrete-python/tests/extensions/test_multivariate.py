"""
Tests of 'multivariate' extension.
"""

import pytest

from concrete import fhe


def test_bad_multivariate(helpers):
    """
    Test 'multivariate' extension with bad parameters.
    """

    # direct definition without outputs

    with pytest.raises(ValueError) as excinfo:

        @fhe.circuit({"x": "encrypted"}, helpers.configuration())
        def no_outputs_in_direct_definition(x: fhe.uint3):
            return fhe.multivariate(lambda x: x**2)(x)

    assert str(excinfo.value) == (
        "Multivariate extension requires `outputs` argument for direct circuit definition "
        "(e.g., fhe.multivariate(function, outputs=fhe.uint4)(x, y, z))"
    )

    # clear inputs

    with pytest.raises(ValueError) as excinfo:

        @fhe.compiler({"x": "encrypted", "y": "clear"})
        def clear_inputs(x, y):
            return fhe.multivariate(lambda x, y: x * y)(x, y)

        inputset = [(1, 50)]
        clear_inputs.compile(inputset, helpers.configuration())

    assert str(excinfo.value) == (
        "Multivariate extension requires all of its inputs to be encrypted"
    )

    # rounded inputs

    with pytest.raises(ValueError) as excinfo:

        @fhe.compiler({"x": "encrypted", "y": "encrypted"})
        def rounded_inputs(x, y):
            y = fhe.round_bit_pattern(y, lsbs_to_remove=2)
            return fhe.multivariate(lambda x, y: x * y)(x, y)

        inputset = [(1, 50)]
        rounded_inputs.compile(inputset, helpers.configuration())

    assert str(excinfo.value) == "Multivariate extension cannot be used with rounded inputs"

    # bad output shape

    with pytest.raises(ValueError) as excinfo:

        @fhe.compiler({"x": "encrypted", "y": "encrypted"})
        def invalid_output_shape(x, y):
            return fhe.multivariate(lambda x, y: (x * y)[0])(x, y)

        inputset = [(1, [1, 2, 3])]
        invalid_output_shape.compile(inputset, helpers.configuration())

    assert str(excinfo.value) == "Function <lambda> is not compatible with fhe.multivariate"
