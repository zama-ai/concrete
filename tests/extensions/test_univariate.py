"""
Tests of 'univariate' extension.
"""

import pytest

import concrete.numpy as cnp


def test_bad_univariate(helpers):
    """
    Test 'univariate' extension with bad parameters.
    """

    with pytest.raises(ValueError) as excinfo:

        @cnp.circuit({"x": "encrypted"}, helpers.configuration())
        def function(x: cnp.uint3):
            return cnp.univariate(lambda x: x**2)(x)

    assert str(excinfo.value) == (
        "Univariate extension requires `outputs` argument for direct circuit definition "
        "(e.g., cnp.univariate(function, outputs=cnp.uint4)(x))"
    )
