"""
Tests of 'univariate' extension.
"""

import pytest

from concrete import fhe


def test_bad_univariate(helpers):
    """
    Test 'univariate' extension with bad parameters.
    """

    with pytest.raises(ValueError) as excinfo:

        @fhe.circuit({"x": "encrypted"}, helpers.configuration())
        def function(x: fhe.uint3):
            return fhe.univariate(lambda x: x**2)(x)

    assert str(excinfo.value) == (
        "Univariate extension requires `outputs` argument for direct circuit definition "
        "(e.g., fhe.univariate(function, outputs=fhe.uint4)(x))"
    )
