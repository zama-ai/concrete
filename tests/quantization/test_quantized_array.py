"""Tests for the quantized array/tensors."""
import numpy
import pytest

from concrete.quantization import QuantizedArray

N_BITS_ATOL_TUPLE_LIST = [
    (32, 10 ** -2),
    (28, 10 ** -2),
    (20, 10 ** -2),
    (16, 10 ** -1),
    (8, 10 ** -0),
    (4, 10 ** -0),
]


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
@pytest.mark.parametrize("values", [pytest.param(numpy.random.randn(2000))])
def test_quant_dequant_update(values, n_bits, atol, is_signed):
    """Test the quant and dequant function."""

    quant_array = QuantizedArray(n_bits, values, is_signed)
    qvalues = quant_array.quant()

    # Quantized values must be contained between 0 and 2**n_bits
    assert numpy.max(qvalues) <= 2 ** (n_bits) - 1 - quant_array.offset
    assert numpy.min(qvalues) >= -quant_array.offset

    # Dequantized values must be close to original values
    dequant_values = quant_array.dequant()

    # Check that all values are close
    assert numpy.isclose(dequant_values, values, atol=atol).all()

    # Test update functions
    new_values = numpy.array([0.3, 0.5, -1.2, -3.4])
    new_qvalues_ = quant_array.update_values(new_values)

    # Make sure the shape changed for the qvalues
    assert new_qvalues_.shape != qvalues.shape

    new_qvalues = numpy.array([1, 4, 7, 29])
    new_values_updated = quant_array.update_qvalues(new_qvalues)

    # Make sure that we can see at least one change.
    assert not numpy.array_equal(new_qvalues, new_qvalues_)
    assert not numpy.array_equal(new_values, new_values_updated)

    # Check that the __call__ returns also the qvalues.
    assert numpy.array_equal(quant_array(), new_qvalues)
