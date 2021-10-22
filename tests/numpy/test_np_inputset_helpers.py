"""Test file for numpy inputset helpers"""

import numpy as np
import pytest

from concrete.common.compilation import CompilationConfiguration
from concrete.common.data_types import Float, UnsignedInteger
from concrete.common.data_types.base import BaseDataType
from concrete.common.values import BaseValue, EncryptedScalar, EncryptedTensor
from concrete.numpy.np_inputset_helpers import _generate_random_inputset


def test_generate_random_inputset():
    """Test function for generate_random_inputset"""

    inputset = _generate_random_inputset(
        {
            "x1": EncryptedScalar(UnsignedInteger(4)),
            "x2": EncryptedTensor(UnsignedInteger(4), shape=(2, 3)),
            "x3": EncryptedScalar(Float(64)),
            "x4": EncryptedTensor(Float(64), shape=(3, 2)),
        },
        CompilationConfiguration(random_inputset_samples=15),
    )

    assert isinstance(inputset, list)
    assert len(inputset) == 15

    for sample in inputset:
        assert isinstance(sample, tuple)
        assert len(sample) == 4

        assert isinstance(sample[0], int)
        assert 0 <= sample[0] < 2 ** 4

        assert isinstance(sample[1], np.ndarray)
        assert sample[1].dtype == np.uint64
        assert sample[1].shape == (2, 3)
        assert (sample[1] >= 0).all()
        assert (sample[1] < 2 ** 4).all()

        assert isinstance(sample[2], float)
        assert 0 <= sample[2] < 1

        assert isinstance(sample[3], np.ndarray)
        assert sample[3].dtype == np.float64
        assert sample[3].shape == (3, 2)
        assert (sample[3] >= 0).all()
        assert (sample[3] < 1).all()


def test_fail_generate_random_inputset():
    """Test function for failed generate_random_inputset"""

    class MockDtype(BaseDataType):
        """Unsupported dtype to check error messages"""

        def __eq__(self, o: object) -> bool:
            return False

        def __str__(self):
            return "MockDtype"

    class MockValue(BaseValue):
        """Unsupported value to check error messages"""

        def __init__(self):
            super().__init__(MockDtype(), is_encrypted=True)

        def __eq__(self, other: object) -> bool:
            return False

        def __str__(self):
            return "MockValue"

    with pytest.raises(ValueError):
        try:
            _generate_random_inputset(
                {"x": MockValue()},
                CompilationConfiguration(random_inputset_samples=15),
            )
        except Exception as error:
            expected = "Random inputset cannot be generated for MockValue parameters"
            assert str(error) == expected
            raise

    with pytest.raises(ValueError):
        try:
            _generate_random_inputset(
                {"x": EncryptedScalar(MockDtype())},
                CompilationConfiguration(random_inputset_samples=15),
            )
        except Exception as error:
            expected = "Random inputset cannot be generated for parameters of type MockDtype"
            assert str(error) == expected
            raise
