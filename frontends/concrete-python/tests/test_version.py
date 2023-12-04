"""
Tests of `version` module.
"""

from concrete import fhe


def test_version_exists():
    """
    Test `concrete.fhe` has `__version__`.
    """

    assert hasattr(fhe, "__version__")
