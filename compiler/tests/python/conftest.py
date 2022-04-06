import os
import tempfile
import pytest
from concrete.compiler import KeySetCache

KEY_SET_CACHE_PATH = os.path.join(tempfile.gettempdir(), "KeySetCache")


def pytest_configure(config):
    config.addinivalue_line("markers", "parallel: mark parallel tests")


@pytest.fixture(scope="session")
def keyset_cache():
    return KeySetCache.new(KEY_SET_CACHE_PATH)
