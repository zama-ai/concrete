import os
import tempfile
import pytest
from concrete.compiler import KeysetCache

KEY_SET_CACHE_PATH = os.path.join(tempfile.gettempdir(), "KeySetCache")


def pytest_configure(config):
    config.addinivalue_line("markers", "parallel: mark parallel tests")


@pytest.fixture(scope="session")
def keyset_cache():
    return KeysetCache(KEY_SET_CACHE_PATH)


@pytest.fixture(scope="session")
def no_parallel(request):
    """Check if parallel tests have been selected."""
    session = request.node
    found_parallel = False
    for item in session.items:
        for marker in item.iter_markers():
            if marker.name == "parallel":
                found_parallel = True
                break
        if found_parallel:
            break
    return not found_parallel
