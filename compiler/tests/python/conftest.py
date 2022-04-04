def pytest_configure(config):
    config.addinivalue_line("markers", "parallel: mark parallel tests")
