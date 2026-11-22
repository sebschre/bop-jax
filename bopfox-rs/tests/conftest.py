def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long statistical tests (deselect with -m 'not slow')")
