import pytest


def pytest_collect_file(parent, path):
    if path.basename.endswith("_tests.py"):
        return pytest.Module.from_parent(parent, fspath=path)
