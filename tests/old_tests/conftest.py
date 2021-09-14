import pytest
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def pytest_collect_file(parent, path):
    if path.basename.endswith("_tests.py"):
        return pytest.Module.from_parent(parent, fspath=path)
