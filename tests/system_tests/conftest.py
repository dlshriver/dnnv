import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from system_tests.artifacts.build_artifacts import build

build()
