import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.marabou import Marabou


@unittest.skipIf(not Marabou.is_installed(), "Marabou is not installed")
class MarabouVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Marabou
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
