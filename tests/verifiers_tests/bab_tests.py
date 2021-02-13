import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.bab import BaB


@unittest.skipIf(not BaB.is_installed(), "BaB is not installed")
class BabVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = BaB
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
