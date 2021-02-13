import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.verinet import VeriNet


@unittest.skipIf(not VeriNet.is_installed(), "VeriNet is not installed")
class VeriNetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = VeriNet
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
