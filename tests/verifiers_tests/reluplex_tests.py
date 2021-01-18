import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.reluplex import Reluplex


@unittest.skipIf(not Reluplex.is_installed(), "Reluplex is not installed")
class ReluplexVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Reluplex
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
