import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.babsb import BaBSB


@unittest.skipIf(not BaBSB.is_installed(), "BaBSB is not installed")
class BaBSBVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = BaBSB
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
