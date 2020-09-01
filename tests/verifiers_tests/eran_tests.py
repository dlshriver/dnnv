import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.eran import ERAN


@unittest.skipIf(not ERAN.is_installed(), "ERAN is not installed")
class ERANVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = ERAN
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
