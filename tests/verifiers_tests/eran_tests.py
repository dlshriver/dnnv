import os
import unittest

from tests.verifiers_tests.utils import VerifierTests, has_verifier

from tests.utils import network_artifact_dir, property_artifact_dir

eran = None
if has_verifier("eran"):
    import dnnv.verifiers.eran as eran


@unittest.skipIf(not has_verifier("eran"), "ERAN is not installed")
class ERANVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = eran
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
