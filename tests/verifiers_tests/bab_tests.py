import os
import unittest

from tests.verifiers_tests.utils import VerifierTests, has_verifier

from tests.utils import network_artifact_dir, property_artifact_dir

bab = None
if has_verifier("bab"):
    import dnnv.verifiers.bab as bab


@unittest.skipIf(not has_verifier("bab"), "BaB is not installed")
class BabVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = bab
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
