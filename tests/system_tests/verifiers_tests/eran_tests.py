import unittest

from dnnv.verifiers.eran import ERAN

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not ERAN.is_installed(), "ERAN is not installed")
class ERANVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = ERAN
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
