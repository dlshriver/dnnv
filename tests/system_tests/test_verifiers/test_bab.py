import unittest

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.bab import BaB


@unittest.skipIf(not BaB.is_installed(), "BaB is not installed")
class BabVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = BaB
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
