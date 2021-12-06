import unittest

from dnnv.verifiers.bab import BaB

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not BaB.is_installed(), "BaB is not installed")
class BabVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = BaB
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
