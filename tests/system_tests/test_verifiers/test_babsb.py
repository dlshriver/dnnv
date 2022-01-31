import unittest

from dnnv.verifiers.babsb import BaBSB

from system_tests.test_verifiers.utils import VerifierTests


@unittest.skipIf(not BaBSB.is_installed(), "BaBSB is not installed")
class BaBSBVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = BaBSB
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
