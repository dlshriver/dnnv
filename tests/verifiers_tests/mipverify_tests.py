import os
import unittest

from dnnv.verifiers.mipverify import MIPVerify

from tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not MIPVerify.is_installed(), "MIPVerify is not installed")
class MIPVerifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = MIPVerify
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
