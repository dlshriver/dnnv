import unittest
from functools import partial

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.mipverify import MIPVerify


@unittest.skipIf(not MIPVerify.is_installed(), "MIPVerify is not installed")
class MIPVerifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = MIPVerify
        self.verifier.verify = partial(self.verifier.verify, optimizer="HiGHS")
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
