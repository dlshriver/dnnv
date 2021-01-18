import os
import unittest

from dnnv.verifiers.mipverify import MIPVerify

from tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not MIPVerify.is_installed(), "MIPVerify is not installed")
class MIPVerifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = MIPVerify
        self.is_complete = False

    @unittest.skip("MIPVerify throws unexpected error.")
    def test_a_gt_b_localrobustness_sat(self):
        super().test_a_gt_b_localrobustness_sat()


if __name__ == "__main__":
    unittest.main()
