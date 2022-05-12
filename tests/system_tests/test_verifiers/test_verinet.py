import unittest

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.verinet import VeriNet


@unittest.skipIf(not VeriNet.is_installed(), "VeriNet is not installed")
class VeriNetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = VeriNet
        self.is_complete = False

    @unittest.skip("VeriNet is too slow on this problem.")
    def test_a_gt_b_output_greater_than_x_unsat(self):
        super().test_a_gt_b_output_greater_than_x_unsat()


if __name__ == "__main__":
    unittest.main()
