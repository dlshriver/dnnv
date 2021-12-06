import unittest

from dnnv.verifiers.marabou import Marabou

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not Marabou.is_installed(), "Marabou is not installed")
class MarabouVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Marabou
        self.is_complete = True

    @unittest.skip("Marabou is numerically unstable.")
    def test_const_one_ge1_unsat(self):
        super().test_const_one_ge1_unsat()

    @unittest.skip("Marabou is numerically unstable.")
    def test_const_zero_localrobustness(self):
        super().test_const_zero_localrobustness()

    @unittest.skip("Marabou is numerically unstable.")
    def test_sum_gt_one_localrobustness_shift_left_unsat(self):
        super().test_sum_gt_one_localrobustness_shift_left_unsat()


if __name__ == "__main__":
    unittest.main()
