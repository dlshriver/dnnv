import unittest

from dnnv.verifiers.nnenum import Nnenum

from system_tests.test_verifiers.utils import VerifierTests


@unittest.skipIf(not Nnenum.is_installed(), "nnenum is not installed")
class NnenumVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Nnenum
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
