import unittest

from dnnv.verifiers.verinet import VeriNet

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not VeriNet.is_installed(), "VeriNet is not installed")
class VeriNetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = VeriNet
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
