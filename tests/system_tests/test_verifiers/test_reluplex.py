import unittest

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.reluplex import Reluplex


@unittest.skipIf(not Reluplex.is_installed(), "Reluplex is not installed")
class ReluplexVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Reluplex
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
