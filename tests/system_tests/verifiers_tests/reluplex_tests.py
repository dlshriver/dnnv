import unittest

from dnnv.verifiers.reluplex import Reluplex

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not Reluplex.is_installed(), "Reluplex is not installed")
class ReluplexVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Reluplex
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
