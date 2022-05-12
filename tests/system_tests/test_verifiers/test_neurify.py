import unittest

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.neurify import Neurify


@unittest.skipIf(not Neurify.is_installed(), "Neurify is not installed")
class NeurifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Neurify
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
