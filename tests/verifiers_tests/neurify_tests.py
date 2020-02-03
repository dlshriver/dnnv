import os
import unittest

import dnnv.verifiers.neurify as neurify

from tests.verifiers_tests.utils import VerifierTests


class NeurifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = neurify
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
