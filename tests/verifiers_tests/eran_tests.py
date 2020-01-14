import os
import unittest

import dnnv.verifiers.eran as eran

from tests.verifiers_tests.utils import VerifierTests

from tests.utils import network_artifact_dir, property_artifact_dir


class ERANVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = eran
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
