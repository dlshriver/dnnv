import os
import unittest

import dnnv.verifiers.bab as bab

from tests.verifiers_tests.utils import VerifierTests

from tests.utils import network_artifact_dir, property_artifact_dir


class BabVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = bab
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
