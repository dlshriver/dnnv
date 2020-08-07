import os
import unittest

import dnnv.verifiers.reluplex as reluplex

from tests.verifiers_tests.utils import VerifierTests, has_verifier

from tests.utils import network_artifact_dir, property_artifact_dir


@unittest.skipIf(not has_verifier("reluplex"), "Reluplex is not installed")
class ReluplexVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = reluplex
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
