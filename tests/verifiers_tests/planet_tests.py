import os
import unittest

import dnnv.verifiers.planet as planet

from tests.verifiers_tests.utils import VerifierTests, has_verifier

from tests.utils import network_artifact_dir, property_artifact_dir


@unittest.skipIf(not has_verifier("planet"), "Planet is not installed")
class PlanetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = planet
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
