import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.planet import Planet


@unittest.skipIf(not Planet.is_installed(), "Planet is not installed")
class PlanetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Planet
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
