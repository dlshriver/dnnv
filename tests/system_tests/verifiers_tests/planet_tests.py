import unittest

from dnnv.verifiers.planet import Planet

from system_tests.verifiers_tests.utils import VerifierTests


@unittest.skipIf(not Planet.is_installed(), "Planet is not installed")
class PlanetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Planet
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
