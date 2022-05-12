import unittest

from system_tests.test_verifiers.utils import VerifierTests

from dnnv.verifiers.planet import Planet


@unittest.skipIf(not Planet.is_installed(), "Planet is not installed")
class PlanetVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Planet
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
