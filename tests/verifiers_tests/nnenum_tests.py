import os
import unittest

from tests.verifiers_tests.utils import VerifierTests
from tests.utils import network_artifact_dir, property_artifact_dir

from dnnv.verifiers.nnenum import Nnenum


@unittest.skipIf(not Nnenum.is_installed(), "nnenum is not installed")
class NnenumVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Nnenum
        self.is_complete = True


if __name__ == "__main__":
    unittest.main()
