import os
import unittest

from dnnv import nn
from dnnv import properties
from dnnv.properties import Symbol
from dnnv.verifiers.common import SAT, UNSAT, UNKNOWN
from dnnv.verifiers import bab, eran, neurify, planet, reluplex

from tests.utils import network_artifact_dir, property_artifact_dir

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))


class MNISTTests(unittest.TestCase):
    def setUp(self):
        self.reset_property_context()
        for varname in ["SEED", "EPSILON", "INPUT_LAYER", "OUTPUT_LAYER"]:
            if varname in os.environ:
                del os.environ[varname]

    def reset_property_context(self):
        # TODO : refactor property implementation so this can be removed
        # required to ensure concretized symbols don't carry over
        Symbol._instances = {}

    def check_results(self, result, results):
        if len(results) == 0:
            return
        if result == UNKNOWN:
            return
        previous_result = results[-1]
        self.assertEqual(result, previous_result)
        results.append(result)

    def test_mnist_relu_3_50(self):
        verifiers = {
            # "bab": bab, # too slow
            "eran": eran,
            "neurify": neurify,
            # "planet": planet, # too slow
            # "reluplex": reluplex, # too slow
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in verifiers.items():
                self.reset_property_context()
                dnn = nn.parse(network_artifact_dir / "mnist_relu_3_50.onnx")
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                result = verifier.verify(dnn, phi)
                self.check_results(result, results)

    def test_convSmallRELU__Point(self):
        os.environ["INPUT_LAYER"] = "4"
        verifiers = {
            # "bab": bab, # too slow
            "eran": eran,
            "neurify": neurify,
            # "planet": planet, # too slow
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in verifiers.items():
                self.reset_property_context()
                dnn = nn.parse(network_artifact_dir / "convSmallRELU__Point.onnx")
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                result = verifier.verify(dnn, phi)
                self.check_results(result, results)


if __name__ == "__main__":
    unittest.main()
