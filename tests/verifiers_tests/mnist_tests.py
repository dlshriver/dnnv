import os
import unittest

from dnnv import nn
from dnnv import properties
from dnnv.properties import Symbol
from dnnv.properties.context import get_context

from dnnv.verifiers import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.bab import BaB
from dnnv.verifiers.eran import ERAN
from dnnv.verifiers.mipverify import MIPVerify
from dnnv.verifiers.neurify import Neurify
from dnnv.verifiers.planet import Planet
from dnnv.verifiers.reluplex import Reluplex

from tests.utils import network_artifact_dir, property_artifact_dir

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))


class MNISTTests(unittest.TestCase):
    def setUp(self):
        self.reset_property_context()
        for varname in ["SEED", "EPSILON", "INPUT_LAYER", "OUTPUT_LAYER"]:
            if varname in os.environ:
                del os.environ[varname]

    def reset_property_context(self):
        get_context().reset()

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
            # "bab": BaB, # too slow
            "eran": ERAN,
            "neurify": Neurify,
            # "planet": Planet, # too slow
            # "reluplex": Reluplex, # too slow
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in verifiers.items():
                if not verifier.is_installed():
                    continue
                self.reset_property_context()
                dnn = nn.parse(network_artifact_dir / "mnist_relu_3_50.onnx")
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                phi.concretize(N=dnn)
                result = verifier.verify(phi)
                self.check_results(result, results)

    def test_convSmallRELU__Point(self):
        os.environ["INPUT_LAYER"] = "0"
        verifiers = {
            # "bab": BaB, # too slow
            "eran": ERAN,
            "neurify": Neurify,
            # "planet": Planet, # too slow
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in verifiers.items():
                if not verifier.is_installed():
                    continue
                self.reset_property_context()
                dnn = nn.parse(network_artifact_dir / "convSmallRELU__Point.onnx")
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                phi.concretize(N=dnn)
                result = verifier.verify(phi)
                self.check_results(result, results)


if __name__ == "__main__":
    unittest.main()
