import os
import unittest

from dnnv import nn
from dnnv import properties
from dnnv.properties import get_context

from dnnv.verifiers import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.bab import BaB
from dnnv.verifiers.babsb import BaBSB
from dnnv.verifiers.eran import ERAN
from dnnv.verifiers.marabou import Marabou
from dnnv.verifiers.mipverify import MIPVerify
from dnnv.verifiers.neurify import Neurify
from dnnv.verifiers.nnenum import Nnenum
from dnnv.verifiers.planet import Planet
from dnnv.verifiers.reluplex import Reluplex
from dnnv.verifiers.verinet import VeriNet

from system_tests.utils import network_artifact_dir, property_artifact_dir

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))

VERIFIERS = {
    "bab": BaB,
    "babsb": BaBSB,
    "eran_deepzono": ERAN,
    "eran_deeppoly": ERAN,
    # "eran_refinezono": ERAN, # TODO : is_installed (needs to check for gurobi license)
    # "eran_refinepoly": ERAN, # TODO : is_installed (needs to check for gurobi license)
    "marabou": Marabou,
    "mipverify": MIPVerify,
    "neurify": Neurify,
    "nnenum": Nnenum,
    "planet": Planet,
    "reluplex": Reluplex,
    "verinet": VeriNet,
}

VERIFIER_KWARGS = {
    "eran_deepzono": {"domain": "deepzono"},
    "eran_deeppoly": {"domain": "deeppoly"},
    "eran_refinezono": {"domain": "refinezono"},
    "eran_refinepoly": {"domain": "refinepoly"},
}


class MNISTTests(unittest.TestCase):
    def setUp(self):
        self.reset_property_context()
        for varname in ["SEED", "EPSILON", "INPUT_LAYER", "OUTPUT_LAYER"]:
            if varname in os.environ:
                os.environ.pop(varname)

    def tearDown(self):
        self.reset_property_context()
        for varname in ["SEED", "EPSILON", "INPUT_LAYER", "OUTPUT_LAYER"]:
            if varname in os.environ:
                os.environ.pop(varname)

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
        excluded_verifiers = {
            "bab",  # too slow
            "babsb",  # too slow
            "planet",  # too slow
            "reluplex",  # too slow
            "verinet",  # returns error # TODO: fix?
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in VERIFIERS.items():
                if name in excluded_verifiers:
                    continue
                if not verifier.is_installed():
                    continue
                self.reset_property_context()
                dnn = nn.parse(network_artifact_dir / "mnist_relu_3_50.onnx").simplify()
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                phi.concretize(N=dnn)
                result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                self.check_results(result, results)

    @unittest.skip("Too slow")
    def test_convSmallRELU__Point(self):
        excluded_verifiers = {
            "bab",  # too slow
            "babsb",  # too slow
            "mipverify",  # too slow
            "planet",  # too slow
            "reluplex",  # doesn't support Conv
        }
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i)
            results = []
            for name, verifier in VERIFIERS.items():
                if name in excluded_verifiers:
                    continue
                if not verifier.is_installed():
                    continue
                self.reset_property_context()
                dnn = nn.parse(
                    network_artifact_dir / "convSmallRELU__Point.onnx"
                ).simplify()
                phi = properties.parse(
                    property_artifact_dir / "mnist_localrobustness_rand.py"
                )
                phi.concretize(N=dnn)
                result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                self.check_results(result, results)

    def test_mnist_300_200_100(self):
        excluded_verifiers = {
            "bab",  # too slow
            "babsb",  # too slow
            "marabou",  # too slow
            "mipverify",  # property not supported (multiple epsilon values) # TODO : fix?
            "planet",  # too slow
            "reluplex",  # too slow
        }
        results = []
        for name, verifier in VERIFIERS.items():
            if name in excluded_verifiers:
                continue
            if not verifier.is_installed():
                continue
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "mnist_300_200_100.onnx").simplify()
            phi = properties.parse(property_artifact_dir / "mnist_localrobustness_x.py")
            phi.concretize(N=dnn)
            result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
            self.check_results(result, results)


if __name__ == "__main__":
    unittest.main()
