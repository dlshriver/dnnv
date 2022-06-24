import os
import unittest

from system_tests.utils import network_artifact_dir, property_artifact_dir

from dnnv import nn, properties
from dnnv.properties import get_context
from dnnv.verifiers import SAT, UNKNOWN, UNSAT
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

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))

VERIFIERS = {
    "bab": BaB,
    "babsb": BaBSB,
    "eran": ERAN,
    "marabou": Marabou,
    "mipverify": MIPVerify,
    "neurify": Neurify,
    "nnenum": Nnenum,
    "planet": Planet,
    "reluplex": Reluplex,
    "verinet": VeriNet,
}

VERIFIERS = {
    "bab": BaB,
    "babsb": BaBSB,
    "eran_deepzono": ERAN,
    "eran_deeppoly": ERAN,
    # "eran_refinezono": ERAN, # TODO : is_installed (needs to check for gurobi license)
    # "eran_refinepoly": ERAN, # TODO : is_installed (needs to check for gurobi license)
    "marabou": Marabou,
    # "mipverify": MIPVerify, # TODO : is_installed (needs to check for gurobi license)
    "mipverify_HiGHS": MIPVerify,
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
    "mipverify_HiGHS": {"optimizer": "HiGHS"},
}


@unittest.skipIf(
    sum(v.is_installed() for v in VERIFIERS.values()) < 2,
    "Not enough verifiers installed",
)
class RandomTests(unittest.TestCase):
    def setUp(self):
        self.reset_property_context()
        for varname in [
            "SEED",
            "SHIFT",
            "SCALE",
            "EPSILON",
            "INPUT_LAYER",
            "OUTPUT_LAYER",
        ]:
            if varname in os.environ:
                del os.environ[varname]

    def reset_property_context(self):
        get_context().reset()

    def check_results(self, result, results):
        if result == UNKNOWN:
            return
        results.append(result)
        if len(results) == 1:
            return
        previous_result = results[-2]
        self.assertEqual(result, previous_result)

    def test_random_fc_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("marabou", 0.5),  # numerical inconsistency
            ("marabou", 1.0),  # numerical inconsistency
            ("verinet", 0.5),  # too slow
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_0.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_fc_1(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("verinet", 0.5),  # too slow
            ("verinet", 1.0),  # too slow
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_1.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_fc_2(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("verinet", 0.5),  # too slow
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_2.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_conv_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("reluplex", 0.01),  # conv unsupported
            ("reluplex", 0.1),  # conv unsupported
            ("reluplex", 0.5),  # conv unsupported
            ("reluplex", 1.0),  # conv unsupported
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(
                        network_artifact_dir / "random_conv_0.onnx"
                    ).simplify()
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_conv_1(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("reluplex", 0.01),  # conv unsupported
            ("reluplex", 0.1),  # conv unsupported
            ("reluplex", 0.5),  # conv unsupported
            ("reluplex", 1.0),  # conv unsupported
            ("verinet", 0.01),  # too slow
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(
                        network_artifact_dir / "random_conv_1.onnx"
                    ).simplify()
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_conv_2(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_configs = {
            ("reluplex", 0.01),  # conv unsupported
            ("reluplex", 0.1),  # conv unsupported
            ("reluplex", 0.5),  # conv unsupported
            ("reluplex", 1.0),  # conv unsupported
            ("marabou", 1.0),  # numerical instability
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(
                        network_artifact_dir / "random_conv_2.onnx"
                    ).simplify()
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_random_residual_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {
            "eran": ERAN,
            "planet": Planet,
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_residual_0.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)

    def test_hyperlocal_random_conv_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        excluded_verifiers = {
            "reluplex",
            "verinet",
        }
        excluded_configs = {
            ("marabou", 0.5),  # numerical inconsistencies
            ("marabou", 1.0),  # numerical inconsistencies
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in VERIFIERS.items():
                    if (name, epsilon) in excluded_configs:
                        continue
                    if name in excluded_verifiers:
                        continue
                    if not verifier.is_installed():
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(
                        network_artifact_dir / "random_conv_0.onnx"
                    ).simplify()
                    phi = properties.parse(
                        property_artifact_dir / "hyperlocalrobustness.py"
                    )
                    phi.concretize(N=dnn)
                    result, _ = verifier.verify(phi, **VERIFIER_KWARGS.get(name, {}))
                    self.check_results(result, results)


if __name__ == "__main__":
    unittest.main()
