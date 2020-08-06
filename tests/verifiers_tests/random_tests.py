import os
import unittest

from dnnv import nn
from dnnv import properties
from dnnv.properties import Symbol
from dnnv.properties.context import get_context
from dnnv.verifiers.common import SAT, UNSAT, UNKNOWN
from dnnv.verifiers import neurify, planet, reluplex

from tests.verifiers_tests.utils import has_verifier
from tests.utils import network_artifact_dir, property_artifact_dir

bab, eran = None, None
if has_verifier("bab"):
    import dnnv.verifiers.bab as bab
if has_verifier("eran"):
    import dnnv.verifiers.eran as eran

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))


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
        verifiers = {
            "bab": bab,
            "eran": eran,
            "neurify": neurify,
            "planet": planet,
            "reluplex": reluplex,
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_0.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_fc_1(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {
            "bab": bab,
            "eran": eran,
            "neurify": neurify,
            "planet": planet,
            "reluplex": reluplex,
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_1.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_fc_2(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {
            "bab": bab,
            "eran": eran,
            "neurify": neurify,
            "planet": planet,
            "reluplex": reluplex,
        }
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_fc_2.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_conv_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {"bab": bab, "eran": eran, "neurify": neurify, "planet": planet}
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_conv_0.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_conv_1(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {"bab": bab, "eran": eran, "neurify": neurify, "planet": planet}
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_conv_1.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_conv_2(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {"bab": bab, "eran": eran, "neurify": neurify, "planet": planet}
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_conv_2.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    result = verifier.verify(dnn, phi)
                    self.check_results(result, results)

    def test_random_residual_0(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        verifiers = {"eran": eran, "planet": planet}
        for epsilon in [0.01, 0.1, 0.5, 1.0]:
            os.environ["EPSILON"] = str(epsilon)
            for i in range(RUNS_PER_PROP):
                os.environ["SEED"] = str(i)
                results = []
                for name, verifier in verifiers.items():
                    if not has_verifier(name):
                        continue
                    self.reset_property_context()
                    dnn = nn.parse(network_artifact_dir / "random_residual_0.onnx")
                    phi = properties.parse(property_artifact_dir / "localrobustness.py")
                    if name == "eran":
                        result = verifier.verify(dnn, phi, domain="refinezono")
                    else:
                        result = verifier.verify(dnn, phi)
                    self.check_results(result, results)


if __name__ == "__main__":
    unittest.main()
