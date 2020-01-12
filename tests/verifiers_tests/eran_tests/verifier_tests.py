import os
import unittest

import dnnv.verifiers.eran as verifier

from dnnv import nn
from dnnv import properties
from dnnv.properties import Symbol
from dnnv.verifiers.common import SAT, UNSAT, UNKNOWN

from tests.utils import network_artifact_dir, property_artifact_dir

# TODO : reduce redundant code in verifier_tests files


class ERANVerifierTests(unittest.TestCase):
    def setUp(self):
        self.reset_property_context()
        for varname in ["SHIFT", "INPUT_LAYER", "OUTPUT_LAYER"]:
            if varname in os.environ:
                del os.environ[varname]

    def reset_property_context(self):
        # TODO : refactor property implementation so this can be removed
        # required to ensure concretized symbols don't carry over
        Symbol._instances = {}

    def test_sum_gt_one_localrobustness_shift_left_unsat(self):
        os.environ["SHIFT"] = "-100"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)

    def test_sum_gt_one_localrobustness_shift_right_unsat(self):
        os.environ["SHIFT"] = "100"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)

    def test_sum_gt_one_localrobustness_no_shift_sat(self):
        os.environ["SHIFT"] = "0"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            result = verifier.verify(dnn, phi)
            self.assertIn(result, [UNKNOWN, SAT])

    def test_const_zero_localrobustness(self):
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_zero.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)

    def test_const_one_localrobustness(self):
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)

    def test_a_gt_b_localrobustness_unsat(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        os.environ["SHIFT"] = "np.asarray([[100,0]], dtype=np.float32)"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)
        os.environ["SHIFT"] = "np.asarray([[0,100]], dtype=np.float32)"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)

    def test_a_gt_b_localrobustness_sat(self):
        os.environ["OUTPUT_LAYER"] = "-1"
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            result = verifier.verify(dnn, phi)
            self.assertIn(result, [UNKNOWN, SAT])

    def test_const_zero_ge1_sat(self):
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_zero.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_0.py")
            result = verifier.verify(dnn, phi)
            self.assertIn(result, [UNKNOWN, SAT])

    def test_const_one_ge1_unsat(self):
        for i in range(10):
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_one.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_0.py")
            result = verifier.verify(dnn, phi)
            self.assertEqual(result, UNSAT)


if __name__ == "__main__":
    unittest.main()
