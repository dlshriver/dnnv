import os

from system_tests.utils import network_artifact_dir, property_artifact_dir

from dnnv import nn, properties
from dnnv.properties import get_context
from dnnv.verifiers import SAT, UNKNOWN, UNSAT

RUNS_PER_PROP = int(os.environ.get("_DNNV_TEST_RUNS_PER_PROP", "1"))


class VerifierTests:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = None
        self.is_complete = False
        self.initialize()

    def initialize(self):
        raise NotImplementedError()

    def setUp(self):
        self.reset_property_context()
        os.environ["SEED"] = "1"
        os.environ["SHIFT"] = "0"
        os.environ["INPUT_LAYER"] = "0"
        os.environ["OUTPUT_LAYER"] = "None"

    def tearDown(self):
        self.reset_property_context()
        for varname in [
            "SEED",
            "SHIFT",
            "INPUT_LAYER",
            "OUTPUT_LAYER",
            "INPUT_LB",
            "INPUT_UB",
            "OUTPUT_LB",
        ]:
            if varname in os.environ:
                os.environ.pop(varname)

    def reset_property_context(self):
        get_context().reset()

    def test_sum_gt_one_localrobustness_shift_left_unsat(self):
        os.environ["SHIFT"] = "-100"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_sum_gt_one_localrobustness_shift_right_unsat(self):
        os.environ["SHIFT"] = "100"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_sum_gt_one_localrobustness_no_shift_sat(self):
        os.environ["SHIFT"] = "0"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "sum_gt_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            if self.is_complete:
                self.assertEqual(result, SAT)
            else:
                self.assertIn(result, [UNKNOWN, SAT])

    def test_const_zero_localrobustness(self):
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_zero.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_const_one_localrobustness(self):
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_one.onnx")
            phi = properties.parse(
                property_artifact_dir / "regression_localrobustness_0.py"
            )
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_a_gt_b_localrobustness_unsat(self):
        os.environ["SHIFT"] = "[100,0]"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_1.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)
        os.environ["SHIFT"] = "[0,100]"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_1.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_a_gt_b_localrobustness_sat(self):
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_0.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            if self.is_complete:
                self.assertEqual(result, SAT)
            else:
                self.assertIn(result, [UNKNOWN, SAT])
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "class_localrobustness_1.py")
            phi.concretize(N=dnn.simplify()[:-1])
            result, _ = self.verifier.verify(phi)
            if self.is_complete:
                self.assertEqual(result, SAT)
            else:
                self.assertIn(result, [UNKNOWN, SAT])

    def test_const_zero_ge1_sat(self):
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_zero.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_0.py")
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            if self.is_complete:
                self.assertEqual(result, SAT)
            else:
                self.assertIn(result, [UNKNOWN, SAT])
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_zero.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_1.py")
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            if self.is_complete:
                self.assertEqual(result, SAT)
            else:
                self.assertIn(result, [UNKNOWN, SAT])

    def test_const_one_ge1_unsat(self):
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_one.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_0.py")
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "const_one.onnx")
            phi = properties.parse(property_artifact_dir / "output_ge1_1.py")
            phi.concretize(N=dnn.simplify())
            result, _ = self.verifier.verify(phi)
            self.assertEqual(result, UNSAT)

    def test_a_gt_b_output_greater_than_x_unsat(self):
        os.environ["OUTPUT_LB"] = "-0.001"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "output_greater_than_X.py")
            phi.concretize(N=dnn[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertIn(result, [UNKNOWN, UNSAT])

    def test_a_gt_b_output_greater_than_x_sat(self):
        os.environ["OUTPUT_LB"] = "1.0"
        for i in range(RUNS_PER_PROP):
            os.environ["SEED"] = str(i + 1)
            self.reset_property_context()
            dnn = nn.parse(network_artifact_dir / "a_gt_b.onnx")
            phi = properties.parse(property_artifact_dir / "output_greater_than_X.py")
            phi.concretize(N=dnn[:-1])
            result, _ = self.verifier.verify(phi)
            self.assertIn(result, [UNKNOWN, SAT])
