import io
import sys
import unittest

from dnnv.nn import parse, OperationGraph
from dnnv.nn.visitors import PrintVisitor
from ...utils import network_artifact_dir as artifact_dir

RESNET34_TEXT = """Input_0                         : Input([  1   3 224 224], dtype=float32)
Conv_0                          : Conv(Input_0)
BatchNormalization_0            : BatchNormalization(Conv_0)
Relu_0                          : Relu(BatchNormalization_0)
MaxPool_0                       : MaxPool(Relu_0)
Conv_1                          : Conv(MaxPool_0)
BatchNormalization_1            : BatchNormalization(Conv_1)
Relu_1                          : Relu(BatchNormalization_1)
Conv_2                          : Conv(Relu_1)
BatchNormalization_2            : BatchNormalization(Conv_2)
Add_0                           : Add(BatchNormalization_2, MaxPool_0)
Relu_2                          : Relu(Add_0)
Conv_3                          : Conv(Relu_2)
BatchNormalization_3            : BatchNormalization(Conv_3)
Relu_3                          : Relu(BatchNormalization_3)
Conv_4                          : Conv(Relu_3)
BatchNormalization_4            : BatchNormalization(Conv_4)
Add_1                           : Add(BatchNormalization_4, Relu_2)
Relu_4                          : Relu(Add_1)
Conv_5                          : Conv(Relu_4)
BatchNormalization_5            : BatchNormalization(Conv_5)
Relu_5                          : Relu(BatchNormalization_5)
Conv_6                          : Conv(Relu_5)
BatchNormalization_6            : BatchNormalization(Conv_6)
Add_2                           : Add(BatchNormalization_6, Relu_4)
Relu_6                          : Relu(Add_2)
Conv_7                          : Conv(Relu_6)
BatchNormalization_7            : BatchNormalization(Conv_7)
Relu_7                          : Relu(BatchNormalization_7)
Conv_8                          : Conv(Relu_7)
BatchNormalization_8            : BatchNormalization(Conv_8)
Conv_9                          : Conv(Relu_6)
BatchNormalization_9            : BatchNormalization(Conv_9)
Add_3                           : Add(BatchNormalization_8, BatchNormalization_9)
Relu_8                          : Relu(Add_3)
Conv_10                         : Conv(Relu_8)
BatchNormalization_10           : BatchNormalization(Conv_10)
Relu_9                          : Relu(BatchNormalization_10)
Conv_11                         : Conv(Relu_9)
BatchNormalization_11           : BatchNormalization(Conv_11)
Add_4                           : Add(BatchNormalization_11, Relu_8)
Relu_10                         : Relu(Add_4)
Conv_12                         : Conv(Relu_10)
BatchNormalization_12           : BatchNormalization(Conv_12)
Relu_11                         : Relu(BatchNormalization_12)
Conv_13                         : Conv(Relu_11)
BatchNormalization_13           : BatchNormalization(Conv_13)
Add_5                           : Add(BatchNormalization_13, Relu_10)
Relu_12                         : Relu(Add_5)
Conv_14                         : Conv(Relu_12)
BatchNormalization_14           : BatchNormalization(Conv_14)
Relu_13                         : Relu(BatchNormalization_14)
Conv_15                         : Conv(Relu_13)
BatchNormalization_15           : BatchNormalization(Conv_15)
Add_6                           : Add(BatchNormalization_15, Relu_12)
Relu_14                         : Relu(Add_6)
Conv_16                         : Conv(Relu_14)
BatchNormalization_16           : BatchNormalization(Conv_16)
Relu_15                         : Relu(BatchNormalization_16)
Conv_17                         : Conv(Relu_15)
BatchNormalization_17           : BatchNormalization(Conv_17)
Conv_18                         : Conv(Relu_14)
BatchNormalization_18           : BatchNormalization(Conv_18)
Add_7                           : Add(BatchNormalization_17, BatchNormalization_18)
Relu_16                         : Relu(Add_7)
Conv_19                         : Conv(Relu_16)
BatchNormalization_19           : BatchNormalization(Conv_19)
Relu_17                         : Relu(BatchNormalization_19)
Conv_20                         : Conv(Relu_17)
BatchNormalization_20           : BatchNormalization(Conv_20)
Add_8                           : Add(BatchNormalization_20, Relu_16)
Relu_18                         : Relu(Add_8)
Conv_21                         : Conv(Relu_18)
BatchNormalization_21           : BatchNormalization(Conv_21)
Relu_19                         : Relu(BatchNormalization_21)
Conv_22                         : Conv(Relu_19)
BatchNormalization_22           : BatchNormalization(Conv_22)
Add_9                           : Add(BatchNormalization_22, Relu_18)
Relu_20                         : Relu(Add_9)
Conv_23                         : Conv(Relu_20)
BatchNormalization_23           : BatchNormalization(Conv_23)
Relu_21                         : Relu(BatchNormalization_23)
Conv_24                         : Conv(Relu_21)
BatchNormalization_24           : BatchNormalization(Conv_24)
Add_10                          : Add(BatchNormalization_24, Relu_20)
Relu_22                         : Relu(Add_10)
Conv_25                         : Conv(Relu_22)
BatchNormalization_25           : BatchNormalization(Conv_25)
Relu_23                         : Relu(BatchNormalization_25)
Conv_26                         : Conv(Relu_23)
BatchNormalization_26           : BatchNormalization(Conv_26)
Add_11                          : Add(BatchNormalization_26, Relu_22)
Relu_24                         : Relu(Add_11)
Conv_27                         : Conv(Relu_24)
BatchNormalization_27           : BatchNormalization(Conv_27)
Relu_25                         : Relu(BatchNormalization_27)
Conv_28                         : Conv(Relu_25)
BatchNormalization_28           : BatchNormalization(Conv_28)
Add_12                          : Add(BatchNormalization_28, Relu_24)
Relu_26                         : Relu(Add_12)
Conv_29                         : Conv(Relu_26)
BatchNormalization_29           : BatchNormalization(Conv_29)
Relu_27                         : Relu(BatchNormalization_29)
Conv_30                         : Conv(Relu_27)
BatchNormalization_30           : BatchNormalization(Conv_30)
Conv_31                         : Conv(Relu_26)
BatchNormalization_31           : BatchNormalization(Conv_31)
Add_13                          : Add(BatchNormalization_30, BatchNormalization_31)
Relu_28                         : Relu(Add_13)
Conv_32                         : Conv(Relu_28)
BatchNormalization_32           : BatchNormalization(Conv_32)
Relu_29                         : Relu(BatchNormalization_32)
Conv_33                         : Conv(Relu_29)
BatchNormalization_33           : BatchNormalization(Conv_33)
Add_14                          : Add(BatchNormalization_33, Relu_28)
Relu_30                         : Relu(Add_14)
Conv_34                         : Conv(Relu_30)
BatchNormalization_34           : BatchNormalization(Conv_34)
Relu_31                         : Relu(BatchNormalization_34)
Conv_35                         : Conv(Relu_31)
BatchNormalization_35           : BatchNormalization(Conv_35)
Add_15                          : Add(BatchNormalization_35, Relu_30)
Relu_32                         : Relu(Add_15)
GlobalAveragePool_0             : GlobalAveragePool(Relu_32)
Flatten_0                       : Flatten(GlobalAveragePool_0)
Gemm_0                          : Gemm(Flatten_0, ndarray(shape=(1000, 512)), ndarray(shape=(1000,)))
"""


class PrintVisitorTests(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.maxDiff = 15000

    def tearDown(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_resnet34(self):
        op_graph = parse(artifact_dir / "resnet34.onnx")
        op_graph.pprint()
        self.assertEqual(self.stdout.getvalue(), RESNET34_TEXT)


if __name__ == "__main__":
    unittest.main()
