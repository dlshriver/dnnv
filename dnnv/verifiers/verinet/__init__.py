import tempfile

import numpy as np

from dnnv.nn import OperationTransformer, operations
from dnnv.nn.graph import OperationGraph
from dnnv.nn.visitors import EnsureSupportVisitor
from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNKNOWN, UNSAT

from .errors import VerinetError, VerinetTranslatorError


class VeriNet(Verifier):
    translator_error = VerinetTranslatorError
    verifier_error = VerinetError
    parameters = {
        "max_proc": Parameter(int, help="Maximum number of processes to use."),
        "no_split": Parameter(bool, help="Whether or not to do splitting."),
    }

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            op_graph = prop.suffixed_op_graph().simplify()
            supported_operations = [
                operations.Conv,
                operations.Flatten,
                operations.Gemm,
                operations.Input,
                operations.Relu,
                operations.Reshape,
                operations.Shape,
                operations.Sigmoid,
                operations.Tanh,
            ]
            op_graph.walk(
                EnsureSupportVisitor(supported_operations, self.translator_error)
            )

            output_op = op_graph.output_operations[0]
            b = output_op.b
            if output_op.transpose_b:
                b = b.T

            output_op.transpose_b = True
            output_op.b = np.vstack(
                [
                    b,
                    np.zeros(
                        b.T.shape,
                        dtype=b.dtype,
                    ),
                ]
            )
            output_op.c = np.hstack(
                [
                    output_op.c,
                    np.zeros(
                        output_op.c.shape,
                        dtype=output_op.c.dtype,
                    ),
                ]
            )

            class TransposeB(OperationTransformer):
                def visit_Gemm(self, op: operations.Gemm):
                    a = op.a
                    if isinstance(a, operations.Operation):
                        a = self.visit(op.a)
                    b = op.b
                    if isinstance(b, operations.Operation):
                        b = self.visit(op.b)
                    c = op.c
                    if isinstance(c, operations.Operation):
                        c = self.visit(op.c)
                    if op.transpose_b:
                        op.a = a
                        op.b = b
                        op.c = c
                        return op
                    return operations.Gemm(
                        a,
                        b.T,
                        c,
                        transpose_a=op.transpose_a,
                        transpose_b=True,
                        alpha=op.alpha,
                        beta=op.beta,
                    )

            op_graph = OperationGraph(op_graph.walk(TransposeB()))
            op_graph.export_onnx(onnx_model_file.name)

        lb = prop.input_constraint.lower_bounds[0].flatten().copy()
        ub = prop.input_constraint.upper_bounds[0].flatten().copy()

        input_bounds = np.array(list(zip(lb, ub))).reshape(
            tuple(prop.input_constraint.lower_bounds[0].shape) + (2,)
        )
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as input_bounds_file:
            np.save(input_bounds_file.name, input_bounds)

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file
        args = (
            "verinet",
            onnx_model_file.name,
            input_bounds_file.name,
            "-o",
            self._tmp_output_file.name,
        )
        if (
            "max_proc" in self.parameter_values
            and self.parameter_values["max_proc"] is not None
        ):
            value = self.parameter_values["max_proc"]
            args += (f"--max_procs={value}",)
        if "no_split" in self.parameter_values and self.parameter_values["no_split"]:
            args += ("--no_split",)
        return args

    def parse_results(self, prop, results):
        try:
            status, cex = np.load(self._tmp_output_file.name, allow_pickle=True)
            if status == "Safe":
                return UNSAT, None
            elif status == "Unsafe":
                input_shape, input_dtype = prop.op_graph.input_details[0]
                return SAT, cex.reshape(input_shape)
            elif status == "Unknown":
                return UNKNOWN, None
            elif status == "Undecided":
                raise self.verifier_error("Undecided")
            raise self.translator_error(f"Unknown verification result: {status}")
        finally:
            del self._tmp_output_file
