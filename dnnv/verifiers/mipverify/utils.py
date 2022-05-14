import tempfile
from collections import defaultdict
from pathlib import Path
from typing import IO, Dict, Optional, Set, Type

import numpy as np
import scipy.io

from dnnv.nn import Operation, OperationGraph, OperationVisitor, operations
from dnnv.verifiers.common import VerifierTranslatorError


class MIPVerifyInputBuilder(OperationVisitor):
    def __init__(self, translator_error=VerifierTranslatorError):
        super().__init__()
        self.lines = []
        self.layers = []
        self.params = {}
        self.op_counts: Dict[str, int] = defaultdict(int)
        self.visited: Set[Operation] = set()
        self.translator_error = translator_error

    def build(
        self,
        julia_file: IO,
        weights_file_path: Path,
        cex_file_path: Path,
        center: np.ndarray,
        linf_ub: float,
        tightening_algorithm: str = "mip",
        optimizer: str = "Gurobi",
    ):
        self.params["input"] = center
        input_shape = tuple(center.shape)

        scipy.io.savemat(weights_file_path, self.params)

        lines = (
            [
                f"using MIPVerify, {optimizer}, JuMP, MAT, MathOptInterface",
                f'param_dict = matread("{weights_file_path}")',
            ]
            + self.lines
            + [
                "nn = Sequential([",
                ",\n".join(self.layers),
                f'], "{weights_file_path.stem}")',
                f'input = reshape(collect(param_dict["input"]), {input_shape})',
                # 'print(nn(input), "\\n")',
                # class 1 (1-indexed) if property is FALSE
                "d = MIPVerify.find_adversarial_example("
                f"nn, input, 1, {optimizer}.Optimizer, Dict(),"
                f" pp=MIPVerify.LInfNormBoundedPerturbationFamily({linf_ub}),"
                f" tightening_algorithm={tightening_algorithm}, norm_order=Inf,"
                " solve_if_predicted_in_targeted=false)",
                "if (d[:PredictedIndex] == 1)",
                f'    MAT.matwrite("{cex_file_path}", Dict("cex" => input))',
                '    print("TRIVIAL\\n")',
                "elseif (d[:SolveStatus] == MathOptInterface.OPTIMAL)",
                f'    MAT.matwrite("{cex_file_path}", Dict("cex" => JuMP.value.(d[:PerturbedInput])))',
                # '    print(JuMP.value.(d[:PerturbedInput]), "\\n")',
                '    print(nn(JuMP.value.(d[:PerturbedInput])), "\\n")',
                '    print(d[:SolveStatus], "\\n")',
                "else",
                '    print(d[:SolveStatus], "\\n")',
                "end",
            ]
        )
        julia_file.write("\n".join(lines))

    def visit(self, operation: Operation):
        if operation in self.visited:
            raise self.translator_error(
                "Multiple computation paths is not currently supported"
            )
        self.visited.add(operation)
        result = super().visit(operation)
        return result

    def generic_visit(self, operation: Operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(
                f"MIPVerify converter not implemented for operation type {type(operation).__name__}"
            )
        return super().generic_visit(operation)

    def visit_Conv(self, operation: operations.Conv):
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        _ = self.visit(operation.x)
        w = self.params[f"{opname}/weight"] = operation.w.transpose((2, 3, 1, 0))
        b = self.params[f"{opname}/bias"] = operation.b

        if any(s != operation.strides[0] for s in operation.strides):
            raise self.translator_error("Multiple stride lengths are not supported")
        assert operation.group == 1
        assert np.all(operation.dilations == 1)
        pads = (
            operation.pads[0],
            operation.pads[2],
            operation.pads[1],
            operation.pads[3],
        )

        self.lines.append(
            f'{opname}_W = reshape(collect(param_dict["{opname}/weight"]), {tuple(w.shape)})'
        )
        self.lines.append(
            f'{opname}_b = reshape(collect(param_dict["{opname}/bias"]), {tuple(b.shape)})'
        )
        self.lines.append(
            f"{opname} = Conv2d({opname}_W, {opname}_b, {operation.strides[0]}, {pads})"
        )
        self.layers.append(f"{opname}")

    def visit_Flatten(self, operation: operations.Flatten):
        assert operation.axis == 1
        _ = self.visit(operation.x)
        output_shape = OperationGraph([operation.x]).output_shape[0]
        if len(output_shape) == 4:
            self.layers.append("Flatten(4, [1, 3, 2, 4])")
        else:
            self.layers.append(f"Flatten({len(output_shape)})")

    def visit_Gemm(self, operation: operations.Gemm):
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        assert operation.alpha == 1.0
        assert operation.beta == 1.0
        assert isinstance(operation.a, Operation) or isinstance(operation.b, Operation)
        assert not isinstance(operation.a, Operation) or not isinstance(
            operation.b, Operation
        )
        if isinstance(operation.a, Operation):
            assert not operation.transpose_a
            _ = self.visit(operation.a)

            weights = operation.b
            if operation.transpose_b:
                weights = weights.T
            self.params[f"{opname}/weight"] = weights

            if operation.c is not None:
                bias = operation.c
            else:
                bias = np.zeros(weights.shape[1], dtype=weights.dtype)
            self.params[f"{opname}/bias"] = bias
        else:
            raise NotImplementedError()

        self.lines.append(
            f'{opname}_W = reshape(collect(param_dict["{opname}/weight"]), {tuple(weights.shape)})'
        )
        self.lines.append(
            f'{opname}_b = reshape(collect(param_dict["{opname}/bias"]), {tuple(bias.shape)})'
        )
        self.lines.append(f"{opname} = Linear({opname}_W, {opname}_b)")
        self.layers.append(f"{opname}")

    def visit_Input(self, operation: operations.Input):
        if len(operation.shape) == 2 and operation.shape[0] in [-1, 1]:
            self.layers.append("Flatten(2)")

    def visit_MaxPool(self, operation: operations.MaxPool):
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        _ = self.visit(operation.x)

        if any(p != 0 for p in operation.pads):
            raise self.translator_error("Padded max pooling is not supported.")
        if any(k != s for k, s in zip(operation.kernel_shape, operation.strides)):
            raise self.translator_error("Max pool stride must be equal to kernel size.")
        s_h, s_w = operation.strides
        self.lines.append(f"{opname} = MaxPool((1, {s_h}, {s_w}, 1))")
        self.layers.append(f"{opname}")

    def visit_Relu(self, operation: operations.Relu):
        _ = self.visit(operation.x)
        self.layers.append("ReLU()")

    def visit_Reshape(self, operation: operations.Reshape):
        assert len(operation.shape) == 2 or len(operation.shape) == 1
        _ = self.visit(operation.x)
        output_shape = OperationGraph([operation.x]).output_shape[0]
        if len(output_shape) == 4:
            self.layers.append("Flatten(4, [1, 3, 2, 4])")
        else:
            self.layers.append(f"Flatten({len(output_shape)})")


def to_mipverify_inputs(
    lb: np.ndarray,
    ub: np.ndarray,
    op_graph: OperationGraph,
    optimizer: str = "Gurobi",
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Dict[str, str]:
    mipverify_inputs = {}

    if lb.ndim == 4:
        lb = lb.transpose((0, 3, 2, 1))
    if ub.ndim == 4:
        ub = ub.transpose((0, 3, 2, 1))
    if np.any(lb < 0) or np.any(ub > 1):
        raise translator_error(
            "Input intervals that extend outside of [0, 1] are not supported"
        )
    dtype = op_graph.input_details[0].dtype
    radii = (ub - lb) / 2
    max_radii = radii.max()
    if not np.allclose(radii.min(), radii.max()):
        raise translator_error(
            "MIPVerify does not support problems with non-hypercube input regions"
        )
    center = ((lb + ub) / 2).astype(dtype)
    assert np.all(center >= 0)
    assert np.all(center <= 1)

    builder = MIPVerifyInputBuilder(translator_error=translator_error)
    _ = op_graph.walk(builder)

    with tempfile.NamedTemporaryFile(
        dir=dirname, suffix=".mat", delete=False
    ) as cex_file:
        with tempfile.NamedTemporaryFile(
            dir=dirname, suffix=".mat", delete=False
        ) as weights_file:
            with tempfile.NamedTemporaryFile(
                mode="w+", dir=dirname, suffix=".jl", delete=False
            ) as julia_file:
                builder.build(
                    julia_file,
                    Path(weights_file.name),
                    Path(cex_file.name),
                    center,
                    max_radii,
                    optimizer=optimizer,
                )
                mipverify_inputs["julia_file_path"] = julia_file.name
            mipverify_inputs["weights_file_path"] = weights_file.name
        mipverify_inputs["cex_file_path"] = cex_file.name

    return mipverify_inputs
