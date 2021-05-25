from __future__ import annotations

from .. import install
from ..install.common import gurobi_installer

verinet_runner = """#!{python_venv}/bin/python
import argparse
import numpy as np
import sys

sys.path.insert(0, "{verifier_dir}/VeriNet/")

from src.algorithm.verification_objectives import LocalRobustnessObjective
from src.algorithm.verinet import VeriNet
from src.algorithm.verinet_util import Status
from src.data_loader.onnx_parser import ONNXParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input_bounds")

    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("-p", "--max_procs", "--procs", type=int, default=1)
    parser.add_argument("-T", "--timeout", type=float, default=24 * 60 * 60)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--no_split", action="store_true")
    return parser.parse_args()


def main(args):
    onnx_parser = ONNXParser(args.model)
    model = onnx_parser.to_pytorch()

    input_bounds = np.load(args.input_bounds)
    output_bounds = np.array([[0.0, np.finfo(np.float32).max], [0.0, 0.0]])

    solver = VeriNet(model, max_procs=args.max_procs)
    objective = LocalRobustnessObjective(
        correct_class=0, input_bounds=input_bounds, output_bounds=output_bounds
    )

    solver.verify(
        objective, timeout=args.timeout, verbose=args.verbose, no_split=args.no_split
    )

    if args.output is not None:
        np.save(
            args.output,
            (str(solver.status).split(".")[-1], solver.counter_example),
        )
    print(str(solver.status).split(".")[-1])
    if solver.counter_example is not None:
        print(solver.counter_example)

    print("")
    print(f"Result: {{solver.status}}")
    print(f"Branches explored: {{solver.branches_explored}}")
    print(f"Maximum depth reached: {{solver.max_depth}}")


if __name__ == "__main__":
    main(parse_args())
"""


def configure_install(manager: install.InstallationManager):
    verifier_base_dir = manager.base_dir / "verifiers" / __name__.split(".")[-1]

    with manager.using_python_venv(verifier_base_dir / ".venv"):
        manager.pip_install("pip", extra_args="--upgrade")(manager)

        install_gurobi = gurobi_installer(
            install_dir=manager.base_dir,
            cache_dir=manager.cache_dir,
            version="9.0.2",
            python_venv=manager.active_venv,
            install_python_package=True,
        )
        manager.require_program("grbgetkey", action_if_not_found=install_gurobi)
        manager.require_library("libgurobi90", action_if_not_found=install_gurobi)
        manager.require_header("gurobi_c.h", action_if_not_found=install_gurobi)
        manager.require_python_package("gurobipy", action_if_not_found=install_gurobi)

        manager.pip_install(
            "numba>=0.50,<0.60",
            "onnx>=1.8,<1.9",
            "torch>=1.8,<1.9",
            "torchvision>=0.9,<0.10",
        )(manager)

    with manager.using_base_dir(verifier_base_dir):
        with manager.using_python_venv(verifier_base_dir / ".venv"):
            dnnv_version = "0.4.0"
            install_verinet = install.installer_builder(
                install.wget_download(
                    f"https://github.com/dlshriver/DNNV/archive/refs/tags/{dnnv_version}.tar.gz"
                ),
                install.command(
                    f"tar xf {dnnv_version}.tar.gz --wildcards */third_party/VeriNet --strip-components=2"
                ),
            )
            manager.require_program("verinet", action_if_not_found=install_verinet)

    with open(manager.base_dir / "bin/verinet", "w+") as f:
        f.write(
            verinet_runner.format(
                verifier_dir=verifier_base_dir, python_venv=verifier_base_dir / ".venv"
            )
        )
    (manager.base_dir / "bin/verinet").chmod(0o700)
