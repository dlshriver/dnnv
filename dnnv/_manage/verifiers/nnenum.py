from __future__ import annotations
import sys

from .. import install

nnenum_runner = """#!{python_venv}/bin/python
import argparse
import numpy as np
import sys

sys.path.insert(0, "{verifier_dir}/nnenum/src/")

from pathlib import Path

from nnenum.enumerate import enumerate_network
from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.specification import Specification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_processes", "--procs", type=int)

    return parser.parse_args()


def main(args):
    Settings.PRINT_PROGRESS = False
    if args.num_processes is not None:
        Settings.NUM_PROCESSES = args.num_processes

    (lb, ub), (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = load_onnx_network(args.model)
    ninputs = A_input.shape[1]

    init_box = np.array(list(zip(lb.flatten("F"), ub.flatten("F"))), dtype=np.float32)
    init_star = LpStar(
        np.eye(ninputs, dtype=np.float32), np.zeros(ninputs, dtype=np.float32), init_box
    )
    for a, b in zip(A_input, b_input):
        a_ = a.reshape(network.get_input_shape()).flatten("F")
        init_star.lpi.add_dense_row(a_, b)

    spec = Specification(A_output, b_output)

    result = enumerate_network(init_star, network, spec)

    print(result.result_str)
    if args.output is not None:
        cex = None
        if result.cinput is not None:
            cex = (
                np.array(list(result.cinput))
                .astype(np.float32)
                .reshape(network.get_input_shape(), order="F")
            )
            print(cex)
        np.save(args.output, (result.result_str, cex))

    return


if __name__ == "__main__":
    main(parse_args())
"""


def configure_install(manager: install.InstallationManager):
    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    verifier_base_dir = manager.base_dir / "verifiers" / __name__.split(".")[-1]
    with manager.using_base_dir(verifier_base_dir):
        with manager.using_python_venv(verifier_base_dir / ".venv"):
            manager.pip_install("pip", extra_args="--upgrade")(manager)
            commit_hash = "81178bc"
            install_nnenum = install.installer_builder(
                install.create_build_dir(
                    manager.base_dir / f"nnenum-{commit_hash}", enter_dir=True
                ),
                install.git_download(
                    "https://github.com/stanleybak/nnenum.git", commit_hash=commit_hash
                ),
                install.command(f". {manager.active_venv}/bin/activate"),
                install.command("pip install -r requirements.txt"),
            )
            manager.require_program("nnenum", action_if_not_found=install_nnenum)

    with open(manager.base_dir / "bin/nnenum", "w+") as f:
        f.write(
            nnenum_runner.format(
                verifier_dir=verifier_base_dir, python_venv=verifier_base_dir / ".venv"
            )
        )
    (manager.base_dir / "bin/nnenum").chmod(0o700)
