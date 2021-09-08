from __future__ import annotations

import subprocess as sp
import sys

from ..environment import (
    Environment,
    Dependency,
    ProgramDependency,
    Installer,
)
from ...errors import InstallError, UninstallError

runner_template = """#!{python_venv}/bin/python
import argparse
import numpy as np

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
    Settings.UNDERFLOW_BEHAVIOR = "warn"
    Settings.PRINT_PROGRESS = False
    if args.num_processes is not None:
        Settings.NUM_PROCESSES = args.num_processes

    (lb, ub), (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = load_onnx_network(args.model)
    ninputs = A_input.shape[1]

    init_box = np.array(
        list(zip(lb.flatten(), ub.flatten())),
        dtype=np.float32,
    )
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
                .reshape(network.get_input_shape())
            )
            print(cex)
        np.save(args.output, (result.result_str, cex))

    return


if __name__ == "__main__":
    main(parse_args())
"""


class NnenumInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "6854a6b0bb766806bb42570230adcbd0ecfab809"
        name = "nnenum"

        cache_dir = env.cache_dir / f"{name}-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        python_major_version, python_minor_version = sys.version_info[:2]

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            f"rm -rf {name}",
            f"python -m venv {name}",
            f". {name}/bin/activate",
            "pip install --upgrade pip",
            f"cd {cache_dir}",
            f"rm -rf {name}",
            "git clone https://github.com/stanleybak/nnenum.git",
            f"cd {name}",
            f"git checkout {commit_hash}",
            "pip install -r requirements.txt",
            f"cp -r src/nnenum {verifier_venv_path}/lib/python{python_major_version}.{python_minor_version}/site-packages/",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")

        with open(installation_path / name, "w+") as f:
            f.write(runner_template.format(python_venv=verifier_venv_path))
        (installation_path / name).chmod(0o700)


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "nnenum",
            installer=NnenumInstaller(),
            dependencies=(ProgramDependency("git"),),
        )
    )


def uninstall(env: Environment):
    name = "nnenum"
    exe_path = env.env_dir / "bin" / name
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError(f"Uninstallation of {name} failed")


__all__ = ["install", "uninstall"]
