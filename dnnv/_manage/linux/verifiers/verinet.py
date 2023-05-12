from __future__ import annotations

import subprocess as sp
import sys

from ...errors import InstallError, UninstallError
from ..environment import (
    Dependency,
    Environment,
    Installer,
    ProgramDependency,
)

WRAPPED_PYTHON_TEMPLATE = """#!/bin/bash
{python_venv}/bin/python $@
"""

RUNNER_TEMPLATE = """#!{python_venv}/bin/wrappedpython
import argparse
import numpy as np

from verinet.verification.objective import Objective
from verinet.verification.verinet import VeriNet
from verinet.verification.verifier_util import Status
from verinet.parsers.onnx_parser import ONNXParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input_bounds")

    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("-p", "--max_procs", "--procs", type=int, default=1)
    parser.add_argument('-g', "--use_gpu", "--gpu", default=False, action='store_true')
    parser.add_argument("-T", "--timeout", type=float, default=24 * 60 * 60)

    parser.add_argument("--no_split", action="store_true")
    return parser.parse_args()


def main(args):
    onnx_parser = ONNXParser(args.model)
    model = onnx_parser.to_pytorch()

    input_bounds = np.load(args.input_bounds)

    solver = VeriNet(max_procs=args.max_procs, use_gpu=args.use_gpu)
    objective = Objective(input_bounds, output_size=2, model=model)
    objective.add_constraints(objective.output_vars[0] >= 5e-7)
    objective.add_constraints(objective.output_vars[1] <= 5e-7)
    objective.add_constraints(objective.output_vars[1] >= -5e-7)

    solver.verify(
        objective, timeout=args.timeout
    )

    if args.output is not None:
        np.save(
            args.output,
            np.array([str(solver.status).split(".")[-1], solver.counter_example], dtype=object),
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


class VeriNetInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        name = "verinet"
        version = "1.0"

        cache_dir = env.cache_dir / f"{name}-{version}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)


        python_major_version, python_minor_version = sys.version_info[:2]
        python_version = f"python{python_major_version}.{python_minor_version}"
        site_packages_dir = f"{verifier_venv_path}/lib/{python_version}/site-packages/"

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            f"rm -rf {name}",
            f"python -m venv {name}",
            f". {name}/bin/activate",
            "pip install --upgrade pip",
            (
                "pip install"
                ' "numpy"'
                ' "scipy"'
                ' "setuptools"'
                ' "torch>=1.8"'
                ' "torchvision>=0.9"'
                ' "pillow"'
                ' "psutil"'
                ' "onnx"'
                ' "tqdm"'
                ' "xpress"'
                ' "matplotlib"'
                ' "netron"'
                ' "pipenv"'
                #' "dnnv"' # VeriNet uses dnnv for network simplification, so it has to be installed in the venv as well
                # However, VeriNet tries to use dnnv.nn which fails because "module 'dnnv' has no attribute 'nn'" if
                # this is fixed we can re-add this line so that VeriNet can make use of dnnv
            ),
            f"cd {cache_dir}",
            f"rm -rf VeriNet",
            f"git clone https://github.com/vas-group-imperial/VeriNet",
            f"cd VeriNet",
            f"rm Pipfile.lock", # The Pipfile.lock has package versions that are not distributed anymore
            f"pipenv install --skip-lock",
            f"cp -r verinet {site_packages_dir}",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")

        with open(verifier_venv_path / "bin" / "wrappedpython", "w+") as f:
            f.write(
                WRAPPED_PYTHON_TEMPLATE.format(
                    python_venv=verifier_venv_path,
                )
            )
        (verifier_venv_path / "bin" / "wrappedpython").chmod(0o700)
        with open(installation_path / name, "w+") as f:
            f.write(RUNNER_TEMPLATE.format(python_venv=verifier_venv_path))
        (installation_path / name).chmod(0o700)


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "verinet",
            installer=VeriNetInstaller(),
        )
    )


def uninstall(env: Environment):
    name = "verinet"
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
