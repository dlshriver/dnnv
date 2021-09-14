from __future__ import annotations

import subprocess as sp
import sys

from ..environment import (
    Environment,
    Dependency,
    HeaderDependency,
    LibraryDependency,
    ProgramDependency,
    Installer,
    GurobiInstaller,
)
from ...errors import InstallError, UninstallError

wrapped_python_template = """#!/bin/bash

export GUROBI_HOME={gurobi_home}
export PATH={gurobi_home}/bin:$PATH
export LD_LIBRARY_PATH={gurobi_home}/lib:$LD_LIBRARY_PATH
{python_venv}/bin/python $@
"""

runner_template = """#!{python_venv}/bin/wrappedpython
import argparse
import numpy as np

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


class VeriNetInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        name = "verinet"
        version = "1.0"
        dnnv_version = "v0.4.5"

        cache_dir = env.cache_dir / f"{name}-{version}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        gurobi_path = LibraryDependency("libgurobi91").get_path(env).parent.parent

        python_major_version, python_minor_version = sys.version_info[:2]

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            f"rm -rf {name}",
            f"python -m venv {name}",
            f". {name}/bin/activate",
            "pip install --upgrade pip",
            'pip install "numba>=0.50,<0.60" "onnx>=1.8,<1.9" "torch>=1.8,<1.9" "torchvision>=0.9,<0.10"',
            f"cd {gurobi_path}",
            "python setup.py install",
            f"cd {cache_dir}",
            f"rm -rf {name}",
            f"wget https://github.com/dlshriver/DNNV/archive/refs/tags/{dnnv_version}.tar.gz",
            f"tar xf {dnnv_version}.tar.gz --wildcards */third_party/VeriNet --strip-components=2",
            f"cp -r VeriNet/src {verifier_venv_path}/lib/python{python_major_version}.{python_minor_version}/site-packages/",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")

        with open(verifier_venv_path / "bin" / "wrappedpython", "w+") as f:
            f.write(
                wrapped_python_template.format(
                    python_venv=verifier_venv_path,
                    gurobi_home=envvars.get("GUROBI_HOME", "."),
                )
            )
        (verifier_venv_path / "bin" / "wrappedpython").chmod(0o700)
        with open(installation_path / name, "w+") as f:
            f.write(runner_template.format(python_venv=verifier_venv_path))
        (installation_path / name).chmod(0o700)


def install(env: Environment):
    gurobi_installer = GurobiInstaller("9.1.2")
    env.ensure_dependencies(
        ProgramDependency(
            "verinet",
            installer=VeriNetInstaller(),
            dependencies=(
                HeaderDependency("gurobi_c.h", installer=gurobi_installer),
                LibraryDependency("libgurobi91", installer=gurobi_installer),
                ProgramDependency("grbgetkey", installer=gurobi_installer),
            ),
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
