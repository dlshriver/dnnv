from __future__ import annotations

import subprocess as sp
import sys

from ..environment import (
    Environment,
    Dependency,
    LibraryDependency,
    ProgramDependency,
    Installer,
    GNUInstaller,
    OpenBLASInstaller,
)
from ...errors import InstallError, UninstallError

marabou_runner = """#!{python_venv}/bin/python
import argparse
import numpy as np

from maraboupy import Marabou, MarabouCore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_workers", "--workers", type=int, default=1)

    return parser.parse_args()


def main(args):
    (lb, ub), (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = Marabou.read_onnx(args.model)

    inputVars = network.inputVars[0]
    outputVars = network.outputVars

    for x, l, u in zip(inputVars.flatten(), lb, ub):
        network.setLowerBound(x, l)
        network.setUpperBound(x, u)

    for i, (a, b) in enumerate(zip(A_input, b_input)):
        network.addInequality(list(inputVars.flatten()), list(a), b)

    for i, (a, b) in enumerate(zip(A_output, b_output)):
        network.addInequality(list(outputVars.flatten()), list(a), b)

    options = MarabouCore.Options()
    options._numWorkers = args.num_workers
    result = network.solve(options=options)

    is_unsafe = bool(result[0])
    print("UNSAFE" if is_unsafe else "SAFE")

    if args.output is not None:
        cex = None
        if is_unsafe:
            cex = np.zeros_like(inputVars, dtype=np.float32)
            for flat_index, multi_index in enumerate(np.ndindex(cex.shape)):
                cex[multi_index] = result[0][flat_index]
            print(cex)
        np.save(
            args.output,
            (is_unsafe, cex),
        )


if __name__ == "__main__":
    main(parse_args())
"""


class MarabouInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "b0e29fb43b6722dfe9b5a90cc1353990aa732327"

        cache_dir = env.cache_dir / f"marabou-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "marabou"
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        openblas_path = LibraryDependency("libopenblas").get_path(env).parent.parent

        python_major_version, python_minor_version = sys.version_info[:2]

        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            "rm -rf marabou",
            "python -m venv marabou",
            ". marabou/bin/activate",
            "pip install --upgrade pip",
            'pip install "numpy>=1.19,<1.20" "onnx>=1.8,<1.9" "onnxruntime>=1.7,<1.8"',
            f"cd {cache_dir}",
            "rm -rf Marabou",
            "git clone https://github.com/NeuralNetworkVerification/Marabou.git",
            "cd Marabou",
            f"git checkout {commit_hash}",
            "rm -rf build",
            "mkdir -p build",
            "cd build",
            "mkdir -p OpenBlas",
            f"rm -f {cache_dir}/Marabou/build/OpenBlas/installed",
            f"ln -s {openblas_path} {cache_dir}/Marabou/build/OpenBlas/installed",
            f"cmake -D OPENBLAS_DIR={cache_dir}/Marabou/build/OpenBlas ..",
            "cmake --build .",
            f"cp -r ../maraboupy {verifier_venv_path}/lib/python{python_major_version}.{python_minor_version}/site-packages/",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of marabou failed")

        with open(installation_path / "marabou", "w+") as f:
            f.write(marabou_runner.format(python_venv=verifier_venv_path))
        (installation_path / "marabou").chmod(0o700)


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "marabou",
            installer=MarabouInstaller(),
            dependencies=(
                ProgramDependency("git"),
                LibraryDependency("libopenblas", installer=OpenBLASInstaller("0.3.9")),
                ProgramDependency(
                    "cmake",
                    installer=GNUInstaller(
                        "cmake",
                        "3.18.2",
                        "https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz",
                    ),
                ),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "marabou"
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "marabou"
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of marabou failed")


__all__ = ["install", "uninstall"]
