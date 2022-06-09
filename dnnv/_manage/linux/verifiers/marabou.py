from __future__ import annotations

import subprocess as sp
import sys

from ...errors import InstallError, UninstallError
from ..environment import (
    Dependency,
    Environment,
    GNUInstaller,
    Installer,
    LibraryDependency,
    OpenBLASInstaller,
    ProgramDependency,
)

MARABOU_RUNNER = """#!{python_venv}/bin/python
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
    outputVars = network.outputVars[0]

    for x, l, u in zip(inputVars.flatten(), lb, ub):
        network.setLowerBound(x, l)
        network.setUpperBound(x, u)

    for i, (a, b) in enumerate(zip(A_input, b_input)):
        network.addInequality(list(inputVars.flatten()), list(a), b)

    for i, (a, b) in enumerate(zip(A_output, b_output)):
        network.addInequality(list(outputVars.flatten()), list(a), b)

    options = MarabouCore.Options()
    options._numWorkers = args.num_workers
    result_str, vals, stats = network.solve(options=options)
    print(result_str)
    is_unsafe = result_str == "sat"

    if args.output is not None:
        cex = None
        if is_unsafe:
            cex = np.zeros_like(inputVars, dtype=np.float32)
            for flat_index, multi_index in enumerate(np.ndindex(cex.shape)):
                cex[multi_index] = vals[flat_index]
            print(cex)
        np.save(
            args.output,
            (result_str, cex),
        )


if __name__ == "__main__":
    main(parse_args())
"""


class MarabouInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "492c1b8c703c8a383f421468a104c34710e6d26d"

        cache_dir = env.cache_dir / f"marabou-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "marabou"
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        libopenblas_path = LibraryDependency("libopenblas").get_path(env)
        assert libopenblas_path is not None
        openblas_path = libopenblas_path.parent.parent

        python_major_version, python_minor_version, *_ = sys.version_info
        python_version = f"python{python_major_version}.{python_minor_version}"
        site_packages_dir = f"{verifier_venv_path}/lib/{python_version}/site-packages/"

        marabou_url = "https://github.com/NeuralNetworkVerification/Marabou.git"

        build_dir = cache_dir / f"Marabou/build-{python_version}"
        openblas_vars = f"-D OPENBLAS_DIR={build_dir}/OpenBlas"

        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            "rm -rf marabou",
            "python -m venv marabou",
            ". marabou/bin/activate",
            "pip install --upgrade pip",
            (
                "pip install"
                ' "numpy>=1.19,<1.22"'
                ' "onnx>=1.8,<1.12"'
                ' "onnxruntime>=1.7,<1.12"'
                ' "protobuf<=3.20"'
            ),
            f"cd {cache_dir}",
            f"if [ ! -e Marabou ]",
            f"then git clone {marabou_url}",
            "cd Marabou",
            f"git checkout {commit_hash}",
            "fi",
            f"if [ ! -e {build_dir} ]",
            f"then mkdir -p {build_dir}",
            f"cd {build_dir}",
            "mkdir -p OpenBlas",
            f"ln -s {openblas_path} {build_dir}/OpenBlas/installed",
            f"cmake {openblas_vars} ..",
            "cmake --build .",
            f"else cd {build_dir}",
            "fi",
            f"cp -r ../maraboupy {site_packages_dir}",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError("Installation of marabou failed")

        with open(installation_path / "marabou", "w+") as f:
            f.write(MARABOU_RUNNER.format(python_venv=verifier_venv_path))
        (installation_path / "marabou").chmod(0o700)


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "marabou",
            installer=MarabouInstaller(),
            dependencies=(
                ProgramDependency("make"),
                ProgramDependency("gcc"),
                ProgramDependency("git"),
                ProgramDependency("curl", min_version="7.16.0"),
                LibraryDependency(
                    "libopenblas",
                    installer=OpenBLASInstaller("0.3.19"),
                    allow_from_system=False,
                ),
                ProgramDependency(
                    "cmake",
                    installer=GNUInstaller(
                        "cmake",
                        "3.18.2",
                        (
                            "https://github.com/Kitware/CMake/"
                            "releases/download/v3.18.2/cmake-3.18.2.tar.gz"
                        ),
                    ),
                    min_version="3.12.0",
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
