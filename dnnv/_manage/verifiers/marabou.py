from __future__ import annotations

from .. import install

marabou_runner = """#!{python_venv}/bin/python
import argparse
import numpy as np
import sys

sys.path.insert(0, "{verifier_dir}/Marabou")

from maraboupy import Marabou, MarabouCore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_workers", "--workers", type=int, default=1)

    return parser.parse_args()


def main(args):
    (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = Marabou.read_onnx(args.model)

    inputVars = network.inputVars[0]
    outputVars = network.outputVars

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


def configure_install(manager: install.InstallationManager):
    verifier_base_dir = manager.base_dir / "verifiers" / __name__.split(".")[-1]
    with manager.using_python_venv(verifier_base_dir / ".venv"):
        manager.pip_install("pip", extra_args="--upgrade")(manager)

    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    manager.require_program(
        "cmake",
        action_if_not_found=manager.gnu_install(
            "cmake",
            "3.18.2",
            "https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz",
        ),
    )

    with manager.using_base_dir(verifier_base_dir):
        with manager.using_python_venv(path=verifier_base_dir / ".venv"):
            manager.pip_install(
                "numpy>=1.19,<1.20",
                "onnx>=1.8,<1.9",
                "onnxruntime>=1.7,<1.8",
            )(manager)
            commit_hash = "754cd96"
            install_marabou = install.installer_builder(
                install.command(f"cd {manager.base_dir}"),
                install.git_download(
                    "https://github.com/NeuralNetworkVerification/Marabou",
                    commit_hash=commit_hash,
                ),
                install.command("rm -rf build; mkdir -p build"),
                install.command("cd build"),
                install.command("cmake .."),
                install.command("cmake --build ."),
            )
            manager.require_program("marabou", action_if_not_found=install_marabou)

    with open(manager.base_dir / "bin/marabou", "w+") as f:
        f.write(
            marabou_runner.format(
                verifier_dir=verifier_base_dir, python_venv=verifier_base_dir / ".venv"
            )
        )
    (manager.base_dir / "bin/marabou").chmod(0o700)
