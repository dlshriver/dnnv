from __future__ import annotations

import subprocess as sp

from ...errors import InstallError, UninstallError
from ..environment import (
    Dependency,
    Environment,
    GurobiInstaller,
    HeaderDependency,
    Installer,
    LibraryDependency,
    ProgramDependency,
)

GUROBI_PYTHON_TEMPLATE = """#!/bin/bash

export GUROBI_HOME={gurobi_home}
export PATH={gurobi_home}/bin:$PATH
export LD_LIBRARY_PATH={gurobi_home}/lib:$LD_LIBRARY_PATH
{python_venv}/bin/python $@
"""

BAB_RUNNER_TEMPLATE = """#!{python_venv}/bin/gurobipython
import argparse
import numpy as np

from plnn.branch_and_bound import bab
from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from plnn.model import load_and_simplify
from plnn.network_linear_approximation import LinearizedNetwork

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")

    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("--smart_branching", type=bool, default=False)
    parser.add_argument("--reluify_maxpools", type=bool, default=False)

    return parser.parse_args()


def main(args):
    with open(args.model) as input_file:
        network, domain = load_and_simplify(input_file, LinearizedNetwork)

    if args.reluify_maxpools:
        network.remove_maxpools(domain)

    smart_brancher = None
    if args.smart_branching:
        with open(args.model) as input_file:
            smart_brancher, _ = load_and_simplify(
                input_file, LooseDualNetworkApproximation
            )
        smart_brancher.remove_maxpools(domain)

    epsilon = 1e-2
    decision_bound = 0
    min_lb, min_ub, ub_point, nb_visited_states = bab(
        network, domain, epsilon, decision_bound, smart_brancher
    )

    if min_lb > 0:
        print("safe")
    elif min_ub <= 0:
        candidate_ctx = ub_point.view(1, -1)
        val = network.net(candidate_ctx)
        margin = val.squeeze().item()
        if margin > 0:
            print("error")
        else:
            print("unsafe")
        if args.output is not None:
            np.save(args.output, candidate_ctx.cpu().detach().numpy())
    else:
        print("unknown")


if __name__ == "__main__":
    main(parse_args())
"""


class BaBInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "d0e20eed8d395c723d7b2903746feb7d0ec7db1c"

        cache_dir = env.cache_dir / f"bab-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "bab"
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        libgurobi_path = LibraryDependency("libgurobi91").get_path(env)
        assert libgurobi_path is not None
        gurobi_path = libgurobi_path.parent.parent

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            "rm -rf bab",
            "python -m venv bab",
            ". bab/bin/activate",
            "pip install --upgrade pip",
            (
                "pip install"
                ' "numpy>=1.19,<1.20"'
                ' "torch>=1.8,<1.9"'
                ' "sh>=1.14,<1.15"'
                ' "scipy>=1.6,<1.7"'
            ),
            f"cd {gurobi_path}",
            "python setup.py install",
            f"cd {cache_dir}",
            "rm -rf PLNN-verification",
            "git clone https://github.com/oval-group/PLNN-verification.git",
            "cd PLNN-verification",
            f"git checkout {commit_hash}",
            "sed -i 's#torch==0.4.0#torch>=0.4.0#' setup.py",
            "python setup.py install",
            "rm -rf convex_adversarial",
            "git clone https://github.com/locuslab/convex_adversarial.git",
            "cd convex_adversarial",
            "git checkout 45a0dca",
            "python setup.py install",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError("Installation of bab failed")

        with open(verifier_venv_path / "bin" / "gurobipython", "w+") as f:
            f.write(
                GUROBI_PYTHON_TEMPLATE.format(
                    python_venv=verifier_venv_path,
                    gurobi_home=envvars.get("GUROBI_HOME", "."),
                )
            )
        (verifier_venv_path / "bin" / "gurobipython").chmod(0o700)
        with open(installation_path / "bab", "w+") as f:
            f.write(BAB_RUNNER_TEMPLATE.format(python_venv=verifier_venv_path))
        (installation_path / "bab").chmod(0o700)


def install(env: Environment):
    gurobi_installer = GurobiInstaller("9.1.2")
    env.ensure_dependencies(
        ProgramDependency(
            "bab",
            installer=BaBInstaller(),
            dependencies=(
                ProgramDependency("git"),
                ProgramDependency("curl", min_version="7.16.0"),
                HeaderDependency("gurobi_c.h", installer=gurobi_installer),
                LibraryDependency("libgurobi91", installer=gurobi_installer),
                ProgramDependency("grbgetkey", installer=gurobi_installer),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "bab"
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "bab"
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of bab failed")


__all__ = ["install", "uninstall"]
