from __future__ import annotations

from .. import install
from ..install.common import gurobi_installer

bab_runner = """#!{python_venv}/bin/python
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
            "numpy>=1.19,<1.20",
            "torch>=1.8,<1.9",
            "sh>=1.14,<1.15",
            "scipy>=1.6,<1.7",
        )(manager)

    with manager.using_base_dir(verifier_base_dir):
        with manager.using_python_venv(verifier_base_dir / ".venv"):
            plnn_commit_hash = "d0e20ee"
            ca_commit_hash = "45a0dca"
            install_verinet = install.installer_builder(
                install.git_download(
                    "https://github.com/oval-group/PLNN-verification.git",
                    commit_hash=plnn_commit_hash,
                ),
                install.command(f"sed -i 's#torch==0.4.0#torch>=0.4.0#' setup.py"),
                install.command(f". {manager.active_venv}/bin/activate"),
                install.command("python setup.py install"),
                install.command("rm -rf convex_adversarial"),
                install.git_download(
                    "https://github.com/locuslab/convex_adversarial",
                    commit_hash=ca_commit_hash,
                ),
                install.command("python setup.py install"),
            )
            manager.require_program("bab", action_if_not_found=install_verinet)

    with open(manager.base_dir / "bin/bab", "w+") as f:
        f.write(
            bab_runner.format(
                verifier_dir=verifier_base_dir, python_venv=verifier_base_dir / ".venv"
            )
        )
    (manager.base_dir / "bin/bab").chmod(0o700)
