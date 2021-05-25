from __future__ import annotations

from .. import install
from ..install import git_download
from ..install.common import gurobi_installer

eran_runner = """#!{python_venv}/bin/python
import argparse
import numpy as np
import sys

sys.path.insert(0, "{verifier_dir}/eran/tf_verify")

from eran import ERAN
from read_net_file import read_onnx_net


def parse_args():
    parser = argparse.ArgumentParser("eran_runner")
    parser.add_argument("network", type=str, help="the onnx model to verify")
    parser.add_argument(
        "input_constraints", type=str, help="numpy file containing input constraints"
    )
    parser.add_argument(
        "--domain", type=str, default="deepzono", help="abstract domain to use"
    )
    parser.add_argument(
        "--timeout_lp", type=float, default=1.0, help="time limit for the LP solver"
    )
    parser.add_argument(
        "--timeout_milp", type=float, default=1.0, help="time limit for the MILP solver"
    )
    parser.add_argument(
        "--use_default_heuristic",
        type=bool,
        default=True,
        help="whether or not to use the default heuristics",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="where to store a found counter example"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, is_conv = read_onnx_net(args.network)

    eran = ERAN(model, is_onnx=True)

    spec_lb, spec_ub = np.load(args.input_constraints)

    label, nn, nlb, nub, _, _ = eran.analyze_box(
        spec_lb.flatten().copy(),
        spec_ub.flatten().copy(),
        args.domain,
        args.timeout_lp,
        args.timeout_milp,
        args.use_default_heuristic,
    )

    output_lower_bound = np.asarray(nlb[-1]).item()
    output_upper_bound = np.asarray(nub[-1]).item()

    if output_lower_bound > 0:
        print("safe")
    elif output_upper_bound <= 0:
        print("unsafe")
        if args.output is not None:
            # all inputs violate property, choose center
            cex = (spec_lb + spec_ub) / 2
            np.save(args.output, cex)
    else:
        print("unknown")


if __name__ == "__main__":
    main()
"""


def configure_install(manager: install.InstallationManager):
    verifier_base_dir = manager.base_dir / "verifiers" / __name__.split(".")[-1]
    with manager.using_python_venv(verifier_base_dir / ".venv"):
        manager.pip_install("pip", extra_args="--upgrade")(manager)

    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    manager.require_program(
        "m4",
        action_if_not_found=manager.gnu_install(
            "m4", "1.4.1", "https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz"
        ),
    )

    gnu_install_gmp = manager.gnu_install(
        "gmp", "6.1.2", "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    )
    manager.require_library("libgmp", action_if_not_found=gnu_install_gmp)
    manager.require_header("gmp.h", action_if_not_found=gnu_install_gmp)

    gnu_install_mpfr = manager.gnu_install(
        "mpfr", "4.1.0", "https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz"
    )
    manager.require_library("libmpfr", action_if_not_found=gnu_install_mpfr)
    manager.require_header("mpfr.h", action_if_not_found=gnu_install_mpfr)

    gnu_install_cddlib = manager.gnu_install(
        "cddlib",
        "0.94j",
        "https://github.com/cddlib/cddlib/releases/download/0.94j/cddlib-0.94j.tar.gz",
    )
    manager.require_library("libcdd", action_if_not_found=gnu_install_cddlib)
    manager.require_header("cdd.h", action_if_not_found=gnu_install_cddlib)

    with manager.using_python_venv(path=verifier_base_dir / ".venv"):
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

    commit_hash = "19ee79b"
    gmp_prefix = ""
    if (manager.base_dir / "lib" / "libgmp.so").exists() or (
        manager.base_dir / "include" / "gmp.h"
    ).exists():
        gmp_prefix = f"-gmp-prefix {manager.base_dir}"
    mpfr_prefix = ""
    if (manager.base_dir / "lib" / "libmpfr.so").exists() or (
        manager.base_dir / "include" / "mpfr.h"
    ).exists():
        mpfr_prefix = f"-mpfr-prefix {manager.base_dir}"
    cdd_prefix = ""
    if (manager.base_dir / "lib" / "libcdd.so").exists() or (
        manager.base_dir / "include" / "cdd.h"
    ).exists():
        cdd_prefix = f"-cdd-prefix {manager.base_dir}"
    install_elina = install.installer_builder(
        install.create_build_dir(
            manager.cache_dir / f"elina-{commit_hash}", enter_dir=True
        ),
        install.git_download(
            "https://github.com/eth-sri/ELINA.git", commit_hash=commit_hash
        ),
        install.command(
            "git revert -n b347e3c"
        ),  # undoing commits that force reliance on local files
        install.command(
            "git revert -n 16d2a6d"
        ),  # undoing commits that force reliance on local files bda9f58
        install.command(
            "git revert -n bda9f58"
        ),  # undoing commits that force reliance on local files
        # TODO: check for cuda
        install.command(
            f'LDFLAGS="-L{manager.base_dir}/lib" ./configure -use-deeppoly -use-gurobi -use-fconv -prefix {manager.cache_dir}/elina-{commit_hash} {gmp_prefix} {mpfr_prefix} {cdd_prefix}'
        ),
        install.command(
            f"sed -i 's#CDD_PREFIX = .*$#CDD_PREFIX = {manager.base_dir}/include -L{manager.base_dir}/lib#' Makefile.config"
        ),
        install.make_install(),
        install.copy_install(
            build_dir=manager.cache_dir / f"elina-{commit_hash}",
            install_dir=manager.base_dir,
        ),
        install.command(
            f"cd {manager.cache_dir}/elina-{commit_hash}/ELINA/python_interface"
        ),
        install.command(
            f"cp $(ls -p | grep -v /) {verifier_base_dir}/.venv/lib/python3.7/site-packages/"
        ),
    )
    manager.require_library("libzonoml", action_if_not_found=install_elina)

    with manager.using_base_dir(verifier_base_dir):
        with manager.using_python_venv(path=verifier_base_dir / ".venv"):
            manager.pip_install(
                "numpy>=1.19,<1.20",
                "tensorflow>=2.4,<2.5",
                "onnx>=1.8,<1.9",
                "onnxruntime>=1.7,<1.8",
                "torch>=1.8,<1.9",
                "torchvision>=0.9,<0.10",
                "mpmath>=1.2,<1.3",
                "pillow>=8.1",
            )(manager)
            commit_hash = "3344959"
            install_eran = install.installer_builder(
                install.command(f"cd {manager.base_dir}"),
                git_download(
                    "https://github.com/eth-sri/eran.git",
                    commit_hash=commit_hash,
                ),
            )
            manager.require_program("eran", action_if_not_found=install_eran)

    with open(manager.base_dir / "bin/eran", "w+") as f:
        f.write(
            eran_runner.format(
                verifier_dir=verifier_base_dir, python_venv=verifier_base_dir / ".venv"
            )
        )
    (manager.base_dir / "bin/eran").chmod(0o700)
