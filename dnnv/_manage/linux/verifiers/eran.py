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
    GNUInstaller,
    GurobiInstaller,
)
from ...errors import InstallError, UninstallError

wrapped_python_template = """#!/bin/bash

export GUROBI_HOME={gurobi_home}
export PATH={gurobi_home}/bin:$PATH
export LD_LIBRARY_PATH={gurobi_home}/lib:{elina_home}/lib:$LD_LIBRARY_PATH
export PYTHONPATH={elina_home}/python_interface:$PYTHONPATH
{python_venv}/bin/python $@
"""

runner_template = """#!{python_venv}/bin/wrappedpython
import argparse
import numpy as np

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


def return_result(result, cex=None, output_file=None):
    print(result)
    if cex is not None and output_file is not None:
        np.save(output_file, cex)


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
        return return_result("safe")
    elif output_upper_bound <= 0:
        return return_result("unsafe", (spec_lb + spec_ub) / 2, args.output)
    return return_result("unknown")


if __name__ == "__main__":
    main()
"""


class ELINAInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "7e0e6fef43c9676c869199782f4beadd542449f6"
        name = "elina"

        cache_dir = env.cache_dir / f"{name}-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "opt"
        installation_path.parent.mkdir(exist_ok=True, parents=True)

        env.ld_library_paths.append(installation_path / "ELINA" / "lib")
        if dependency.is_installed(env):
            return

        library_paths = " ".join(f"-L{p}" for p in env.ld_library_paths)
        include_paths = " ".join(f"-I{p}" for p in env.include_paths)

        mpfr_path = LibraryDependency("libmpfr").get_path(env).parent.parent
        gmp_path = LibraryDependency("libgmp").get_path(env).parent.parent
        # cdd_path = LibraryDependency("libcdd").get_path(env).parent.parent
        cdd_prefix = HeaderDependency("cddlib/cdd.h").get_path(env).parent

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {cache_dir}",
            "rm -rf ELINA",
            "git clone https://github.com/eth-sri/ELINA.git",
            "cd ELINA",
            f"git checkout {commit_hash}",
            # "git revert -n b347e3c",  # undoing commits that force reliance on local files
            # "git revert -n 16d2a6d --strategy=recursive -Xours",  # undoing commits that force reliance on local files
            # "git revert -n bda9f58",  # undoing commits that force reliance on local files
            f'CFLAGS="{include_paths} {library_paths}" ./configure -prefix {cache_dir}/ELINA -gmp-prefix {gmp_path} -mpfr-prefix {mpfr_path} -cdd-prefix {cdd_prefix} -use-deeppoly -use-gurobi -use-fconv',
            # f"sed -i 's#CDD_PREFIX = .*$#CDD_PREFIX = {cdd_path}/include -L{cdd_path}/lib#' Makefile.config",
            "make",
            "make install",
            f"cd {installation_path}",
            "rm -rf ELINA",
            # "mkdir ELINA",
            # "cd ELINA",
            # f"cp -r {cache_dir}/ELINA/lib .",
            # f"cp -r {cache_dir}/ELINA/python_interface .",
            # f"cp -r {cdd_path}/lib .",
            f"cp -r {cache_dir}/ELINA .",
            f"cp -r {cdd_prefix.parent.parent}/lib ELINA/",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")


class ERANInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "d60cf5767da31e7834b202fbcbb840e9c7d3ef5e"
        name = "eran"

        cache_dir = env.cache_dir / f"{name}-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        gurobi_path = LibraryDependency("libgurobi91").get_path(env).parent.parent
        elina_path = LibraryDependency("libzonoml").get_path(env).parent.parent

        python_major_version, python_minor_version = sys.version_info[:2]

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            f"rm -rf {name}",
            f"python -m venv {name}",
            f". {name}/bin/activate",
            "pip install --upgrade pip",
            'pip install "numpy>=1.19,<1.20" "tensorflow>=2.4,<2.5" "onnx>=1.8,<1.9" "onnxruntime>=1.7,<1.8" "torch>=1.8,<1.9" "torchvision>=0.9,<0.10" "mpmath>=1.2,<1.3" "pillow>=8.1"',
            f"cd {gurobi_path}",
            "python setup.py install",
            f"cd {cache_dir}",
            "rm -rf eran",
            "git clone https://github.com/eth-sri/eran.git",
            "cd eran",
            f"git checkout {commit_hash}",
            f"cp -r tf_verify/* {verifier_venv_path}/lib/python{python_major_version}.{python_minor_version}/site-packages/",
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
                    elina_home=elina_path,
                )
            )
        (verifier_venv_path / "bin" / "wrappedpython").chmod(0o700)
        with open(installation_path / name, "w+") as f:
            f.write(runner_template.format(python_venv=verifier_venv_path))
        (installation_path / name).chmod(0o700)


def install(env: Environment):
    m4_installer = GNUInstaller(
        "m4", "1.4.1", "https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz"
    )
    gmp_installer = GNUInstaller(
        "gmp", "6.1.2", "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    )
    mpfr_installer = GNUInstaller(
        "mpfr", "4.1.0", "https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz"
    )
    gurobi_installer = GurobiInstaller("9.1.2")
    cddlib_installer = GNUInstaller(
        "cddlib",
        "0.94m",
        "https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz",
    )
    elina_installer = ELINAInstaller()
    env.ensure_dependencies(
        ProgramDependency(
            "eran",
            installer=ERANInstaller(),
            dependencies=(
                ProgramDependency("git"),
                HeaderDependency("gurobi_c.h", installer=gurobi_installer),
                LibraryDependency("libgurobi91", installer=gurobi_installer),
                ProgramDependency("grbgetkey", installer=gurobi_installer),
                LibraryDependency(
                    "libzonoml",
                    installer=elina_installer,
                    dependencies=(
                        LibraryDependency(
                            "libgmp",
                            installer=gmp_installer,
                            dependencies=(
                                ProgramDependency("m4", installer=m4_installer),
                            ),
                        ),
                        HeaderDependency(
                            "mpfr.h",
                            installer=mpfr_installer,
                            dependencies=(
                                HeaderDependency("gmp.h", installer=gmp_installer),
                            ),
                        ),
                        LibraryDependency("libmpfr", installer=mpfr_installer),
                        HeaderDependency(
                            "cddlib/cdd.h",
                            installer=cddlib_installer,
                            dependencies=(
                                HeaderDependency(
                                    "gmp.h",
                                    installer=gmp_installer,
                                    dependencies=(
                                        ProgramDependency("m4", installer=m4_installer),
                                    ),
                                ),
                                LibraryDependency(
                                    "libgmp",
                                    installer=gmp_installer,
                                    dependencies=(
                                        ProgramDependency("m4", installer=m4_installer),
                                    ),
                                ),
                            ),
                        ),
                        LibraryDependency(
                            "libcdd",
                            installer=cddlib_installer,
                            dependencies=(
                                HeaderDependency(
                                    "gmp.h",
                                    installer=gmp_installer,
                                    dependencies=(
                                        ProgramDependency("m4", installer=m4_installer),
                                    ),
                                ),
                                LibraryDependency(
                                    "libgmp",
                                    installer=gmp_installer,
                                    dependencies=(
                                        ProgramDependency("m4", installer=m4_installer),
                                    ),
                                ),
                            ),
                        ),
                        HeaderDependency("gurobi_c.h", installer=gurobi_installer),
                        LibraryDependency("libgurobi91", installer=gurobi_installer),
                        ProgramDependency("grbgetkey", installer=gurobi_installer),
                    ),
                ),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "eran"
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "eran"
    elina_path = env.env_dir / "opt" / "ELINA"
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
        f"rm -rf {elina_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of eran failed")


__all__ = ["install", "uninstall"]
