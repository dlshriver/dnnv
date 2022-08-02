from __future__ import annotations

import subprocess as sp

from ...errors import InstallError, UninstallError
from ..environment import (
    Dependency,
    Environment,
    GNUInstaller,
    GurobiInstaller,
    HeaderDependency,
    Installer,
    LibraryDependency,
    ProgramDependency,
)

MIPVERIFY_RUNNER = """#!/bin/bash
export GUROBI_HOME={gurobi_home}
cd {venv_path}
./julia --project=. -g 0 --track-allocation=none --code-coverage=none -O0 $@
"""


class MIPVerifyInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "mipverify"
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        libjulia_path = LibraryDependency("libjulia").get_path(env)
        assert libjulia_path is not None
        julia_dir = libjulia_path.parent.parent

        julia_cmd = "./julia --project=. -e"

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            "rm -rf mipverify",
            "mkdir mipverify",
            "cd mipverify",
            f"cp -r {julia_dir} .",
            f"ln -s {julia_dir}/bin/julia julia",
            (
                f"{julia_cmd} '"
                "using Pkg;"
                'Pkg.add("Gurobi");'
                'Pkg.add("MathOptInterface");'
                'Pkg.add("JuMP");'
                'Pkg.add("MAT");'
                'Pkg.add("GLPK");'
                'Pkg.add("HiGHS");'
                'Pkg.add(name="MIPVerify", version="0.3");'
                "Pkg.build();"
                "Pkg.precompile();"
                "'"
            ),
            f"{julia_cmd} 'using Pkg; Pkg.update(); Pkg.precompile()'",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError("Installation of MIPVerify failed")

        with open(installation_path / "mipverify", "w+") as f:
            f.write(
                MIPVERIFY_RUNNER.format(
                    venv_path=verifier_venv_path,
                    gurobi_home=envvars.get("GUROBI_HOME", "."),
                )
            )
        (installation_path / "mipverify").chmod(0o700)


class JuliaInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        version = "1.7.2"
        major_minor = ".".join(version.split(".")[:2])

        cache_dir = env.cache_dir / f"julia-{version}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        env.paths.append(cache_dir / f"julia-{version}" / "bin")
        env.include_paths.append(cache_dir / f"julia-{version}" / "include")
        env.ld_library_paths.append(cache_dir / f"julia-{version}" / "lib")
        if dependency.is_installed(env):
            return

        julia_url = (
            "https://julialang-s3.julialang.org/bin/linux/x64/"
            f"{major_minor}/julia-{version}-linux-x86_64.tar.gz"
        )
        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"curl -o julia-{version}.tar.gz -L {julia_url}",
            f"tar xf julia-{version}.tar.gz",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError("Installation of julia failed")


def install(env: Environment):
    zlib_installer = GNUInstaller(
        "zlib",
        "1.2.12",
        "https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz",
    )
    gurobi_installer = GurobiInstaller("9.1.2")
    env.ensure_dependencies(
        ProgramDependency(
            "mipverify",
            installer=MIPVerifyInstaller(),
            dependencies=(
                ProgramDependency("julia", installer=JuliaInstaller()),
                LibraryDependency("libjulia", installer=JuliaInstaller()),
                ProgramDependency("git"),
                HeaderDependency("zlib.h", installer=zlib_installer),
                LibraryDependency("libz", installer=zlib_installer),
                HeaderDependency("gurobi_c.h", installer=gurobi_installer),
                LibraryDependency("libgurobi91", installer=gurobi_installer),
                ProgramDependency("grbgetkey", installer=gurobi_installer),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "mipverify"
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "mipverify"
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of planet failed")


__all__ = ["install", "uninstall"]
