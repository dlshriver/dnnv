from __future__ import annotations

import subprocess as sp

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

mipverify_runner = """#!/bin/bash
export GUROBI_HOME={gurobi_home}
cd {venv_path}
./julia --project=. $@
"""


class MIPVerifyInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "36fd890"

        cache_dir = env.cache_dir / f"mipverify-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / "mipverify"
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        julia_dir = LibraryDependency("libjulia").get_path(env).parent.parent

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            "rm -rf mipverify",
            "mkdir mipverify",
            "cd mipverify",
            f"cp -r {julia_dir} .",
            f"ln -s {julia_dir}/bin/julia julia",
            './julia --project=. -e \'using Pkg; Pkg.add("Gurobi"); Pkg.build("Gurobi")\'',
            './julia --project=. -e \'using Pkg; Pkg.add("MathOptInterface"); Pkg.build("MathOptInterface")\'',
            './julia --project=. -e \'using Pkg; Pkg.add("JuMP"); Pkg.build("JuMP")\'',
            './julia --project=. -e \'using Pkg; Pkg.add("MAT"); Pkg.build("MAT")\'',
            f'./julia --project=. -e \'using Pkg; Pkg.add(PackageSpec(url="https://github.com/vtjeng/MIPVerify.jl", rev="{commit_hash}"))\'',
            "./julia --project=. -e 'using Pkg; Pkg.update(); Pkg.precompile()'",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of MIPVerify failed")

        with open(installation_path / "mipverify", "w+") as f:
            f.write(
                mipverify_runner.format(
                    venv_path=verifier_venv_path,
                    gurobi_home=envvars.get("GUROBI_HOME", "."),
                )
            )
        (installation_path / "mipverify").chmod(0o700)


class JuliaInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        version = "1.6.1"
        major_minor = ".".join(version.split(".")[:2])

        cache_dir = env.cache_dir / f"julia-{version}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        env.paths.append(cache_dir / f"julia-{version}" / "bin")
        env.include_paths.append(cache_dir / f"julia-{version}" / "include")
        env.ld_library_paths.append(cache_dir / f"julia-{version}" / "lib")
        if dependency.is_installed(env):
            return

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"wget -O julia-{version}.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/{major_minor}/julia-{version}-linux-x86_64.tar.gz",
            f"tar xf julia-{version}.tar.gz",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of julia failed")


def install(env: Environment):
    zlib_installer = GNUInstaller(
        "zlib", "1.2.11", "https://www.zlib.net/zlib-1.2.11.tar.xz"
    )
    gurobi_installer = GurobiInstaller("9.1.2")
    env.ensure_dependencies(
        ProgramDependency(
            "mipverify",
            installer=MIPVerifyInstaller(),
            dependencies=(
                ProgramDependency("julia", installer=JuliaInstaller()),
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
