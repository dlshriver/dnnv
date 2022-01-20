from __future__ import annotations

import subprocess as sp

from ..environment import (
    Environment,
    Dependency,
    HeaderDependency,
    LibraryDependency,
    ProgramDependency,
    Installer,
    LpsolveInstaller,
    OpenBLASInstaller,
)
from ...errors import InstallError, UninstallError


class NeurifyInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "5987db4f3015ec4f53619171ae4b4480eadaa989"

        cache_dir = env.cache_dir / f"neurify-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        library_paths = " ".join(f"-L{p}" for p in env.ld_library_paths)
        include_paths = " ".join(f"-I{p}" for p in env.include_paths)

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            "rm -rf Neurify",
            "git clone https://github.com/dlshriver/Neurify.git",
            "cd Neurify",
            f"git checkout {commit_hash}",
            "cd generic",
            f'make LDFLAGS="-static {library_paths} -lopenblas -lpthread -lm" INCLUDE_FLAGS="{include_paths}"',
            f"cp src/neurify {installation_path}",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of neurify failed")


def install(env: Environment):
    lpsolve_installer = LpsolveInstaller("5.5.2.5")
    openblas_installer = OpenBLASInstaller("0.3.19")
    env.ensure_dependencies(
        ProgramDependency(
            "neurify",
            installer=NeurifyInstaller(),
            dependencies=(
                ProgramDependency("make"),
                ProgramDependency("gcc"),
                ProgramDependency("git"),
                ProgramDependency("curl", min_version="7.16.0"),
                HeaderDependency("lpsolve/lp_lib.h", installer=lpsolve_installer),
                LibraryDependency("liblpsolve55", installer=lpsolve_installer),
                HeaderDependency("cblas.h", installer=openblas_installer),
                LibraryDependency("libopenblas", installer=openblas_installer),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "neurify"
    commands = [
        f"rm -f {exe_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of neurify failed")


__all__ = ["install", "uninstall"]
