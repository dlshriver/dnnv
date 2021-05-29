from __future__ import annotations

import subprocess as sp

from ..environment import (
    Environment,
    Dependency,
    ProgramDependency,
    Installer,
)
from ...errors import InstallError, UninstallError


class ReluplexInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "7976635"

        cache_dir = env.cache_dir / f"reluplex-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            "rm -rf ReluplexCav2017",
            "git clone https://github.com/dlshriver/ReluplexCav2017.git",
            "cd ReluplexCav2017",
            f"git checkout {commit_hash}",
            f"make",
            f"cp check_properties/generic_prover/generic_prover.elf {installation_path}/reluplex",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of reluplex failed")


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "reluplex",
            installer=ReluplexInstaller(),
            dependencies=(
                ProgramDependency("make"),
                ProgramDependency("gcc"),
                ProgramDependency("git"),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "reluplex"
    commands = [
        f"rm -f {exe_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of reluplex failed")


__all__ = ["install", "uninstall"]
