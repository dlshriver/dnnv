from __future__ import annotations

import subprocess as sp
import typing

from pathlib import Path
from typing import Dict, Optional, Sequence

from ....errors import InstallError

if typing.TYPE_CHECKING:
    from .installers import Installer
    from ..base import Environment


class Dependency:
    def __init__(
        self,
        name,
        *,
        installer: Optional[Installer] = None,
        dependencies: Optional[Sequence[Dependency]] = None,
        extra_search_paths: Optional[Dict[str, Sequence[Path]]] = None,
    ):
        self.name = name
        self.installer = installer
        self.dependencies = dependencies or []

        self.extra_search_paths = extra_search_paths or {}

    def get_path(self, env: Environment) -> Optional[Path]:
        raise NotImplementedError()

    def install(self, env: Environment) -> None:
        if self.installer is None:
            raise InstallError(f"Missing required dependency: {self.name}")
        self.installer.run(env, self)

    def is_installed(self, env: Environment) -> bool:
        if self.get_path(env) is None:
            return False
        return True


class HeaderDependency(Dependency):
    def get_path(self, env: Environment) -> Optional[Path]:
        header = self.name
        include_paths = " ".join([f"-I{p}" for p in env.include_paths])
        proc = sp.run(
            f"gcc -Wp,-v -xc++ /dev/null -fsyntax-only {include_paths}",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
        )
        output = proc.stdout.decode("utf8")
        start = output.index("#include <...> search starts here:") + 35
        end = output.index("End of search list.") - 1
        new_paths = (Path(p.strip()) for p in output[start:end].split("\n"))
        for p in new_paths:
            if (p / header).exists():
                return p / header
        return None


class LibraryDependency(Dependency):
    def get_path(self, env: Environment) -> Optional[Path]:
        envvars = env.vars()
        for path in env.ld_library_paths:
            if (path / f"{self.name}.so").exists():
                return path / f"{self.name}.so"
            if (path / f"{self.name}.a").exists():
                return path / f"{self.name}.a"
        proc = sp.run(
            f"ldconfig -p | grep /{self.name}.so$",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf8",
            env=envvars,
        )
        if proc.returncode == 0:
            return Path(proc.stdout.split("=>")[-1].strip())
        proc = sp.run(
            f"ldconfig -p | grep /{self.name}.a$",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf8",
            env=envvars,
        )
        if proc.returncode == 0:
            return Path(proc.stdout.split("=>")[-1].strip())
        return None


class ProgramDependency(Dependency):
    def is_installed(self, env: Environment) -> bool:
        proc = sp.run(
            f"command -v {self.name}",
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            env=env.vars(),
        )
        if proc.returncode == 0:
            return True
        return False


__all__ = [
    "Dependency",
    "HeaderDependency",
    "LibraryDependency",
    "ProgramDependency",
]
