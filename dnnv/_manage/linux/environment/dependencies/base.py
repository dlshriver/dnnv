from __future__ import annotations

import shlex
import subprocess as sp
import typing
from pathlib import Path
from typing import Dict, Optional, Sequence

from ....errors import InstallError

if typing.TYPE_CHECKING:
    from ..base import Environment
    from .installers import Installer


class Dependency:
    def __init__(
        self,
        name,
        *,
        installer: Optional[Installer] = None,
        dependencies: Optional[Sequence[Dependency]] = None,
        extra_search_paths: Optional[Dict[str, Sequence[Path]]] = None,
        allow_from_system: bool = True,
    ):
        self.name = name
        self.installer = installer
        self.dependencies = dependencies or []

        self.extra_search_paths = extra_search_paths or {}
        self.allow_from_system = allow_from_system

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
            shlex.split(f"gcc -Wp,-v -xc++ /dev/null -fsyntax-only {include_paths}"),
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
        if not self.allow_from_system:
            return None
        proc = sp.run(
            shlex.split("ldconfig -p"),
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf8",
            env=envvars,
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.split("\n"):
            if line.endswith(f"/{self.name}.so") or line.endswith(f"/{self.name}.a"):
                return Path(line.split("=>")[-1].strip())
        return None


class ProgramDependency(Dependency):
    def __init__(
        self,
        name,
        *,
        installer: Optional[Installer] = None,
        dependencies: Optional[Sequence[Dependency]] = None,
        extra_search_paths: Optional[Dict[str, Sequence[Path]]] = None,
        allow_from_system: bool = True,
        min_version: Optional[str] = None,
        version_arg: str = "--version",
    ):
        super().__init__(
            name,
            installer=installer,
            dependencies=dependencies,
            extra_search_paths=extra_search_paths,
            allow_from_system=allow_from_system,
        )
        self.min_version = min_version
        self.version_arg = version_arg

    def is_installed(self, env: Environment) -> bool:
        proc = sp.run(
            shlex.split(f"which {self.name}"),
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            env=env.vars(),
        )
        if proc.returncode == 0:
            if self.min_version:
                import re

                version_proc = sp.run(
                    shlex.split(f"{self.name} {self.version_arg}"),
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    encoding="utf8",
                    env=env.vars(),
                )
                min_version = tuple(int(v) for v in self.min_version.split("."))
                version_pattern = re.compile(
                    ".".join(r"(\d+)" for _ in range(len(min_version)))
                )
                match = re.search(version_pattern, version_proc.stdout)
                if match is None:
                    return False
                version = tuple(int(v) for v in match.groups())
                return version >= min_version
            return True
        return False


__all__ = [
    "Dependency",
    "HeaderDependency",
    "LibraryDependency",
    "ProgramDependency",
]
