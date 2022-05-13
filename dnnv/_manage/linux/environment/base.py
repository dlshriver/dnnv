from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

from ...errors import InstallError
from .dependencies import Dependency


class Environment:
    def __init__(self):
        self.env_dir = (
            (
                Path(
                    os.getenv(
                        "VIRTUAL_ENV",
                        os.path.join(
                            os.getenv("XDG_DATA_HOME", "~/.local/share"),
                            "dnnv",
                        ),
                    )
                )
            )
            .expanduser()
            .resolve()
        )
        self.env_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = (
            (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "dnnv")
            .expanduser()
            .resolve()
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.paths: List[Path] = []
        self.ld_library_paths: List[Path] = []
        self.include_paths: List[Path] = []

    def add_search_paths(
        self,
        *,
        paths: Optional[Sequence[Path]] = None,
        ld_library_paths: Optional[Sequence[Path]] = None,
        include_paths: Optional[Sequence[Path]] = None,
    ):
        self.paths = list(paths or []) + self.paths
        self.ld_library_paths = list(ld_library_paths or []) + self.ld_library_paths
        self.include_paths = list(include_paths or []) + self.include_paths

    def vars(self):
        envvars = os.environ.copy()

        env_paths = ":".join(str(p) for p in self.paths)
        if "PATH" in envvars:
            orig_path = envvars.get("PATH")
            env_paths = f"{env_paths}:{orig_path}"
        envvars["PATH"] = env_paths

        env_ld_library_paths = ":".join(str(p) for p in self.ld_library_paths)
        if "LD_LIBRARY_PATH" in envvars:
            orig_ld_library_paths = envvars.get("LD_LIBRARY_PATH")
            env_ld_library_paths = f"{env_ld_library_paths}:{orig_ld_library_paths}"
        envvars["LD_LIBRARY_PATH"] = env_ld_library_paths

        env_include_paths = ":".join(str(p) for p in self.include_paths)
        if "C_INCLUDE_PATH" in envvars:
            orig_include_paths = envvars.get("C_INCLUDE_PATH")
            env_include_paths = f"{env_include_paths}:{orig_include_paths}"
        envvars["C_INCLUDE_PATH"] = env_include_paths

        env_include_paths = ":".join(str(p) for p in self.include_paths)
        if "CPLUS_INCLUDE_PATH" in envvars:
            orig_include_paths = envvars.get("CPLUS_INCLUDE_PATH")
            env_include_paths = f"{env_include_paths}:{orig_include_paths}"
        envvars["CPLUS_INCLUDE_PATH"] = env_include_paths

        envvars["GUROBI_HOME"] = envvars.get(
            "GUROBI_HOME", str(self.env_dir / "opt" / "gurobi912" / "linux64")
        )

        return envvars

    def ensure_dependency(self, dependency: Dependency) -> Environment:
        self.add_search_paths(**dependency.extra_search_paths)
        if not dependency.is_installed(self):
            self.ensure_dependencies(*dependency.dependencies)
            dependency.install(self)
        if not dependency.is_installed(self):
            raise InstallError(f"Failed to install dependency: {dependency.name}")
        return self

    def ensure_dependencies(self, *dependencies: Dependency) -> Environment:
        for dependency in dependencies:
            self.ensure_dependency(dependency)
        return self


__all__ = ["Environment"]
