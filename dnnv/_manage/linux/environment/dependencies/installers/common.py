from __future__ import annotations

import subprocess as sp
import typing
from pathlib import Path
from typing import Optional

from .....errors import InstallError
from ..base import Dependency
from .base import Installer

if typing.TYPE_CHECKING:
    from ...base import Environment


class GNUInstaller(Installer):
    def __init__(self, name: str, version: str, url: Optional[str] = None):
        self.name = name
        self.version = version
        self.url = url or f"https://ftp.gnu.org/gnu/{name}/{name}-{version}.tar.gz"

    def run(self, env: Environment, dependency: Dependency):
        identifier = f"{self.name}-{self.version}"
        cache_dir = env.cache_dir / identifier
        cache_dir.mkdir(exist_ok=True, parents=True)

        env.paths.append(cache_dir / "bin")
        env.include_paths.append(cache_dir / "include")
        env.ld_library_paths.append(cache_dir / "lib")
        if dependency.is_installed(env):
            return

        library_paths = " ".join(f"-L{p}" for p in env.ld_library_paths)
        include_paths = " ".join(f"-I{p}" for p in env.include_paths)
        cflags = f'CFLAGS="{include_paths} {library_paths}"'

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"curl -o {identifier}.tar.gz -L {self.url}",
            f"tar xf {identifier}.tar.gz",
            f"cd {identifier}",
            f"{cflags} ./configure --prefix={cache_dir}",
            "make",
            "make install",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of {identifier} failed")


class GurobiInstaller(Installer):
    def __init__(self, version: str):
        self.version = version

    def run(self, env: Environment, dependency: Dependency):
        envvars = env.vars()
        if "GUROBI_HOME" in envvars:
            gurobi_home = Path(envvars["GUROBI_HOME"])
            env.paths.append(gurobi_home / "bin")
            env.include_paths.append(gurobi_home / "include")
            env.ld_library_paths.append(gurobi_home / "lib")
            if dependency.is_installed(env):
                return

        major_version, minor_version, *patch_version = self.version.split(".")
        major_minor_version = f"{major_version}.{minor_version}"
        nondot_version = "".join([major_version, minor_version] + patch_version)
        gurobi_id = f"gurobi{nondot_version}"

        identifier = f"gurobi-{self.version}"
        cache_dir = env.cache_dir / identifier
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "opt"
        installation_path.mkdir(exist_ok=True, parents=True)
        gurobi_linux_path = installation_path / gurobi_id / "linux64"

        env.paths.append(gurobi_linux_path / "bin")
        env.include_paths.append(gurobi_linux_path / "include")
        env.ld_library_paths.append(gurobi_linux_path / "lib")
        if dependency.is_installed(env):
            return

        gurobi_url = (
            f"https://packages.gurobi.com/{major_minor_version}/"
            f"gurobi{self.version}_linux64.tar.gz"
        )
        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"curl -o {identifier}.tar.gz -L {gurobi_url}",
            f"tar xf {identifier}.tar.gz",
            f"cp -r {gurobi_id} {installation_path}/{gurobi_id}",
            f"ln -s {gurobi_linux_path}/bin/grbgetkey {env.env_dir}/bin/grbgetkey",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of {identifier} failed")


class LpsolveInstaller(Installer):
    def __init__(self, version: str):
        self.version = version

    def run(self, env: Environment, dependency: Dependency):
        name = f"lpsolve-{self.version}"
        cache_dir = env.cache_dir / name
        cache_dir.mkdir(exist_ok=True, parents=True)

        env.include_paths.append(cache_dir)
        env.ld_library_paths.append(cache_dir)
        if dependency.is_installed(env):
            return

        lpsolve_url = (
            "https://downloads.sourceforge.net/project/lpsolve/lpsolve/"
            f"{self.version}/lp_solve_{self.version}_dev_ux64.tar.gz"
        )
        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"curl -o {name}.tar.gz -L {lpsolve_url}",
            f"tar xf {name}.tar.gz",
            "mkdir -p lpsolve",
            "cp *.h lpsolve/",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")


class OpenBLASInstaller(Installer):
    def __init__(self, version: str):
        self.version = version

    def run(self, env: Environment, dependency: Dependency):
        name = f"OpenBLAS-{self.version}"
        cache_dir = env.cache_dir / name
        cache_dir.mkdir(exist_ok=True, parents=True)

        env.include_paths.append(cache_dir / "include")
        env.ld_library_paths.append(cache_dir / "lib")
        if dependency.is_installed(env):
            return

        openblas_url = (
            f"https://github.com/xianyi/OpenBLAS/archive/v{self.version}.tar.gz"
        )
        commands = [
            "set -ex",
            f"cd {cache_dir}",
            f"curl -o {name}.tar.gz -L {openblas_url}",
            f"tar xf {name}.tar.gz",
            f"cd {name}",
            "make",
            f"make PREFIX={cache_dir} install",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")


__all__ = [
    "GNUInstaller",
    "GurobiInstaller",
    "LpsolveInstaller",
    "OpenBLASInstaller",
]
