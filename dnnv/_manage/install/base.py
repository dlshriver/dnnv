from __future__ import annotations

import ast
import logging
import os
import subprocess as sp

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .commands import *
from .errors import InstallationError


def installer_builder(
    *command_generators: Callable[[InstallationManager], Iterator[str]]
) -> Callable[[InstallationManager], sp.CompletedProcess]:
    def installer(
        manager: InstallationManager,
    ) -> sp.CompletedProcess:
        logger = logging.getLogger("dnnv_manage.install")

        commands = []
        commands.append("set -ex")
        for command_generator in command_generators:
            commands.extend(list(command_generator(manager)))

        install_script = "; ".join(commands)

        install_proc = sp.run(install_script, shell=True, cwd=manager.base_dir)
        return install_proc

    return installer


def missing_required(msg: str):
    print(msg)
    exit(1)


class InstallationManager:
    def __init__(self):
        self.base_dir: Path = Path(".dnnv/").resolve()
        self.cache_dir: Path = Path(".dnnv/.cache/").resolve()
        self.active_venv: Optional[Path] = None

        self.base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        self.logger = logging.getLogger("dnnv_manage.install")

    @contextmanager
    def using_base_dir(self: InstallationManager, path: os.PathLike):
        prev_base_dir = self.base_dir
        try:
            path = Path(path).resolve()
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.base_dir = path
            yield
        finally:
            self.base_dir = prev_base_dir

    @contextmanager
    def using_python_venv(
        self: InstallationManager,
        path: Optional[Path] = None,
        python_version: Optional[str] = None,
    ):
        if path is None:
            path = self.base_dir / ".venv"
        if python_version is None:
            python_version = "python"
        try:
            path = path.resolve()
            if not path.exists():
                sp.run(
                    f"{python_version} -m venv {path}", shell=True, cwd=self.base_dir
                )
            self.active_venv = path
            yield
        finally:
            self.active_venv = None

    def require_header(
        self,
        header: str,
        path: Optional[List[Path]] = None,
        action_if_not_found: Optional[
            Callable[[InstallationManager], sp.CompletedProcess]
        ] = None,
    ):
        if path is None:
            path = [self.base_dir / "include"]
        include_path = " ".join([f"-I{p}" for p in path])
        proc = sp.run(
            f"gcc -Wp,-v -xc++ /dev/null -fsyntax-only {include_path}",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
        )
        output = proc.stdout.decode("utf8")
        try:
            start = output.index("#include <...> search starts here:") + 35
            end = output.index("End of search list.") - 1
            new_paths = (Path(p.strip()) for p in output[start:end].split("\n"))
            for p in new_paths:
                if (p / header).exists():
                    self.logger.debug(f"found {header}: {p/header}")
                    return self
        finally:
            pass
        if action_if_not_found is None:
            missing_required(f"Missing required header: {header}")
        proc = action_if_not_found(self)
        if proc.returncode != 0:
            raise InstallationError(
                f"Error when running `action_if_not_found` for require_header('{header}')"
            )
        return self

    def require_library(
        self,
        lib: str,
        path: Optional[List[Path]] = None,
        action_if_not_found: Optional[
            Callable[[InstallationManager], sp.CompletedProcess]
        ] = None,
    ):
        if path is None:
            path_var = "LD_LIBRARY_PATH"
            path_val = os.getenv(path_var, "")
            path = [Path(p) for p in path_val.split(os.path.pathsep)]
            path.insert(0, self.base_dir / "lib")
        for p in path:
            if (p / f"{lib}.so").exists():
                self.logger.debug(f"found {lib}: {p}/{lib}.so")
                return self
            if (p / f"{lib}.a").exists():
                self.logger.debug(f"found {lib}: {p}/{lib}.a")
                return self
        proc = sp.run(
            f"ldconfig -p | grep {lib}.so",
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        if proc.returncode == 0:
            self.logger.debug(f"found {lib}: ?")
            return self
        proc = sp.run(
            f"ldconfig -p | grep {lib}.a",
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        if proc.returncode == 0:
            self.logger.debug(f"found {lib}: ?")
            return self
        if action_if_not_found is None:
            missing_required(f"Missing required library: {lib}")
        proc = action_if_not_found(self)
        if proc.returncode != 0:
            raise InstallationError(
                f"Error when running `action_if_not_found` for require_library('{lib}')"
            )
        return self

    def require_program(
        self,
        prog: str,
        path: Optional[List[Path]] = None,
        action_if_not_found: Optional[
            Callable[[InstallationManager], sp.CompletedProcess]
        ] = None,
    ) -> InstallationManager:
        if path is None:
            path_var = "PATH"
            path_val = os.getenv(path_var, "")
            path = [Path(p) for p in path_val.split(os.path.pathsep)]
            path.insert(0, self.base_dir / "bin")
            if self.active_venv is not None:
                path.insert(0, self.active_venv / "bin")
        for p in path:
            if (p / prog).exists():
                self.logger.debug(f"found {prog}: {p/prog}")
                return self
        if action_if_not_found is None:
            missing_required(f"Missing required program: {prog}")
        proc = action_if_not_found(self)
        if proc.returncode != 0:
            raise InstallationError(
                f"Error when running `action_if_not_found` for require_program('{prog}')"
            )
        return self

    def require_python_package(
        self,
        pkg: str,
        version: str = "",
        action_if_not_found: Optional[
            Callable[[InstallationManager], sp.CompletedProcess]
        ] = None,
    ) -> InstallationManager:
        proc = sp.run(
            f". {self.active_venv}/bin/activate; pip list --format=json",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
        )
        packages = ast.literal_eval(proc.stdout.decode("utf8").split("\n")[0])
        for package in packages:
            if package["name"] == pkg:
                name = package["name"]
                installed_version = package["version"]
                self.logger.debug(f"found {pkg}: {name}=={installed_version}")
                if version == "":
                    return self
                # TODO: check version
                return self

        if action_if_not_found is None:
            missing_required(f"Missing required python package: {pkg}")
        proc = action_if_not_found(self)
        if proc.returncode != 0:
            raise InstallationError(
                f"Error when running `action_if_not_found` for require_python_package('{pkg}')"
            )
        return self

    def gnu_install(
        self: InstallationManager, name: str, version: str, url: str
    ) -> Callable[[InstallationManager], sp.CompletedProcess]:
        identifier = f"{name}-{version}"
        build_dir = self.cache_dir / identifier
        wget_filename = Path(url).name
        return installer_builder(
            create_build_dir(self.cache_dir / identifier, enter_dir=True),
            wget_download(url),
            extract_tar(wget_filename),
            command(f"cd {build_dir/identifier}"),
            command(
                f'CFLAGS="-I{self.base_dir}/include -L{self.base_dir}/lib" ./configure --prefix={build_dir}'
            ),
            make_install(),
            copy_install(build_dir, self.base_dir),
        )

    def pip_install(
        self: InstallationManager, *packages: str, extra_args: str = ""
    ) -> Callable[[InstallationManager], sp.CompletedProcess]:
        packages_str = " ".join([f'"{pkg}"' for pkg in packages])
        commands = []
        if self.active_venv is not None:
            commands.append(command(f". {self.active_venv}/bin/activate"))
        commands.append(command(f"pip install {extra_args} {packages_str}"))
        return installer_builder(*commands)


__all__ = ["installer_builder", "InstallationManager"]