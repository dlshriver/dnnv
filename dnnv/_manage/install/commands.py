from pathlib import Path
from typing import Callable, Iterator, List, Optional

from . import base


def command(cmd: str, conditional: bool = True) -> Callable[..., Iterator[str]]:
    if not conditional:
        return iter(())

    def generator(
        manager: base.InstallationManager, *, cmd: str = cmd
    ) -> Iterator[str]:
        yield cmd

    return generator


def copy_install(build_dir: Path, install_dir: Path) -> Callable[..., Iterator[str]]:
    def generator(
        manager: base.InstallationManager,
        *,
        build_dir: Path = build_dir,
        install_dir: Path = install_dir,
    ) -> Iterator[str]:
        for dirname in ["bin", "lib", "include", "info", "share"]:
            yield f"[ ! -e {build_dir}/{dirname} ] || cp -r {build_dir}/{dirname} {install_dir}"

    return generator


def create_build_dir(
    build_dir: Path, enter_dir: bool = False
) -> Callable[..., Iterator[str]]:
    def generator(
        manager: base.InstallationManager,
        *,
        build_dir: Path = build_dir,
        enter_dir: bool = enter_dir,
    ) -> Iterator[str]:
        build_dir = build_dir.resolve()

        if not build_dir.exists():
            yield f"mkdir -p {build_dir}"
        if enter_dir:
            yield f"cd {build_dir}"

    return generator


def extract_tar(tarball: str) -> Callable[..., Iterator[str]]:
    def generator(
        manager: base.InstallationManager, *, tarball: str = tarball
    ) -> Iterator[str]:
        yield f"tar xf {tarball}"

    return generator


def git_download(
    url: str, commit_hash: Optional[str] = None
) -> Callable[..., Iterator[str]]:
    def generator(
        manager: base.InstallationManager,
        *,
        url: str = url,
        commit_hash: Optional[str] = commit_hash,
    ) -> Iterator[str]:
        dirname = Path(url).stem
        yield f"[ -e {dirname} ] || git clone {url}"
        yield f"cd {dirname}"
        if commit_hash is not None:
            yield f"git checkout {commit_hash}"

    return generator


def make_install() -> Callable[..., Iterator[str]]:
    def generator(manager: base.InstallationManager) -> Iterator[str]:
        yield f"make"
        yield f"make install"

    return generator


def wget_download(url: str, options: str = "-q") -> Callable[..., Iterator[str]]:
    def generator(
        manager: base.InstallationManager, *, url: str = url, options: str = options
    ) -> Iterator[str]:
        yield f"[ -e $(pwd)/{Path(url).name} ] || wget {options} {url}"

    return generator


__all__ = [
    "command",
    "copy_install",
    "create_build_dir",
    "extract_tar",
    "git_download",
    "make_install",
    "wget_download",
]
