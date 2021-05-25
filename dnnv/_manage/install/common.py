import subprocess as sp

from pathlib import Path
from typing import Callable, Optional

from . import installer_builder, InstallationManager
from .commands import *


def gurobi_installer(
    install_dir: Path,
    cache_dir: Path,
    version: str = "9.0.2",
    python_venv: Optional[Path] = None,
    install_python_package: bool = False,
) -> Callable[[InstallationManager], sp.CompletedProcess]:
    major_version, minor_version, *patch_version = version.split(".")
    nondot_version = "".join([major_version, minor_version] + patch_version)
    return installer_builder(
        create_build_dir(cache_dir / f"gurobi-{version}", enter_dir=True),
        wget_download(
            f"https://packages.gurobi.com/{major_version}.{minor_version}/gurobi{version}_linux64.tar.gz"
        ),
        extract_tar(f"gurobi{version}_linux64.tar.gz"),
        command(f"cd gurobi{nondot_version}/linux64"),
        command(
            f". {python_venv}/bin/activate",
            conditional=(install_python_package and python_venv is not None),
        ),
        command("python setup.py install", conditional=install_python_package),
        command(f"cp -r bin {install_dir}"),
        command(f"cp lib/*.a {install_dir}/lib/"),
        command(f"cp lib/*.so {install_dir}/lib/"),
        command(f"cp include/*.h {install_dir}/include/"),
    )


def lpsolve_installer(
    install_dir: Path, cache_dir: Path, version: str = "5.5.2.5"
) -> Callable[[InstallationManager], sp.CompletedProcess]:
    return installer_builder(
        create_build_dir(cache_dir / f"lpsolve-{version}", enter_dir=True),
        wget_download(
            f"https://downloads.sourceforge.net/project/lpsolve/lpsolve/{version}/lp_solve_{version}_dev_ux64.tar.gz"
        ),
        extract_tar(f"lp_solve_{version}_dev_ux64.tar.gz"),
        command(f"cp *.h {install_dir}/include"),
        command(f"mkdir -p {install_dir}/include/lpsolve"),
        command(f"cp *.h {install_dir}/include/lpsolve"),
        command(f"cp *.a {install_dir}/lib"),
        command(f"cp *.so {install_dir}/lib"),
    )


def openblas_installer(
    install_dir: Path, cache_dir: Path, version: str = "0.3.6"
) -> Callable[[InstallationManager], sp.CompletedProcess]:
    return installer_builder(
        create_build_dir(cache_dir / f"openblas-{version}", enter_dir=True),
        wget_download(f"https://github.com/xianyi/OpenBLAS/archive/v{version}.tar.gz"),
        extract_tar(f"v{version}.tar.gz"),
        command(f"cd OpenBLAS-{version}"),
        command("make"),
        command(f"make PREFIX={cache_dir}/OpenBLAS-{version} install"),
        copy_install(cache_dir / f"OpenBLAS-{version}", install_dir),
    )


__all__ = ["cmake_installer"]
