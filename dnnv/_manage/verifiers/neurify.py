from __future__ import annotations

from .. import install
from ..install.common import lpsolve_installer, openblas_installer


def configure_install(manager: install.InstallationManager):
    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    manager.require_library(
        "libopenblas",
        action_if_not_found=openblas_installer(
            manager.base_dir, manager.cache_dir, version="0.3.6"
        ),
    )

    install_lpsolve = lpsolve_installer(
        manager.base_dir, manager.cache_dir, version="5.5.2.5"
    )
    manager.require_library("liblpsolve55", action_if_not_found=install_lpsolve)
    manager.require_header("lp_lib.h", action_if_not_found=install_lpsolve)
    manager.require_header("lpsolve/lp_lib.h", action_if_not_found=install_lpsolve)

    commit_hash = "663bdd9"
    install_neurify = install.installer_builder(
        install.create_build_dir(
            manager.cache_dir / f"neurify-{commit_hash}", enter_dir=True
        ),
        install.git_download(
            "https://github.com/dlshriver/Neurify.git", commit_hash=commit_hash
        ),
        install.command("cd generic"),
        install.command(f"make OPENBLAS_HOME={manager.base_dir}"),
        install.command(f"cp src/neurify {manager.base_dir}/bin"),
    )
    manager.require_program("neurify", action_if_not_found=install_neurify)
