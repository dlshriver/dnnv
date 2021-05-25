from __future__ import annotations

from .. import install


def configure_install(manager: install.InstallationManager):
    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    commit_hash = "7976635"
    install_reluplex = install.installer_builder(
        install.create_build_dir(
            manager.cache_dir / f"reluplex-{commit_hash}", enter_dir=True
        ),
        install.git_download(
            "https://github.com/dlshriver/ReluplexCav2017.git", commit_hash=commit_hash
        ),
        install.command("make"),
        install.command(
            f"cp check_properties/generic_prover/generic_prover.elf {manager.base_dir}/bin/reluplex"
        ),
    )
    manager.require_program("reluplex", action_if_not_found=install_reluplex)
