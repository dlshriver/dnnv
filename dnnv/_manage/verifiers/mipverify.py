from __future__ import annotations

from .. import install
from ..install.common import gurobi_installer


def configure_install(manager: install.InstallationManager):
    manager.require_program("make")
    manager.require_program("gcc")

    julia_version = "1.0.5"
    julia_major_minor = ".".join(julia_version.split(".")[:2])
    install_julia = install.installer_builder(
        install.create_build_dir(
            manager.cache_dir / f"julia-{julia_version}", enter_dir=True
        ),
        install.wget_download(
            f"https://julialang-s3.julialang.org/bin/linux/x64/{julia_major_minor}/julia-{julia_version}-linux-x86_64.tar.gz"
        ),
        install.extract_tar(f"julia-{julia_version}-linux-x86_64.tar.gz"),
        install.copy_install(
            build_dir=manager.cache_dir
            / f"julia-{julia_version}"
            / f"julia-{julia_version}",
            install_dir=manager.base_dir,
        ),
    )
    manager.require_program(
        "julia",
        action_if_not_found=install_julia,
    )

    install_gurobi = gurobi_installer(
        install_dir=manager.base_dir,
        cache_dir=manager.cache_dir,
        version="9.0.2",
        python_venv=manager.active_venv,
        install_python_package=True,
    )
    manager.require_program("grbgetkey", action_if_not_found=install_gurobi)
    manager.require_library("libgurobi90", action_if_not_found=install_gurobi)
    manager.require_header("gurobi_c.h", action_if_not_found=install_gurobi)

    gnu_install_zlib = manager.gnu_install(
        "zlib", "1.2.11", "https://www.zlib.net/zlib-1.2.11.tar.xz"
    )
    manager.require_library("libz", action_if_not_found=gnu_install_zlib)
    manager.require_header("zlib.h", action_if_not_found=gnu_install_zlib)

    install.installer_builder(
        install.command("julia -e 'using Pkg; Pkg.add(\"Gurobi\")'"),
        install.command("julia -e 'using Pkg; Pkg.add(\"MAT\")'"),
        install.command(
            'julia -e \'using Pkg; Pkg.add(PackageSpec(url="https://github.com/vtjeng/MIPVerify.jl", rev="2d58aec"))\''
        ),
        install.command("julia -e 'using Pkg; Pkg.update()'"),
    )(manager)
