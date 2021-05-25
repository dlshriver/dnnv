from __future__ import annotations

from .. import install
from ..install.common import lpsolve_installer, openblas_installer


def configure_install(manager: install.InstallationManager):
    manager.require_program("make")
    manager.require_program("gcc")
    manager.require_program("git")

    manager.require_header(
        "valgrind/callgrind.h",
        action_if_not_found=manager.gnu_install(
            "valgrind",
            "3.15.0",
            "https://sourceware.org/pub/valgrind/valgrind-3.15.0.tar.bz2",
        ),
    )

    install_glpk = manager.gnu_install(
        "glpk", "4.65", "https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz"
    )
    manager.require_header("glpk.h", action_if_not_found=install_glpk)
    manager.require_library("libglpk", action_if_not_found=install_glpk)

    manager.require_program(
        "m4",
        action_if_not_found=manager.gnu_install(
            "m4", "1.4.1", "https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz"
        ),
    )

    gnu_install_gmp = manager.gnu_install(
        "gmp", "6.1.2", "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    )
    manager.require_library("libgmp", action_if_not_found=gnu_install_gmp)
    manager.require_header("gmp.h", action_if_not_found=gnu_install_gmp)

    gnu_install_zlib = manager.gnu_install(
        "zlib", "1.2.11", "https://www.zlib.net/zlib-1.2.11.tar.xz"
    )
    manager.require_library("libz", action_if_not_found=gnu_install_zlib)
    manager.require_header("zlib.h", action_if_not_found=gnu_install_zlib)

    commit_hash = "a898a86"
    install_reluplex = install.installer_builder(
        install.create_build_dir(
            manager.cache_dir / f"planet-{commit_hash}", enter_dir=True
        ),
        install.git_download(
            "https://github.com/progirep/planet.git", commit_hash=commit_hash
        ),
        install.command("cd src"),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o Options.o minisat2/Options.cc"
        ),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o Solver.o minisat2/Solver.cc"
        ),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o System.o minisat2/System.cc"
        ),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o main.o main.cpp"
        ),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o verifierContext.o verifierContext.cpp"
        ),
        install.command(
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I{manager.base_dir}/include -L{manager.base_dir}/lib/ -o supersetdatabase.o supersetdatabase.cpp"
        ),
        install.command(
            f"g++ -m64 -Wl,-O1 -L{manager.base_dir}/lib/ -o planet Options.o Solver.o System.o main.o verifierContext.o supersetdatabase.o -Bstatic -lglpk -lgmp -lz"
        ),
        install.command(f"cp planet {manager.base_dir}/bin/"),
    )
    manager.require_program("planet", action_if_not_found=install_reluplex)
