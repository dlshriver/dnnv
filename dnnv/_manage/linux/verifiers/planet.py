from __future__ import annotations

import subprocess as sp

from ..environment import (
    Environment,
    Dependency,
    HeaderDependency,
    LibraryDependency,
    ProgramDependency,
    Installer,
    GNUInstaller,
)
from ...errors import InstallError, UninstallError


class PlanetInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "a898a86"

        cache_dir = env.cache_dir / f"planet-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        library_paths = " ".join(f"-L{p}" for p in env.ld_library_paths)
        include_paths = " ".join(f"-I{p}" for p in env.include_paths)

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            "rm -rf planet",
            "git clone https://github.com/progirep/planet.git",
            "cd planet",
            f"git checkout {commit_hash}",
            "cd src",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o Options.o minisat2/Options.cc",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o Solver.o minisat2/Solver.cc",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o System.o minisat2/System.cc",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o main.o main.cpp",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o verifierContext.o verifierContext.cpp",
            f"g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. {include_paths} -o supersetdatabase.o supersetdatabase.cpp",
            f"g++ -m64 -Wl,-O1 -static {library_paths} -o planet Options.o Solver.o System.o main.o verifierContext.o supersetdatabase.o -Bstatic -lglpk -lgmp -lz",
            f"cp planet {installation_path}",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError(f"Installation of planet failed")


def install(env: Environment):
    m4_installer = GNUInstaller(
        "m4", "1.4.1", "https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz"
    )
    gmp_installer = GNUInstaller(
        "gmp", "6.1.2", "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    )
    zlib_installer = GNUInstaller(
        "zlib", "1.2.11", "https://www.zlib.net/zlib-1.2.11.tar.xz"
    )
    glpk_installer = GNUInstaller("glpk", "4.65")
    valgrind_installer = GNUInstaller(
        "valgrind",
        "3.17.0",
        url="https://sourceware.org/pub/valgrind/valgrind-3.17.0.tar.bz2",
    )
    env.ensure_dependencies(
        ProgramDependency(
            "planet",
            installer=PlanetInstaller(),
            dependencies=(
                ProgramDependency("make"),
                ProgramDependency("gcc"),
                ProgramDependency("git"),
                HeaderDependency(
                    "gmp.h",
                    installer=gmp_installer,
                    dependencies=(ProgramDependency("m4", installer=m4_installer),),
                ),
                LibraryDependency(
                    "libgmp",
                    installer=gmp_installer,
                    dependencies=(ProgramDependency("m4", installer=m4_installer),),
                ),
                HeaderDependency("zlib.h", installer=zlib_installer),
                LibraryDependency("libz", installer=zlib_installer),
                HeaderDependency("glpk.h", installer=glpk_installer),
                LibraryDependency("libglpk", installer=glpk_installer),
                HeaderDependency("valgrind/callgrind.h", installer=valgrind_installer),
            ),
        )
    )


def uninstall(env: Environment):
    exe_path = env.env_dir / "bin" / "planet"
    commands = [
        f"rm -f {exe_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError("Uninstallation of planet failed")


__all__ = ["install", "uninstall"]
