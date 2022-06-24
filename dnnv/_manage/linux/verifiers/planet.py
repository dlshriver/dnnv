from __future__ import annotations

import subprocess as sp

from ...errors import InstallError, UninstallError
from ..environment import (
    Dependency,
    Environment,
    GNUInstaller,
    HeaderDependency,
    Installer,
    LibraryDependency,
    ProgramDependency,
)


class PlanetInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "a898a86"

        cache_dir = env.cache_dir / f"planet-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        library_paths = " ".join(f"-L{p}" for p in env.ld_library_paths)
        include_paths = "-I. " + " ".join(f"-I{p}" for p in env.include_paths)

        compiler_options = (
            f"-c -m64 -pipe -std=c++14 -O2 -fPIC -DUSE_GLPK -DNDEBUG {include_paths}"
        )
        library_options = f"{library_paths} -Wl,-Bstatic -lglpk -lgmp -Wl,-Bdynamic -lz"

        commands = [
            "set -ex",
            f"cd {cache_dir}",
            "if [ ! -e planet ]",
            "then git clone https://github.com/progirep/planet.git",
            "cd planet",
            f"git checkout {commit_hash}",
            "cd src",
            f"g++ {compiler_options} -o Options.o minisat2/Options.cc",
            f"g++ {compiler_options} -o Solver.o minisat2/Solver.cc",
            f"g++ {compiler_options} -o System.o minisat2/System.cc",
            f"g++ {compiler_options} -o main.o main.cpp",
            f"g++ {compiler_options} -o verifierContext.o verifierContext.cpp",
            f"g++ {compiler_options} -o supersetdatabase.o supersetdatabase.cpp",
            (
                "g++ -m64 -o planet"
                " Options.o Solver.o System.o"
                " main.o verifierContext.o supersetdatabase.o"
                f" {library_options}"
            ),
            "else cd planet/src",
            "fi",
            f"cp planet {installation_path}",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=env.vars())
        if proc.returncode != 0:
            raise InstallError("Installation of planet failed")


def install(env: Environment):
    m4_installer = GNUInstaller(
        "m4", "1.4.1", "https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz"
    )
    gmp_installer = GNUInstaller(
        "gmp", "6.1.2", "https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz"
    )
    zlib_installer = GNUInstaller(
        "zlib",
        "1.2.12",
        "https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz",
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
                ProgramDependency("curl", min_version="7.16.0"),
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
