"""
dnnv._manage - management tool for DNNV
"""
from __future__ import annotations

import logging
from typing import List

from .environment import Environment, ProgramDependency
from .verifiers import import_verifier_module, verifier_choices


def install(verifiers: List[str]) -> int:
    logger = logging.getLogger("dnnv_manage.install")
    for verifier in verifiers:
        logger.info("installing %s", verifier)
        environment = Environment()
        import_verifier_module(verifier).install(environment)
    return 0


def uninstall(verifiers: List[str]) -> int:
    logger = logging.getLogger("dnnv_manage.uninstall")
    for verifier in verifiers:
        logger.info("uninstalling %s", verifier)
        environment = Environment()
        import_verifier_module(verifier).uninstall(environment)
    return 0


def list_verifiers():
    environment = Environment()
    installed_verifiers = []
    for verifier in verifier_choices:
        if ProgramDependency(verifier).is_installed(environment):
            installed_verifiers.append(verifier)
    print("verifier")
    print("-" * max(8, max((len(vname) for vname in installed_verifiers), default=0)))
    print("\n".join(installed_verifiers))
    return 0


__all__ = [
    "install",
    "uninstall",
    "list_verifiers",
    "import_verifier_module",
    "verifier_choices",
]
