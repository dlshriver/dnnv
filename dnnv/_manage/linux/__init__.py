"""
dnnv._manage - management tool for DNNV
"""
from __future__ import annotations

import logging

from typing import List

from .environment import Environment, ProgramDependency
from ..errors import *
from .verifiers import *


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
    print("-" * max(len(vname) for vname in installed_verifiers))
    print("\n".join(installed_verifiers))
    return 0


__all__ = [
    "install",
    "uninstall",
    "list_verifiers",
] + verifiers.__all__
