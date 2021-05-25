"""
dnnv._manage - management tool for DNNV
"""
from __future__ import annotations

import logging

from typing import List

from . import install
from .verifiers import import_verifier_module


def install_command(verifiers: List[str]) -> int:
    logger = logging.getLogger("dnnv_manage.install_command")
    installation_manager = install.InstallationManager()
    for verifier in verifiers:
        logger.info("installing %s", verifier)
        import_verifier_module(verifier).configure_install(installation_manager)
    return 0


__all__ = ["install_command"]
