from __future__ import annotations


class DNNVManagerError(Exception):
    pass


class InstallError(DNNVManagerError):
    pass


class UninstallError(DNNVManagerError):
    pass


__all__ = ["DNNVManagerError", "InstallError", "UninstallError"]
