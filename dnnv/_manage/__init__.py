import sys

from .errors import DNNVManagerError, InstallError, UninstallError

if sys.platform == "linux":
    from .linux import (
        import_verifier_module,
        install,
        list_verifiers,
        uninstall,
        verifier_choices,
    )
else:
    print(f"dnnv_manage is not yet supported on {sys.platform} platforms")
    sys.exit(1)

__all__ = [
    # platform specific management methods
    "install",
    "uninstall",
    "list_verifiers",
    "import_verifier_module",
    "verifier_choices",
    # errors
    "DNNVManagerError",
    "InstallError",
    "UninstallError",
]
