import sys

from .errors import *

if sys.platform == "linux":
    from .linux import *
else:
    print(f"dnnv_manage is not yet supported on {sys.platform} platforms")
    exit(1)

__all__ = [
    # platform module
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
