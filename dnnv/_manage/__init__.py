import sys

from .errors import *

__all__ = [
    # errors
    "DNNVManagerError",
    "InstallError",
    "UninstallError",
]

if sys.platform == "linux":
    from .linux import *

    __all__ += [
        "install",
        "uninstall",
        "list_verifiers",
        "import_verifier_module",
        "verifier_choices",
    ]
else:
    print(f"dnnv_manage is not yet supported on {sys.platform} platforms")
    sys.exit(1)
