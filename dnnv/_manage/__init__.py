import sys

__all__ = []

if sys.platform == "linux":
    from .linux import *

    __all__ += linux.__all__
else:
    print(f"dnnv_manage is not yet supported on {sys.platform} platforms")
    exit(1)

from .errors import *

__all__ += errors.__all__
