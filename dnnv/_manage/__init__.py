import sys

if sys.platform == "linux":
    from .linux import *
else:
    print(f"dnnv_manage is not yet supported on {sys.platform} platforms")
    exit(1)

from .errors import *
