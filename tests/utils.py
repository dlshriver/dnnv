import os

from pathlib import Path

artifact_dir = Path(__file__).parent / "artifacts"
network_artifact_dir = Path(__file__).parent / "artifacts" / "networks"
property_artifact_dir = Path(__file__).parent / "artifacts" / "properties"

local_dnnv_dir = (Path.cwd() / ".dnnv").resolve()
home_dnnv_dir = (Path.home() / ".dnnv").resolve()
extend_envvar = lambda var, ext: os.path.pathsep.join(
    [
        str(p)
        for p in (
            local_dnnv_dir / ext,
            home_dnnv_dir / ext,
            os.getenv(var, ""),
        )
        if p
    ]
)
os.environ["PATH"] = extend_envvar("PATH", "bin")
os.environ["LD_LIBRARY_PATH"] = extend_envvar("LD_LIBRARY_PATH", "lib")
