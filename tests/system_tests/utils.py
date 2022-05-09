import os
from pathlib import Path

artifact_dir = Path(__file__).parent / "artifacts"
network_artifact_dir = Path(__file__).parent / "artifacts" / "networks"
property_artifact_dir = Path(__file__).parent / "artifacts" / "properties"

# TODO : only modify path if not in VIRTUAL_ENV
os.environ["PATH"] = ":".join(
    [
        os.environ.get("PATH", ""),
        str(
            (
                Path(os.path.join(os.getenv("XDG_DATA_HOME", "~/.local/share"), "dnnv"))
                / "bin"
            )
            .expanduser()
            .resolve()
        ),
    ]
)
