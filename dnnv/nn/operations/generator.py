from typing import Optional, Sequence

import numpy as np
from onnx import NodeProto

from ..utils import as_numpy
from .base import Operation


class RandomUniform(Operation):
    def __init__(
        self,
        shape: Sequence[int],
        *,
        dtype: np.dtype = np.dtype(np.float32),
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[float] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high
        self.seed = seed

    @classmethod
    def from_onnx(cls, onnx_node: NodeProto, *_):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        shape = attributes["shape"]
        dtype = attributes.get("dtype", np.dtype(np.float32))
        high = attributes.get("high", 1.0)
        low = attributes.get("low", 0.0)
        seed = attributes.get("seed")
        return cls(
            shape=shape,
            dtype=dtype,
            high=high,
            low=low,
            seed=seed,
            name=onnx_node.name,
        )


__all__ = ["RandomUniform"]
