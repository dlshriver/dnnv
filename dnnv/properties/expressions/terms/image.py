from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import skimage.io as sio

from ...errors import NonConcreteExpressionError
from ..base import Expression
from ..context import Context
from .base import Term
from .constant import Constant


class Image(Term):
    def __init__(
        self,
        path: Union[Expression, Path, str],
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        self.path = path

    @classmethod
    def load(cls, path: Union[Expression, Path, str]):
        if isinstance(path, Expression):
            try:
                path = path.value
            except NonConcreteExpressionError:
                return Image(path)
        assert isinstance(path, (Path, str))
        path = Path(path)
        if path.suffix in [".npy", ".npz"]:
            img = np.load(path)
        else:
            img = sio.imread(path)
        return Constant(img)

    @property
    def value(self):
        if isinstance(self.path, Expression):
            return Image.load(self.path.value).value
        return Image.load(self.path).value

    def __hash__(self):
        return super().__hash__() * hash(self.path)

    def __repr__(self):
        return f"Image({self.path!r})"

    def __str__(self):
        return f"Image({self.path})"
