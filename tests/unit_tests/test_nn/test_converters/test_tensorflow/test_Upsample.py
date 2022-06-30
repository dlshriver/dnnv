import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Upsample():
    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    output = np.array(
        [
            [
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    op = Upsample(data, scales=scales, mode="nearest")
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.all(result == output)
