import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Conv_x_is_op():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    w = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    b = np.array([0], dtype=np.float32)

    input_op = Input((1, 1, 5, 5), np.dtype(np.float32))
    op = Conv(
        input_op, w, b, kernel_shape=np.array([3, 3]), pads=np.array([1, 1, 1, 1])
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    y = np.array(
        [
            [
                [
                    [12.0, 21.0, 27.0, 33.0, 24.0],  # (1, 1, 5, 5) output tensor
                    [33.0, 54.0, 63.0, 72.0, 51.0],
                    [63.0, 99.0, 108.0, 117.0, 81.0],
                    [93.0, 144.0, 153.0, 162.0, 111.0],
                    [72.0, 111.0, 117.0, 123.0, 84.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Conv_with_padding():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    w = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    b = np.array([0], dtype=np.float32)

    op = Conv(x, w, b, kernel_shape=np.array([3, 3]), pads=np.array([1, 1, 1, 1]))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [
            [
                [
                    [12.0, 21.0, 27.0, 33.0, 24.0],  # (1, 1, 5, 5) output tensor
                    [33.0, 54.0, 63.0, 72.0, 51.0],
                    [63.0, 99.0, 108.0, 117.0, 81.0],
                    [93.0, 144.0, 153.0, 162.0, 111.0],
                    [72.0, 111.0, 117.0, 123.0, 84.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Conv_without_padding():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    w = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    b = np.array([0], dtype=np.float32)

    op = Conv(x, w, b, kernel_shape=np.array([3, 3]))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [
            [
                [
                    [54.0, 63.0, 72.0],  # (1, 1, 3, 3) output tensor
                    [99.0, 108.0, 117.0],
                    [144.0, 153.0, 162.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Conv_with_padding_and_strides():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    w = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    b = None

    op = Conv(
        x,
        w,
        b,
        kernel_shape=np.array([3, 3]),
        pads=np.array([1, 1, 1, 1]),
        strides=np.array([2, 2]),
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [
            [
                [
                    [12.0, 27.0, 24.0],  # (1, 1, 4, 3) output tensor
                    [63.0, 108.0, 81.0],
                    [123.0, 198.0, 141.0],
                    [112.0, 177.0, 124.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Conv_without_padding_with_strides():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    w = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    b = None

    op = Conv(
        x,
        w,
        b,
        kernel_shape=np.array([3, 3]),
        strides=np.array([2, 2]),
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [
            [
                [
                    [54.0, 72.0],  # (1, 1, 3, 2) output tensor
                    [144.0, 162.0],
                    [234.0, 252.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
