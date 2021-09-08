import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Gemm_all_attributes():
    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    op = Gemm(a, b, c, alpha=0.25, beta=0.35, transpose_a=1, transpose_b=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = 0.25 * a.T @ b.T + 0.35 * c
    assert result.shape == y.shape
    assert np.allclose(result, y)

    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    a_input_op = Input((4, 3), np.dtype(np.float32))
    op = Gemm(a_input_op, b, c, alpha=0.25, beta=0.35, transpose_a=1, transpose_b=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(a).numpy()
    y = 0.25 * a.T @ b.T + 0.35 * c
    assert result.shape == y.shape
    assert np.allclose(result, y)

    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    b_input_op = Input((5, 4), np.dtype(np.float32))
    op = Gemm(a, b_input_op, c, alpha=0.25, beta=0.35, transpose_a=1, transpose_b=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(b).numpy()
    y = 0.25 * a.T @ b.T + 0.35 * c
    assert result.shape == y.shape
    assert np.allclose(result, y)

    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    c_input_op = Input((1, 5), np.dtype(np.float32))
    op = Gemm(a, b, c_input_op, alpha=0.25, beta=0.35, transpose_a=1, transpose_b=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(c).numpy()
    y = 0.25 * a.T @ b.T + 0.35 * c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_alpha():
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    op = Gemm(a, b, c, alpha=0.5)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = 0.5 * a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_beta():
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    op = Gemm(a, b, c, beta=0.5)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + 0.5 * c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_matrix_bias():
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.random.ranf([3, 4]).astype(np.float32)
    op = Gemm(a, b, c)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_no_bias():
    a = np.random.ranf([2, 10]).astype(np.float32)
    b = np.random.ranf([10, 3]).astype(np.float32)
    op = Gemm(a, b)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_scalar_bias():
    a = np.random.ranf([2, 3]).astype(np.float32)
    b = np.random.ranf([3, 4]).astype(np.float32)
    c = np.array(3.14).astype(np.float32)
    op = Gemm(a, b, c)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_single_elem_vector_bias():
    a = np.random.ranf([3, 7]).astype(np.float32)
    b = np.random.ranf([7, 3]).astype(np.float32)
    c = np.random.ranf([1]).astype(np.float32)
    op = Gemm(a, b, c)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_vector_bias():
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    op = Gemm(a, b, c)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_default_zero_bias():
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    op = Gemm(a, b, c)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_transpose_a():
    a = np.random.ranf([6, 3]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    op = Gemm(a, b, c, transpose_a=True)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a.T @ b + c
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gemm_transpose_b():
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([4, 6]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    op = Gemm(a, b, c, transpose_b=True)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = a @ b.T + c
    assert result.shape == y.shape
    assert np.allclose(result, y)
