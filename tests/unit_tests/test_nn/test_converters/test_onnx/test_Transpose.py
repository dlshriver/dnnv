import itertools
import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Transpose():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    z = np.transpose(data)

    op = Transpose(data)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)

    op = Transpose(
        Input(shape, np.dtype(np.float32)),
    )
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [data])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)


def test_Transpose_all_permutations():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    permutations = list(itertools.permutations(np.arange(len(shape))))

    for permutation in permutations:
        z = np.transpose(data, permutation)
        op = Transpose(data, permutation=np.asarray(permutation))
        onnx_model = convert(OperationGraph([op]))

        results = onnxruntime.backend.run(onnx_model, [])
        assert len(results) == 1
        result = results[0]
        assert np.allclose(result, z)
