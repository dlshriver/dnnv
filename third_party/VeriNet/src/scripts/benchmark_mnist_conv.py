"""
Small script for benchmarking the MNIST 1024 network

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np

from src.scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_neurify_mnist

if __name__ == "__main__":

    epsilons = [1, 2, 5, 10, 15]
    timeout = 3600

    num_images = 50
    img_dir: str = f"../../data/mnist_neurify/test_images_100/"

    if not os.path.isdir("../../benchmark_results"):
        os.mkdir("../../benchmark_results")

    run_benchmark(images=load_neurify_mnist(img_dir, list(range(num_images)))[:, np.newaxis, :, :],
                  epsilons=epsilons,
                  timeout=timeout,
                  conv=True,
                  model_path="../../data/models_nnet/neurify/conv.nnet",
                  result_path=f"../../benchmark_results/mnist_{num_images}_imgs_conv.txt")
