"""
Small script for benchmarking the Cifar10 network

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np

from src.scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_cifar10_human_readable

if __name__ == "__main__":

    epsilons = np.array([0.05, 0.1, 0.2, 0.5, 1]) / 255  # Cifar10 images use pixel values 0-1 instead of 0-255
    timeout = 3600

    num_images = 50
    img_dir: str = f"../../data/cifar10_torch/test_images_100/"

    if not os.path.isdir("../../benchmark_results"):
        os.mkdir("../../benchmark_results")

    run_benchmark(images=load_cifar10_human_readable(img_dir, list(range(num_images))),
                  epsilons=epsilons,
                  timeout=timeout,
                  conv=True,
                  model_path="../../data/models_nnet/cifar10_conv.nnet",
                  result_path=f"../../benchmark_results/cifar10_{num_images}_imgs.txt",
                  max_procs=5)
