"""
Small script for benchmarking the FC MNIST Tanh network trained with PGD=0.1 from:

https://github.com/eth-sri/eran

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np

from src.scripts.benchmark import run_benchmark
from src.util.logger import get_logger
from src.util.config import *

benchmark_logger = get_logger(LOGS_LEVEL, __name__, "../../logs/", "benchmark_log")


def load_images() -> tuple:
    """
    Loads the images from the eran csv.

    Returns:
        images, targets
    """

    num_images = 100
    img_csv: str = "../../data/mnist_eran/mnist_test.csv"
    images_array = np.zeros((num_images, 784), dtype=np.float32)
    targets_array = np.zeros(num_images, dtype=int)

    with open(img_csv, "r") as file:
        for j in range(num_images):
            line_arr = file.readline().split(",")
            targets_array[j] = int(line_arr[0])
            images_array[j] = [float(pixel) for pixel in line_arr[1:]]

    return images_array, targets_array


if __name__ == "__main__":

    epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    timeout = 3600

    if not os.path.isdir("../../benchmark_results"):
        os.mkdir("../../benchmark_results")

    images, targets = load_images()
    images = images / 255

    run_benchmark(images=images,
                  epsilons=epsilons,
                  timeout=timeout,
                  conv=False,
                  model_path="../../data/models_nnet/ffnnTANH__PGDK_w_0.1_6_500.nnet",
                  result_path=f"../../benchmark_results/mnist_tanh.txt",
                  targets=targets)
