
"""
Scripts used for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import time
import random

import numpy as np
import torch
from tqdm import tqdm
from shutil import copyfile, rmtree
import gurobipy as grb
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transform

from src.algorithm.verinet import VeriNet
from src.algorithm.verinet_util import Status
from src.algorithm.verification_objectives import LocalRobustnessObjective
from src.data_loader.nnet import NNET
from src.util.logger import get_logger
from src.util.config import *

random_seed = 0
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

benchmark_logger = get_logger(LOGS_LEVEL, __name__, "../../logs/", "benchmark_log")


def pick_random_mnist_images(path: str= "../../data/mnist_neurify/", max_img_num: int=999, num_img: int=75):

    """
    Picks random images from the mnist test-dataset and stores them in human-readable files

    The images in the the given directory should be named image"num" where num is a number from 0-999. The function
    creates a subdirectory and copies num_img random samples into this directory.

    Args:
        path        : The path of the mnist_neurify dataset
        max_img_num : The maximum image number in the original dataset
        num_img     : The number of random images to pick
    """

    if num_img > max_img_num:
        benchmark_logger.error(f"max_image_num: {max_img_num} should be larger than {num_img}")
        raise ValueError(f"max_image_num: {max_img_num} should be larger than {num_img}")

    random_images_dir = os.path.join(path, f"test_images_{num_img}")

    if os.path.isdir(random_images_dir):
        rmtree(random_images_dir)

    os.mkdir(random_images_dir)

    rand_img_nums = np.arange(max_img_num)
    np.random.shuffle(rand_img_nums)
    rand_img_nums = rand_img_nums[:num_img]

    for i, num in enumerate(rand_img_nums):

        filename = f"image{num}"
        new_filename = f"image{i}"
        test_img_dir = f"test_images_{num_img}"

        copyfile(os.path.join(path, filename), os.path.join(os.path.join(path, test_img_dir), new_filename))
        benchmark_logger.debug(f"Image num: {num} copied")


def pick_random_cifar10_images(path: str="../../data/cifar10_torch/", num_img: int=75):

    """
    Picks random images from the cifar10 test-dataset and stores them in human-readable files

    Since the cifar10 test-set is much larger than the mnist testset we don't store all images in human-readable
    format, only the random images picked. Torch data loaders are used to download/ read the data.

    Args:
        path        : The path of the cifar10 dataset in torch format.
        num_img     : The number of random images to pick
    """

    if num_img > 10000:
        msg = f"num_img: {num_img} should be smaller than 10 000"
        benchmark_logger.error(msg)
        raise ValueError(msg)

    cifar10_test = dset.CIFAR10(path, train=False, download=True, transform=transform.ToTensor())
    loader_test = DataLoader(cifar10_test, batch_size=10000)
    images, targets = iter(loader_test).next()

    random_images_dir = os.path.join(path, f"test_images_{num_img}")

    if os.path.isdir(random_images_dir):
        rmtree(random_images_dir)

    os.mkdir(random_images_dir)

    for num in range(num_img):
        with open(os.path.join(random_images_dir, f"image{num}"), "w") as f:
            for channel in images[num]:
                for row in channel:
                    for pixel in row:
                        f.write(f"{pixel},")

    with open(os.path.join(random_images_dir, f"targets"), "w") as f:
        f.write("// GT targets of the images in this folder\n")
        for num in range(num_img):
            f.write(f"{targets[num]},")


# noinspection PyArgumentList,PyShadowingNames
def run_benchmark(images: np.array,
                  epsilons: list,
                  model_path: str,
                  conv: bool,
                  timeout: int,
                  result_path: str,
                  targets: np.array=None,
                  max_procs: int=None,
                  ):

    """
    Runs benchmarking for networks

    Args:
        images      : The images used for benchmarking, should be NxM where N is the number of images and M is the
                      number of pixels for FC networks and NxChannelsxHeightxWidth for convolutional networks.
        epsilons    : A list with the epsilons (maximum pixel change)
        model_path  : The path where the nnet model is stored
        conv        : Has to be true if the given model is a convolutional network
        timeout     : The maximum time in settings before timeout for each image
        result_path : The path where the results are stored
        targets     : The correct classes for the input, if None the predictions are used as correct classes
        max_procs   : The maximum number of processes used.
    """

    # Get the "Academic license" print from gurobi at the beginning
    grb.Model()

    nnet = NNET(model_path)
    model = nnet.from_nnet_to_verinet_nn()
    model.eval()

    img_shape = images.shape[1:]
    batch_size = images.shape[0]

    if targets is None:
        targets = model(torch.Tensor(nnet.normalize_input(images.reshape(batch_size, -1)).
                                     reshape((batch_size, *img_shape)))).argmax(dim=1).numpy()

    if os.path.isfile(result_path):
        copyfile(result_path, result_path + ".bak")

    with open(result_path, 'w', buffering=1) as f:

        benchmark_logger.info(f"Starting benchmarking with timeout: {timeout},  model path: {model_path}")
        f.write(f"Benchmarking with:"
                f"Timeout {timeout} seconds \n" +
                f"Model path: {model_path} \n\n")

        solver = VeriNet(model,
                         gradient_descent_max_iters=5,
                         gradient_descent_step=1e-1,
                         gradient_descent_min_loss_change=1e-2,
                         max_procs=max_procs)

        for eps in epsilons:

            safe = []
            unsafe = []
            undecided = []
            underflow = []

            benchmark_logger.info(f"Starting benchmarking with epsilon: {eps}")
            f.write(f"Benchmarking with epsilon = {eps}: \n\n")
            solver_time = 0

            for i in tqdm(range(len(images))):
                data_i = images[i]

                # Test that the data point is classified correctly
                data_i_flat = data_i.reshape(-1)
                data_i_norm = nnet.normalize_input(data_i_flat).reshape(data_i.shape)
                pred_i = model(torch.Tensor(data_i_norm)).argmax(dim=1).numpy()[0]
                if pred_i != targets[i]:
                    f.write(f"Final result of input {i}: Skipped,correct_label: {targets[i]}, predicted: {pred_i}\n")
                    continue

                # Create input bounds
                if conv:
                    input_bounds = np.zeros((*data_i.shape, 2), dtype=np.float32)
                    input_bounds[:, :, :, 0] = data_i - eps
                    input_bounds[:, :, :, 1] = data_i + eps

                    input_bounds[:, :, :, 0] = nnet.normalize_input(input_bounds[:, :, :, 0].reshape(-1)).\
                        reshape(*data_i.shape)
                    input_bounds[:, :, :, 1] = nnet.normalize_input(input_bounds[:, :, :, 1].reshape(-1)). \
                        reshape(*data_i.shape)
                else:
                    input_bounds = np.zeros((data_i.shape[0], 2), dtype=np.float32)
                    input_bounds[:, 0] = data_i - eps
                    input_bounds[:, 1] = data_i + eps

                    input_bounds = nnet.normalize_input(input_bounds)

                # Run verification
                start = time.time()
                objective = LocalRobustnessObjective(int(targets[i]), input_bounds, output_size=10)
                status = solver.verify(objective,
                                       timeout=timeout,
                                       no_split=False,
                                       gradient_descent_intervals=5,
                                       verbose=False)

                f.write(f"Final result of input {i}: {status}, branches explored: {solver.branches_explored}, "
                        f"max depth: {solver.max_depth}, time spent: {time.time()-start:.2f} seconds\n")
                solver_time += time.time() - start

                if status == Status.Safe:
                    safe.append(i)
                elif status == Status.Unsafe:
                    unsafe.append(i)
                elif status == Status.Undecided:
                    undecided.append(i)
                elif status == Status.Underflow:
                    benchmark_logger.warning(f"Underflow for image {i}")
                    underflow.append(i)

            f.write("\n")
            f.write(f"Time spent in solver: {solver_time}\n")
            f.write(f"Total number of images verified as safe: {len(safe)}\n")
            f.write(f"Safe images: {safe}\n")
            f.write(f"Total number of images verified as unsafe: {len(unsafe)}\n")
            f.write(f"Unsafe images: {unsafe}\n")
            f.write(f"Total number of images timed-out: {len(undecided)}\n")
            f.write(f"Timed-out images: {undecided}\n")
            f.write(f"Total number of images with underflow: {len(underflow)}\n")
            f.write(f"Underflow images: {underflow}\n")
            f.write("\n")
