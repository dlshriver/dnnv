
"""
Functions for loading input data.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np


def load_img(path: str, img_nums: list, shape: tuple) -> np.array:

    """
    Loads a image in the human-readable format.

    Args:
        path:
            The path to the to the folder with mnist images.
        img_nums:
            A list with the numbers of the images we want to load.
        shape:
            The shape of a single image.

    Returns:
        The images as a MxCx28x28 numpy array.
    """

    images = np.zeros((len(img_nums), *shape), dtype=float)

    for idx, i in enumerate(img_nums):
        file = os.path.join(path, "image" + str(i))
        with open(file, "r") as f:
            data = [float(pixel) for pixel in f.readlines()[0].split(",")[:-1]]
            images[idx, :, :] = np.array(data).reshape(*shape)

    return images


def load_mnist_human_readable(path: str, img_nums: list) -> np.array:

    """
    Loads a mnist image from the neurify dataset.

    Args:
        path:
            The path to the to the folder with mnist images.
        img_nums:
            A list with the numbers of the images we want to load.

    Returns:
        The images as a Mx28x28 numpy array.
    """

    return load_img(path, img_nums, (28, 28))


def load_cifar10_human_readable(path: str, img_nums: list) -> np.array:

    """
    Loads the Cifar10 images in human readable format.

    Args:
        path:
            The path to the to the folder with mnist images.
        img_nums:
            A list with the numbers of the images we want to load.

    Returns:
        The images as a Mx3x32x32 numpy array.
    """

    return load_img(path, img_nums, (3, 32, 32))


def load_images_eran(img_csv: str = "../../resources/images/cifar10_test.csv", num_images: int = 100,
                     image_shape: tuple = (3, 32, 32)) -> tuple:

    """
    Loads the images from the eran csv.

    Args:
        The csv path
    Returns:
        images, targets
    """

    num_images = 100

    images_array = np.zeros((num_images, np.prod(image_shape)), dtype=np.float32)
    targets_array = np.zeros(num_images, dtype=int)

    with open(img_csv, "r") as file:
        for j in range(num_images):
            line_arr = file.readline().split(",")
            targets_array[j] = int(line_arr[0])
            images_array[j] = [float(pixel) for pixel in line_arr[1:]]

    return images_array.reshape((num_images, *image_shape)), targets_array

