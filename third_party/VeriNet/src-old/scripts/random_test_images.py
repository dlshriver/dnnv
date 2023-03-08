
"""
Small scripting for selecting random images for testing

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from src.scripts.benchmark import pick_random_mnist_images, pick_random_cifar10_images

num_images = 100
pick_random_mnist_images(num_img=num_images)
pick_random_cifar10_images(num_img=num_images)

