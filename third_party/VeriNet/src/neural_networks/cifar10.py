
"""
A Cifar10 model used for testing the verification algorithm

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

from src.neural_networks.verinet_nn import VeriNetNN


class Cifar10(VeriNetNN):

    """
    A simple fully-convolutional network for Cifar10 classification
    """

    def __init__(self, use_gpu: bool=False):

        """
        Args:
            use_gpu     : If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        layers = [
                  nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(32, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(64, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(64, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(128, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(256, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(256, 64, 2, 1, 0), nn.ReLU()),
                  nn.Sequential(nn.BatchNorm2d(64, momentum=0.05)),

                  nn.Sequential(nn.Conv2d(64, 10, 1, 1, 0))
                  ]

        super().__init__(layers, use_gpu=use_gpu)

        self.cifar10_train = None
        self.cifar10_val = None
        self.cifar10_test = None
        self.loader_train = None
        self.loader_val = None
        self.loader_test = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward calculations

        Args:
            x   : The input, should be BxN for FC or BxNxHxW for Conv2d, where B is the batch size, N is the number
                  of nodes, H his the height and W is the width.

        Returns:
            The network output, of same shape as the input
        """

        y = super().forward(x)
        return F.log_softmax(y, dim=1)

    def init_data_loader(self, data_dir: str, num_train: int=49000):

        """
        Initializes the data loaders.

        If the data isn't found, it will be downloaded

        Args:
            data_dir    : The directory of the data
            num_train   : The number of training examples used
        """

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        trns_norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        trns_aug = T.Compose([T.RandomAffine(15),
                              T.ColorJitter(brightness=0.15, contrast=0.15),
                              T.RandomHorizontalFlip(),
                              T.RandomCrop(size=32, padding=4),
                              T.ToTensor(),
                              T.Normalize(mean, std)])

        self.cifar10_train = dset.CIFAR10(data_dir, train=True, download=True, transform=trns_aug)
        self.loader_train = DataLoader(self.cifar10_train, batch_size=64,
                                       sampler=sampler.SubsetRandomSampler(range(num_train)))

        self.cifar10_val = dset.CIFAR10(data_dir, train=True, download=True, transform=trns_norm)
        self.loader_val = DataLoader(self.cifar10_val, batch_size=64,
                                     sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

        self.cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)
        self.loader_test = DataLoader(self.cifar10_test, batch_size=100)

    def check_accuracy(self, loader: DataLoader) -> tuple:

        """
        Calculates and returns the accuracy of the current model

        Args:
             loader: The data loader for the dataset used to calculate accuracy
        Returns:
            (num_correct, num_samples, accuracy). The number of correct classifciations, the total number of samples
            and the accuracy in percent.
        """

        num_correct = 0
        num_samples = 0

        self.eval()

        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # noinspection PyCallingNonCallable
                scores = self(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            acc = float(num_correct) / num_samples

        return num_correct, num_samples, acc

    def train_model(self, epochs=10, lr=1e-3, l1_reg: float=0, weight_decay=0, verbose: bool=True,):

        """
        Trains the model

        Args:
            epochs          : The number of epochs to train the model
            lr              : The learning rate
            l1_reg          : The l1 regularization multiplier
            weight_decay    : The weight decay used
            verbose         : If true, training progress is printed
        """

        msg = "Initialize data loaders before calling train_model"
        assert (self.loader_train is not None) and (self.loader_val is not None), msg

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        model = self.to(device=self.device)

        for e in range(epochs):

            if verbose:
                print(f"Dataset size: {len(self.loader_train)}")

            for t, (x, y) in enumerate(self.loader_train):

                model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = model(x)

                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))

                loss = F.cross_entropy(scores, y) + l1_reg*regularization_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % 100 == 0:
                    num_correct, num_samples, acc = self.check_accuracy(self.loader_val)
                    print(f"Epoch: {e}, Iteration {t}, loss = {loss.item():.4f}")
                    print(f"Validation set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")

            if verbose:
                num_correct, num_samples, acc = self.check_accuracy(self.loader_train)
                print(f"Training set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")

        num_correct, num_samples, acc = self.check_accuracy(self.loader_test)
        print(f"Final test set results: {num_correct} / {num_samples} correct ({100 * acc:.2f})")
