import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
DATASET_LOC = './data_new'

# list of all datasets
DATASETS = ["mnist"]


def get_dataset(dataset: str, split: str) -> Dataset:
	"""Return the dataset as a PyTorch Dataset object"""
	if dataset == "mnist":
            return _mnist(split)


def get_num_classes(dataset: str):
	"""Return the number of classes in the dataset. """
	if dataset == "mnist":
            return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
	"""Return the dataset's normalization layer"""
	if dataset == "mnist":
            return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)


_MNIST_MEAN = [0.5, ]
_MNIST_STDDEV = [0.5, ]

def _mnist(split: str) -> Dataset:
	if split == "train":
		return datasets.MNIST(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
		transforms.RandomCrop(28),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
	]))
	elif split == "test":
		return datasets.MNIST(DATASET_LOC, train=False, transform=transforms.ToTensor())


class NormalizeLayer(torch.nn.Module):
	"""Standardize the channels of a batch of images by subtracting the dataset mean
	  and dividing by the dataset standard deviation.

	  In order to certify radii in original coordinates rather than standardized coordinates, we
	  add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
	  layer of the classifier rather than as a part of preprocessing as is typical.
	  """

	def __init__(self, means: List[float], sds: List[float]):
		"""
		:param means: the channel means
		:param sds: the channel standard deviations
		"""
		super(NormalizeLayer, self).__init__()
		self.means = torch.tensor(means).to(device)
		self.sds = torch.tensor(sds).to(device)

	def forward(self, input: torch.tensor):
		(batch_size, num_channels, height, width) = input.shape
		means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
		sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
		#print(input)
		return (input - means) / sds
