import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl

# Supress warnings
import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)

class MNIST(pl.LightningDataModule):

  """
  TODO
    - image transformation: augmentation
  """

  def __init__(self, batch_size, root='data', seed=42):
    super().__init__()
    self.root = root
    self.batch_size = batch_size
    self.seed = seed

  def prepare_data(self):
    torchvision.datasets.MNIST(root=self.root, train=True, download=True)
    torchvision.datasets.MNIST(root=self.root, train=False, download=True)
  
  def generate_class_concept(self, y):
    y = torch.tensor(y)
    target_concept = F.one_hot(y, num_classes=10).float() # One-hot encoding of true label
    target_class = ((y%2) == 0)*1 # even/odd
    target_class = target_class
 
    return (target_class, target_concept)

  def setup(self, stage=None):
    # For reproducibility
    self.generator = torch.Generator().manual_seed(self.seed)

    mnist = torchvision.datasets.MNIST(root=self.root, train=True)
    self.mean = torch.mean(mnist.data.float() / 255.0)
    self.std = torch.std(mnist.data.float() / 255.0)
    del mnist

    transform = transforms.Compose([
      transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))
    ])

    target_transform = transforms.Lambda(self.generate_class_concept)

    train_full = torchvision.datasets.MNIST(root=self.root, train=True, transform=transform, target_transform=target_transform)
    self.train, self.val = data.random_split(
      train_full, lengths=[0.9, 0.1], generator=self.generator)

    self.test = torchvision.datasets.MNIST(root=self.root, train=False, transform=transform, target_transform=target_transform)

  def train_dataloader(self):
    return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, generator=self.generator)

  def val_dataloader(self):
    return data.DataLoader(self.val, batch_size=self.batch_size)

  def test_dataloader(self):
    return data.DataLoader(self.test, batch_size=self.batch_size)