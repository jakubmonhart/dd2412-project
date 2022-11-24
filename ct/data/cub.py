# images and annotations: https://data.caltech.edu/records/65de6-vp158
# segmentations: https://data.caltech.edu/records/w9d68-gec53

# TODO - reproducibility - fix the seed?

import os

import pandas as pd

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms

from pytorch_lightning import LightningDataModule


class CUB_dataset(VisionDataset):
  '''
  TODO
    - extract global and parts attributes.
    - train test
  '''

  root = '../data/cub'
  url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'

  def __init__(self):  
    super().__init__(root=self.root, transforms=None, target_transform=None)
  
    # TODO - download only if data is not existent. Move attributes.txt inside the folder
    # # Download dataset
    # print('Downloading dataset')
    # download_and_extract_archive(url=self.url, download_root=self.root)

    # Read metadata
    images = pd.read_csv(os.path.join(self.root, 'images.txt'), names=['id', 'filepath'], sep=' ')
    labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'), names=['id', 'label'], sep=' ')

    self.metadata = images.merge(labels, on='id')

    # Transformation - TODO - move outside?
    # TODO - Resize target dimension ... is it mentioned in the paper? I've taken the value from the paper implemetation
    # Normalize using imagenet mean and std (we are using backbones pretrained on imagenet)
    self.transform = transforms.Compose([
      transforms.Resize((224, 224)), transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  def __getitem__(self, index):
    '''
    Returns: 
      tuple: (image, label) where image is torch.Tensor in (..., C, H, W) shape - channels first
    '''

    image_metadata = self.metadata.loc[index]
    path = os.path.join(self.root, 'images', image_metadata.filepath)
    image = default_loader(path) # Image is loaded as PIL
    label = image_metadata.label - 1 # Shift label to [0,N_CLASSES-1]

    image = self.transform(image) # One of the transformation casts the image from PIL to torch.Tensor
  
    return image, label

  def __len__(self):
    return len(self.metadata)


class CUB(LightningDataModule):
  def __init__(self):
    super().__init__()

    self.batch_size = 8

  def prepare_data(self):
    # TODO - download here
    pass
  
  def setup(self, stage):
    self.train = CUB_dataset()
    self.val = CUB_dataset()
    self.test = CUB_dataset()

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size)