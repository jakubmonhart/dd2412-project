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
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class CUB_dataset(VisionDataset):
  '''
  TODO
    - extract global and parts attributes.
    - train test
  '''

  root = '../data/cub'
  # url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
  url = 'https://data.deepai.org/CUB200(2011).zip'
  folder_name = 'CUB_200_2011'
  croped_folder = os.path.join(root, folder_name, 'croped')
  image_folder = os.path.join(root, folder_name, 'images')
  dataset_folder = os.path.join(root, folder_name)

  def __init__(self):  
    super().__init__(root=self.root, transforms=None, target_transform=None)
    if not os.path.isdir(self.root):
      print('Making the Datafolder')
      os.makedirs(self.root)
    
    if os.path.exists(os.path.join(self.root, self.folder_name)):
      print('Dataset folder found. In order to download it, delete the existing folder!')
    else:
      print('Downloading CUB200-2011 Dataset')
      download_and_extract_archive(self.url, self.root, self.root, self.folder_name+'.tgz')
      os.remove(os.path.join(self.root, self.folder_name+'.tgz'))
      shutil.move(os.path.join(self.root, 'attributes.txt'), os.path.join(self.root, self.folder_name, 'attributes/attributes.txt'))

    images = pd.read_csv(os.path.join(self.dataset_folder, 'images.txt'), sep=' ',
                             names=['id', 'path'])
    
    labels = pd.read_csv(os.path.join(self.dataset_folder, 'image_class_labels.txt'),
                                         sep=' ', names=['id', 'target'])
    
    train_test_split = pd.read_csv(os.path.join(self.dataset_folder, 'train_test_split.txt'),
                                      sep=' ', names=['id', 'is_training'])

    data = images.merge(labels, on='id').merge(train_test_split, on='id')

    bounding_box = pd.read_csv(os.path.join(self.dataset_folder, 'bounding_boxes.txt'), sep=' ',
                           names=['id', 'x1', 'y1', 'w', 'h'])
    data = data.merge(bounding_box, on='id')

    parts = pd.read_csv(os.path.join(self.dataset_folder, 'parts', 'part_locs.txt'),
                            sep=' ', names=['id', 'part_id', 'x', 'y', 'visible'])
    
    attr_list, attributes, attr_id2parts = self._load_attributes()






    # print(parts.head())
    print('\n\n\n')
    print(type(attr_list), type(attributes), type(attr_id2parts))
    print('\n\n\n')
    # img = np.array(Image.open(os.path.join(self.image_folder, data.iloc[10]['path'])))
    # img[:,int(data.iloc[10]['x1']),:] = 0
    # img[:, int(data.iloc[10]['x1'] + data.iloc[10]['w']),:] = 0
    # img[int(data.iloc[10]['y1']),:,:] = 0
    # img[int(data.iloc[10]['y1'] + data.iloc[10]['h']),:,:] = 0
    # plt.imshow(img)
    # plt.show()

    if not os.path.exists(self.croped_folder):
      print('Croping images and saving them!')
      pass # TODO crop images
    else:
      print('Croped foler found!')
    

    
    assert False
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
  
  def _load_attributes(self):
    # Load list of attributes
    attr_list = pd.read_csv(os.path.join(self.dataset_folder, 'attributes/attributes.txt'),
                            sep=' ', names=['attr_id', 'def'])
    # Find parts corresponding to each attribute
    has2part = {
        'bill': [2],
        'wing': [9, 13],
        'has_size': [0],   # zero means non-spatial attribute, starts with "has_size::"
        'has_shape': [0],  # zero means non-spatial attribute, starts with "has_shape::"
        'upperparts': [1, 10],
        'underparts': [3, 4, 15],
        'breast': [4],
        'back': [1],
        'tail': [14],
        'head': [5],
        'throat': [15],  # was 'throat': [17] hefore (typo?)
        'eye': [7, 11],
        'forehead': [6],
        'nape': [10],
        'belly': [3],
        'leg': [8, 12],
        'has_primary_color': [0],  # zero means non-spatial attribute, starts with "has_primary_color::"
        'crown': [5],
    }
    attr_id2parts = defaultdict(list)
    for has, part in has2part.items():
        # need to check only attribute categories (left of "::" in the source), not attribute category values
        # see e.g. leg in "225 has_shape::long-legged-like"
        attr_id = attr_list[attr_list['def'].str.split('::').str[0].str.contains(has)].attr_id
        for k in attr_id:
            attr_id2parts[k] += part

    # Load attributes of each image
    attributes = pd.read_csv(os.path.join(self.dataset_folder, 'attributes', 'image_attribute_labels.txt'),
                              sep=' ', names=['id', 'attr_id', 'is_present', 'certainty', 'time'])
    attributes.img_id = attributes.img_id.astype(int)
    attributes.attr_id = attributes.attr_id.astype(int)
    attributes.is_present = attributes.is_present.astype(int)
    return attr_list, attributes, attr_id2parts

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



  

if __name__ == '__main__':
  dataset = CUB_dataset()