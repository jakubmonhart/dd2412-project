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
import albumentations as A
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split


def display_image_and_key_points(img, kpts, move_axis=False):
  print(img.shape)
  if move_axis:
    img = np.moveaxis(img, 0, -1)
  print(img.shape)
  for kp in kpts:
    if kp[2] == 0 or kp[0] < 0 or kp[1] < 0:
      continue
    img[int(kp[1]-2):int(kp[1]+2), int(kp[0]-2):int(kp[0]+2), :] = 0
  plt.imshow(img)
  plt.show()

class CUB_dataset(VisionDataset):
  '''
  TODO
    - extract global and parts attributes.
    - train test
  '''

  root = '../data/cub'
  url = 'https://data.deepai.org/CUB200(2011).zip'
  folder_name = 'CUB_200_2011'
  patches = (14, 14)
  image_folder = os.path.join(root, folder_name, 'images')
  dataset_folder = os.path.join(root, folder_name)
  attributes_to_retain = [2,5,7,8,11,15,16,21,22,24,25,26,30,31,36,37,39,41,45,46,51,52,54,55,57,58,60,64,70,71,73,
                            76,81,91,92,102,105,107,111,112,117,118,120,126,127,132,133,135,146,150,152,153,154,158,159,
                            164,165,166,169,173,179,180,183,184,188,189,194,195,197,203,204,209,210,212,219,221,222,226,
                            228,236,237,239,241,244,245,249,250,254,255,260,261,263,269,275,278,284,290,293,294,295,299,
                            300,305,306,307,309,311,312]
  old2new_attr_id = {}
  # for i, idx in enumerate(attributes_to_retain):
  #   old2new_attr_id[idx] = i
  for i in range(1000):
    old2new_attr_id[i] = i
  
  

  def __init__(self, transform=None, target_transform=None, crop_images=True, is_test=False):  
    super().__init__(root=self.root)
    if not os.path.isdir(self.root):
      print('Making the Datafolder')
      os.makedirs(self.root)
    self.transform = transform
    self.target_transform = target_transform
    self.crop_images = crop_images
    self.transform = transform
    if os.path.exists(os.path.join(self.root, self.folder_name)):
      print('Dataset folder found. In order to download it, delete the existing folder!')
    else:
      print('Downloading CUB200-2011 Dataset')
      download_and_extract_archive(self.url, self.root, self.root, self.folder_name+'.zip')
      os.remove(os.path.join(self.root, self.folder_name+'.zip'))
      shutil.move(os.path.join(self.root, 'attributes.txt'), os.path.join(self.root, self.folder_name, 'attributes/attributes.txt'))


    self.data = self.load_image_data()

    self.parts = self.load_parts()
    
    self.clean_attributer()
    self.attributes, self.image_attributes = self.load_attributes(self.parts)
    self.attr_id2global_id, self.attr_id2local_id = self.make_global_local_attr_id(self.attributes)
    if is_test:
      self.data = self.data[self.data.is_training == 0]
    else:
      self.data = self.data[self.data.is_training == 1]
    
    self.data.reset_index(drop=True, inplace=True)

    
    
  def load_parts(self):
    parts_df = pd.read_csv(os.path.join(self.dataset_folder, 'parts', 'part_locs.txt'),
                            sep=' ', names=['image_id', 'part_id', 'x', 'y', 'visible'])
    parts_df.set_index(['image_id', 'part_id'], inplace=True)
    return parts_df


  def clean_attributer(self):
    if os.path.exists(os.path.join(self.dataset_folder, 'attributes/image_attribute_labels_processed.txt')): return
    with open(os.path.join(self.dataset_folder, 'attributes/image_attribute_labels.txt'), 'r') as original:
      with open(os.path.join(self.dataset_folder, 'attributes/image_attribute_labels_processed.txt'), 'w') as proc:
        for line in original:
          splitted = line.split()
          if len(splitted) > 5:
            del splitted[-2]
          proc.write(' '.join(splitted)+'\n')

 
  def load_image_data(self):
    images = pd.read_csv(os.path.join(self.dataset_folder, 'images.txt'), sep=' ',
                             names=['image_id', 'path'])
    
    labels = pd.read_csv(os.path.join(self.dataset_folder, 'image_class_labels.txt'),
                                         sep=' ', names=['image_id', 'target'])
    
    train_test_split = pd.read_csv(os.path.join(self.dataset_folder, 'train_test_split.txt'),
                                      sep=' ', names=['image_id', 'is_training'])

    data = images.merge(labels, on='image_id').merge(train_test_split, on='image_id')

    bounding_box = pd.read_csv(os.path.join(self.dataset_folder, 'bounding_boxes.txt'), sep=' ',
                           names=['image_id', 'x1', 'y1', 'w', 'h'])
    data = data.merge(bounding_box, on='image_id')
    return data
  
  def load_attributes(self, parts):
    attributes = pd.read_csv(os.path.join(self.dataset_folder, 'attributes/attributes.txt'),
                            sep=' ', names=['attr_id', 'def'])
    image_attributes = pd.read_csv(os.path.join(self.dataset_folder, 'attributes', 'image_attribute_labels_processed.txt'),
                              sep=' ', names=['image_id', 'attr_id', 'is_present', 'certainty', 'time'])
    # removing unused attributes
    attributes = attributes[attributes.attr_id.isin(self.attributes_to_retain)]
    attributes['attr_id'].replace(self.old2new_attr_id, inplace=True)
    image_attributes = image_attributes[image_attributes.attr_id.isin(self.attributes_to_retain)]
    image_attributes['attr_id'].replace(self.old2new_attr_id, inplace=True)
    # We used the Authors code for the part below
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
        attr_id = attributes[attributes['def'].str.split('::').str[0].str.contains(has)].attr_id
        for k in attr_id:
            attr_id2parts[k] += part
    attributes['part'] = attributes.attr_id.map(attr_id2parts)
    image_attributes['part'] = image_attributes.attr_id.map(attr_id2parts)
    self.num_global_attr = attributes['part'].value_counts()[[0]].item()
    self.num_local_attr = len(attributes) - self.num_global_attr
    return attributes, image_attributes

  def make_global_local_attr_id(self, attributes):
    attr_id2global_id = {}
    attr_id2local_id = {}
    for _, row in attributes.iterrows():
      if row['part'] == [0]:
        attr_id2global_id[row['attr_id']] = len(attr_id2global_id)
      else:
        attr_id2local_id[row['attr_id']] = len(attr_id2local_id)
    return attr_id2global_id, attr_id2local_id
  
  def _get_patch_number(self,image, x, y):
        patch_x = int(x * self.patches[1] / image.shape[1])
        patch_y = int(y * self.patches[0] / image.shape[0])
        return patch_x + (patch_y * self.patches[0])
  
  def __getitem__(self, index):
    image_metadata = self.data.loc[index]
    image_attributes = self.image_attributes[self.image_attributes['image_id'] == image_metadata['image_id']]
    image = np.array(Image.open(os.path.join(self.image_folder, image_metadata['path']))) # Image is loaded as PIL
    image_path = image_metadata['path']
    if len(image.shape) == 2:
      image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
    label = image_metadata.target - 1
    image_parts = self.parts.loc[image_metadata['image_id']].copy()
    if self.crop_images:
      values = image_parts.values
      values[:, 0] -= image_metadata.x1
      values[:, 1] -=image_metadata.y1
      image_parts.iloc[:] = values
      image = image[int(image_metadata.y1):int(image_metadata.y1 + image_metadata.h), int(image_metadata.x1):int(image_metadata.x1 + image_metadata.w), :]
    expl = torch.FloatTensor(self.num_global_attr).fill_(float('nan'))
    spatial_expl = torch.FloatTensor(self.patches[0]*self.patches[1], self.num_local_attr).fill_(float('nan'))
    if not self.transform is None:
      values = image_parts.values[:, :2]
      transformed = self.transform(image=image, keypoints=values)
      image = transformed['image']
      values = np.array(transformed['keypoints'])
      image_parts.loc[:, 'x':'y'] = values
    for _, row in image_attributes.iterrows():
      if row['is_present'] == 0:
        continue     
      if row['part'] == [0]:
        expl[self.attr_id2global_id[row.attr_id]] = row['is_present']
        expl[expl != expl] = 0 # selects only nan rows
        continue
      for p in row['part']:
        x = image_parts.loc[p].x
        y = image_parts.loc[p].y
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]: continue
        patch_id = self._get_patch_number(image, x, y)
        spatial_expl[patch_id, self.attr_id2local_id[row.attr_id]] = row['is_present']
        spatial_expl[patch_id][spatial_expl[patch_id] != spatial_expl[patch_id]] = 0

    return {
      'image': image,
      'global_attr': expl,
      'spatial_attr': spatial_expl,
      'label': label,
      'image_path': [image_path]
    }

  def __len__(self):
    return len(self.data)
  
  def get_n_labels(self):
    return self.data['target'].max()



class CUB(LightningDataModule):
  def __init__(self, batch_size=32, num_workers=4, val_size=0.1):
    super().__init__()
    self.train_transform = A.Compose([Resize(224, 224),
                                  A.HorizontalFlip(p=0.5),
                                  Rotate(limit=(-30,30),p=1.0),
                                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ToTensorV2()],
                                keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))
    self.test_transform = A.Compose([Resize(224, 224),
                                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ToTensorV2()],
                                keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.val_size = val_size
    


  
  
  def setup(self, stage=None):
    self.train = CUB_dataset(transform=self.train_transform, is_test=False)
    self.test = CUB_dataset(transform=self.test_transform, is_test=True)
    self.n_labels = self.train.get_n_labels()
    self.n_spatial_attr = self.train.num_local_attr
    self.n_global_attr = self.train.num_global_attr
    val_size = int(self.val_size * len(self.train))
    self.train, self.val = random_split(self.train, lengths=[len(self.train) - val_size, val_size])
    
    


  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

  

if __name__ == '__main__':

  cub = CUB(32, 0)
  cub.setup()
  print(len(cub.train), len(cub.test), len(cub.val))
  loader = cub.train_dataloader()
  for batch in loader:
    print(type(batch))
    for k in batch.keys():
      print(batch[k].shape)
  

  # dataset_train = CUB_dataset(crop_images=True, transform=train_transform, is_test=False)
  # print(len(dataset_train), len(dataset_test))
  
