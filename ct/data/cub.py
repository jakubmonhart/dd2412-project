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
  
  

  def __init__(self, transform=None, target_transform=None, crop_images=True):  
    super().__init__(root=self.root)
    if not os.path.isdir(self.root):
      print('Making the Datafolder')
      os.makedirs(self.root)
    self.transform = transform
    self.target_transform = target_transform
    self.crop_images = crop_images
    if os.path.exists(os.path.join(self.root, self.folder_name)):
      print('Dataset folder found. In order to download it, delete the existing folder!')
    else:
      print('Downloading CUB200-2011 Dataset')
      download_and_extract_archive(self.url, self.root, self.root, self.folder_name+'.tgz')
      os.remove(os.path.join(self.root, self.folder_name+'.tgz'))
      shutil.move(os.path.join(self.root, 'attributes.txt'), os.path.join(self.root, self.folder_name, 'attributes/attributes.txt'))


    self.data = self.load_image_data()

    self.parts = self.load_parts()
    
    self.clean_attributer()
    self.attributes, self.image_attributes = self.load_attributes(self.parts)
    self.attr_id2global_id, self.attr_id2local_id = self.make_global_local_attr_id(self.attributes)
    
    # print(self.has2part)
    # assert False
    # # attr_id2parts = defaultdict(list)
    # # for has, part in self.has2part.items():
    # #     attr_id = attr_list[attr_list['def'].str.split('::').str[0].str.contains(has)].attr_id
    # #     for k in attr_id:
    # #         attr_id2parts[k] += part


    # # print(parts.head())
    # # print('\n\n\n')
    # # print(self.data.head())
    # # print('\n\n\n')
    # img = np.array(Image.open(os.path.join(self.image_folder, data.iloc[10]['path'])))
    # # img[:,int(data.iloc[10]['x1']),:] = 0
    # # img[:, int(data.iloc[10]['x1'] + data.iloc[10]['w']),:] = 0
    # # img[int(data.iloc[10]['y1']),:,:] = 0
    # # img[int(data.iloc[10]['y1'] + data.iloc[10]['h']),:,:] = 0
    # plt.imshow(img)
    # plt.show()

    # if not os.path.exists(self.croped_folder):
    #   print('Croping images and saving them!')
    #   pass # TODO crop images
    # else:
    #   print('Croped foler found!')
    
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
    '''
    Returns: 
      tuple: (image, label) where image is torch.Tensor in (..., C, H, W) shape - channels first
    '''

    image_metadata = self.data.loc[index]
    image_attributes = self.image_attributes[self.image_attributes['image_id'] == image_metadata['image_id']]
    image = np.array(Image.open(os.path.join(self.image_folder, image_metadata['path']))) # Image is loaded as PIL
    label = image_metadata.target - 1
    # print(image_metadata)
    # print(image_attributes)
    # print(image_parts)
    x_subtract = 0
    y_subtract = 0
    if self.crop_images:
      x_subtract = image_metadata.x1
      y_subtract = image_metadata.y1
      image = image[int(image_metadata.y1):int(image_metadata.y1 + image_metadata.h), int(image_metadata.x1):int(image_metadata.x1 + image_metadata.w),:]
    if not self.transform is None:
      image = self.transform(image) # One of the transformation casts the image from PIL to torch.Tensor

    expl = torch.FloatTensor(self.num_global_attr).fill_(float('nan'))
    spatial_expl = torch.FloatTensor(self.patches[0]*self.patches[1], self.num_local_attr).fill_(float('nan'))
    for index, row in image_attributes.iterrows():
      # print(row)
      # print('\n') 
      if row['is_present'] == 0:
        continue     
      if row['part'] == [0]:
        expl[self.attr_id2global_id[row.attr_id]] = row['is_present']
        expl[expl != expl] = 0 # selects only nan rows
        continue
      for p in row['part']:
        x = self.parts.loc[row['image_id'], p].x - x_subtract
        y = self.parts.loc[row['image_id'], p].y - y_subtract
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]: continue
        patch_id = self._get_patch_number(image, 
                            x,
                            y)
        spatial_expl[patch_id, self.attr_id2local_id[row.attr_id]] = row['is_present']
        spatial_expl[patch_id][spatial_expl[patch_id] != spatial_expl[patch_id]] = 0

    return image, expl, spatial_expl, label

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
  dataset = CUB_dataset(crop_images=True)
  image, expl, spatial_expl, label = dataset[1]
  print(expl)
  print(image.shape)
  print(label)
  for i, r in enumerate(spatial_expl):
    if torch.isnan(r).sum() == 0:
      print(i, r.sum())