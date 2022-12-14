import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils import data

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms

import pytorch_lightning as pl

    
class aPY_torchvision(VisionDataset):
  attributes_url = 'http://vision.cs.uiuc.edu/attributes/attribute_data.tar.gz'
  pascal_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar'
  yahoo_url = 'http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz'

  def __init__(self, download=False, train=True, yahoo=False, transform=None, data_path='data'):
    '''
    Args:
      yahoo (bool): include yahoo test dataset (that corresponds to the aPY dataset) if True, otherwise test only on aPascal test set. 
    '''

    self.root = os.path.join(data_path, 'apy')
    self.raw_folder = os.path.join(self.root, 'raw')
    self.train = train
    self.yahoo = yahoo
    self.transform = transform

    if download:
      self.download()
    else:
      # Check if dataset already exits
      if not os.path.exists(self.root):
        print('Dataset files not found. Run aPY with download=True.')
        return

    train_df = pd.read_csv(                                                      # Train contains just aPascal train data
      os.path.join(self.root, 'apascal_train.txt'), sep=' ', header=None,
      names=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] + list(range(64)))

    apascal_test_df = pd.read_csv(
      os.path.join(self.root, 'apascal_test.txt'), sep=' ', header=None,
      names=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] + list(range(64)))

    ayahoo_test_df = pd.read_csv(
      os.path.join(self.root, 'ayahoo_test.txt'), sep=' ', header=None,
      names=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] + list(range(64)))

    # Prune objects with no assigned attribution
    attributes = train_df[range(64)].to_numpy()
    train_df = train_df[attributes.sum(-1) != 0] 

    attributes = apascal_test_df[range(64)].to_numpy()
    apascal_test_df = apascal_test_df[attributes.sum(-1) != 0]
    
    attributes = ayahoo_test_df[range(64)].to_numpy()
    ayahoo_test_df = ayahoo_test_df[attributes.sum(-1) != 0]

    # Delete objects with bbox of area 0 - there is apparently just one object with area 0. (in yahoo test dataset)
    train_df = train_df[(train_df.xmin != train_df.xmax) & (train_df.ymin != train_df.ymax)]
    apascal_test_df = apascal_test_df[(apascal_test_df.xmin != apascal_test_df.xmax) & (apascal_test_df.ymin != apascal_test_df.ymax)]
    ayahoo_test_df = ayahoo_test_df[(ayahoo_test_df.xmin != ayahoo_test_df.xmax) & (ayahoo_test_df.ymin != ayahoo_test_df.ymax)]

    self.train_df = train_df.reset_index()
    self.test_df = pd.concat([apascal_test_df, ayahoo_test_df], ignore_index=True)

    # Create folder with cropped train and test objects
    if not os.path.exists(os.path.join(self.root, 'train')):
      self.create_objects(train=True)
    
    if not os.path.exists(os.path.join(self.root, 'test')):
      self.create_objects(train=False)

    # Use only apascal data
    if self.yahoo:
      self.test_df = ayahoo_test_df.reset_index()
      self.yahoo_index_offset = len(apascal_test_df)
      self.test_df.index = self.test_df.index + self.yahoo_index_offset
    else:  
      self.test_df = apascal_test_df.reset_index()

    # Train or test
    if self.train:
      self.im_folder = os.path.join(self.root, 'train')
      self.df = self.train_df
    else:
      self.im_folder = os.path.join(self.root, 'test')
      self.df = self.test_df

    # Load class2id
    classes = pd.read_csv(os.path.join(self.root, 'class_names.txt'), header=None)[0].values
    class2id = dict()
    for i, c in enumerate(classes):
      class2id[c] = i
    self.class2id = class2id
    
    # Create id2class
    self.id2class = [None]*len(class2id)
    for k in class2id.keys():
      self.id2class[class2id[k]] = k

    # Load id2attribute
    id2attribute = pd.read_csv(os.path.join(self.root, 'attribute_names.txt'), header=None)
    self.id2attribute = id2attribute.loc[:, 0].values

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    
    item_data = self.df.iloc[index]

    # Load image
    with open(os.path.join(self.im_folder, '{:d}.jpg'.format(self.df.index[index])), "rb") as f:
      img = Image.open(f)
      img = img.convert("RGB")

    if self.transform is not None:
      img = self.transform(img)

    # Load target
    target_class = self.class2id[item_data['class']]
    target_class = torch.tensor(target_class)

    # Load atributes
    target_concept = torch.tensor(item_data[range(64)].to_numpy(dtype='float32'))

    return img, (target_class, target_concept)

  def create_objects(self, train=True):
    if train:
      df = self.train_df
      folder = 'train'
    else:
      df = self.test_df
      folder = 'test'

    target_path = os.path.join(self.root, folder)
    if not os.path.exists(target_path):
      os.makedirs(target_path) 

    bbox = ['xmin', 'ymin', 'xmax', 'ymax']
    for i in tqdm(range(len(df))):
      im_data = df.iloc[i]
    
      if '2008_' in im_data.filename:
        im_folder = 'apascal'
      else:
        im_folder = 'ayahoo'
        
      with Image.open(os.path.join(
        self.root, im_folder, im_data.filename)) as im:
        im = im.crop(box=im_data[bbox])
        try:
           im.save(os.path.join(target_path, '{:d}.jpg'.format(i)))
        except:
          breakpoint()

  def download(self):

    #Check if dataset already exists
    if os.path.exists(self.root):
      return

    # Download and extract attribute info
    download_and_extract_archive(url=self.attributes_url, download_root=self.raw_folder, extract_root=self.raw_folder)

    # Move files
    shutil.move(os.path.join(self.raw_folder, 'attribute_data', 'apascal_train.txt'), os.path.join(self.root, 'apascal_train.txt'))
    shutil.move(os.path.join(self.raw_folder, 'attribute_data', 'apascal_test.txt'), os.path.join(self.root, 'apascal_test.txt'))
    shutil.move(os.path.join(self.raw_folder, 'attribute_data', 'ayahoo_test.txt'), os.path.join(self.root, 'ayahoo_test.txt'))
    shutil.move(os.path.join(self.raw_folder, 'attribute_data', 'class_names.txt'), os.path.join(self.root, 'class_names.txt'))
    shutil.move(os.path.join(self.raw_folder, 'attribute_data', 'attribute_names.txt'), os.path.join(self.root, 'attribute_names.txt'))

    # Download aPascal images
    download_and_extract_archive(url=self.pascal_url, download_root=self.raw_folder, extract_root=self.raw_folder)

    # Move files
    os.rename(
      os.path.join(self.raw_folder, 'VOCdevkit', 'VOC2008', 'JPEGImages'),
      os.path.join(self.raw_folder, 'VOCdevkit', 'VOC2008', 'apascal'))
    shutil.move(os.path.join(self.raw_folder, 'VOCdevkit', 'VOC2008', 'apascal'), self.root)
    
    # Download aYahoo images
    download_and_extract_archive(url=self.yahoo_url, download_root=self.raw_folder, extract_root=self.raw_folder)

    # Move files
    os.rename(
      os.path.join(self.raw_folder, 'ayahoo_test_images'),
      os.path.join(self.raw_folder, 'ayahoo'))
    shutil.move(os.path.join(self.raw_folder, 'ayahoo'), self.root)

    # Delete raw folder
    shutil.rmtree(self.raw_folder)


class aPY(pl.LightningDataModule):

  """
  ResNet (backbone used in CCT Tokenizer) resizes images to 232.

  TODO
    - image transformation: augmentation
    - Normalize with imagenet statistics? (we are using renset pretreined on imagenet in the backbone)
    - Fix seed for random subset selection (numpy seed for sklearn train_test_split)
    - Stratify train/val split?
  """

  def __init__(self, batch_size, image_size=224, yahoo=False, seed=42, data_path='data'):
    super().__init__()
    self.data_path = data_path
    self.batch_size = batch_size
    self.image_size = image_size
    self.yahoo = yahoo
    self.seed = seed
    
  def prepare_data(self):
    aPY_torchvision(download=True, data_path=self.data_path)
  
  def setup(self, stage=None):
    # For reproducibility
    self.generator = torch.Generator().manual_seed(self.seed)
    
    train_transform = transforms.Compose([
      transforms.Resize(size=(self.image_size, self.image_size)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=15),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
      transforms.Resize(size=(self.image_size, self.image_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_full = aPY_torchvision(train=True, transform=train_transform, data_path=self.data_path)
    val_full = aPY_torchvision(train=True, transform=val_transform, data_path=self.data_path)
    
    train_len = round(0.9*len(train_full))
    val_len = len(train_full) - train_len

    self.train, val = data.random_split(
      train_full, lengths=[train_len, val_len], generator=self.generator)
    
    # Create val only to get the indices easily, now create actual val with val_transform
    self.val = data.Subset(dataset=val_full, indices=val.indices)
    
    self.test = aPY_torchvision(
      train=False, transform=val_transform, yahoo=self.yahoo, data_path=self.data_path)

  def train_dataloader(self):
    return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, generator=self.generator)

  def val_dataloader(self):
    return data.DataLoader(self.val, batch_size=self.batch_size)

  def test_dataloader(self, shuffle=False):
    return data.DataLoader(self.test, batch_size=self.batch_size, shuffle=shuffle)


if __name__ == '__main__':

  apy = aPY(8)
  apy.prepare_data()
  apy.setup()

  train = apy.train_dataloader()
  for img, (target_class, target_concept) in train:
    break

  breakpoint()