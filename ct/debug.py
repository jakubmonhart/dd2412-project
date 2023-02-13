from data.cub import CUB, CUB_dataset
import os

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

dataset_folder = os.path.join('../', 'data', 'cub', 'CUB_200_2011')


images = pd.read_csv(os.path.join(dataset_folder, 'images.txt'), sep=' ',
                             names=['image_id', 'path'])

dataset = CUB_dataset()
print(os.getcwd())
# assert False
array = [257, 953, 111, 325, 387, 415, 122, 137, 951, 713]
ats = dataset.attributes.copy()
ats.set_index(ats.attr_id, inplace=True)
# print(ats)
# assert False
for index in array:
  image_metadata = dataset.data.loc[index]
  
  image_attributes = dataset.image_attributes[dataset.image_attributes['image_id'] == image_metadata['image_id']]
  image = np.array(Image.open(os.path.join(dataset.image_folder, image_metadata['path']))) # Image is loaded as PIL
  image = image[int(image_metadata.y1):int(image_metadata.y1 + image_metadata.h), int(image_metadata.x1):int(image_metadata.x1 + image_metadata.w), :]
  label = image_metadata.target - 1
  plt.imshow(resize(image))
  plt.savefig(f'./temp/{index}.png')
  plt.close()
  print('------------------------')
  print(image_metadata)
  for _, row in image_attributes.iterrows():
      if row['is_present'] == 0:
        continue  
      print(ats.loc[row['attr_id'], 'def'])
