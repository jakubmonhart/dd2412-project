from data.cub import CUB
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

dataset_folder = os.path.join('../', 'data', 'cub', 'CUB_200_2011')

images = pd.read_csv(os.path.join(dataset_folder, 'images.txt'), sep=' ',
                             names=['image_id', 'path'])

for idx, row in images.iterrows():
  image_path = os.path.join(dataset_folder, 'images', row['path'])
  image = np.array(Image.open(image_path))
  if len(image.shape) != 3:
    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
    print(image.shape)
    print(image_path)
    plt.imshow(image)
    plt.savefig('temp.png')
    plt.show()
    input('----------')