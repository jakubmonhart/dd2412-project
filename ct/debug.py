from data.cub import CUB
import os

path = os.path.join('../','data', 'cub', 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt')


with open(path, 'r') as file:
  i = 0
  for line in file:
    if len(line.split()) != 5:
      print(i)
    i += 1