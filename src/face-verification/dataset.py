from __future__ import print_function
import re
import os
from os import path

from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CelebADataset(Dataset):
  """CelebA Dataset"""
  def __init__(self, image_dir, label_file, transform=None):
    self.image_dir = image_dir
    self.image_names = [path.join(image_dir, f) for f in os.listdir(image_dir) if path.isfile(path.join(image_dir, f))]
    with open(label_file, 'r') as label_:
      label_txt = label_.read()
      self.labels = np.array(re.findall(r'(?<=.jpg )\d+(?=\n)', label_txt))
    assert(len(self.image_names) == len(self.labels))

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    selected_label = self.labels[idx]
    img_name = self.image_names[idx]
    selected_image = io.imread(img_name)
    return np.array([selected_label, selected_image])

def restucture_celeba(image_dir, label_file):
  """Restructures image folder in-place to be used with torchvision.datasets.ImageFolder"""
  with open(label_file, 'r') as f:
    for line in f:
      line = line.strip()
      fname, label = tuple(line.split(' '))
      fpath_old = path.join(image_dir, fname)
      fpath_new = path.join(image_dir, label)
      if not os.path.exists(fpath_new): os.makedirs(fpath_new)
      fpath_new = path.join(fpath_new, fname)
      if not os.path.exists(fpath_new): os.rename(fpath_old, fpath_new)