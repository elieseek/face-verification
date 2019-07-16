from __future__ import print_function
import re
import os
from os import path
from shutil import rmtree
from random import shuffle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Configs
from config import config

class CelebADataset(Dataset):
  """
  CelebA Dataset
  config.py options:
    transform: a transformation function for each image
    training: bool indicating whether to train or test
    train/test_dir: path to dataset folder
    train/test_samples: number of samples taken from each label
    train/test_classes: number of labels to take for GE2E loss
  """
  def __init__(self):
    self.transform = config.transform
    self.in_chnl = config.in_chnl
    if config.training == True:
      self.path = config.train_dir
      self.n_samples = config.train_samples
    else:
      self.path = config.test_dir
      self.n_samples = config.test_samples
    
    self.labels = os.listdir(self.path)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    label_folder = path.join(self.path, label)
    images = os.listdir(label_folder)
    shuffle(images)
    images = [path.join(label_folder, x) for x in images]
    images = images[:self.n_samples]
    image_arrays = [self.transform(cv2.imread(img)) for img in images]
    return np.array(image_arrays).reshape(-1, self.in_chnl, 218, 178)

def restucture_celeba(image_dir, label_file, eval_file, rm_small_classes=False):
  """
  Restructures image folder in-place for train/test/split with label-folder structuring
  """
  part_dict = read_eval_file(eval_file)
  with open(label_file, 'r') as f:
    for line in f:
      line = line.strip()
      fname, label = tuple(line.split(' '))
      partition = part_dict[fname]

      if partition == '0':
        fpath_new = config.train_dir
      elif partition == '1':
        fpath_new = config.val_dir
      else:
        fpath_new = config.test_dir

      fpath_old = path.join(image_dir, fname)
      fpath_new = path.join(fpath_new, label)

      if not os.path.exists(fpath_new): os.makedirs(fpath_new)
      fpath_new = path.join(fpath_new, fname)
      if not os.path.exists(fpath_new): os.rename(fpath_old, fpath_new)
  
  if rm_small_classes:
    remove_small_classes(config.train_dir)

def read_eval_file(eval_file):
  part_dict = {}
  with open(eval_file, 'r') as f:
    for line in f:
      fname, partition = tuple(line.strip().split(' '))
      part_dict[fname] = partition
  return part_dict

def remove_small_classes(image_dir):
  """
  Removes classes that have less than 10 samples (n=1458 for train)
  Required for GE2E batch training with maximum 10 samples per class
  """
  classes = os.listdir(image_dir)
  count = 0
  for folder in classes:
    path_to_folder = path.join(image_dir, folder)
    if len(os.listdir(path_to_folder)) < 10:
      rmtree(path_to_folder)
      count +=1

  return count

if __name__ == '__main__':
  dataset = CelebADataset()
  import time
  start = time.time()
  [dataset[i] for i in range(64)]
  print("{:0.2f} seconds".format(time.time()-start))