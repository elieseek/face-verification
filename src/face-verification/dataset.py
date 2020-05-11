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
from config import config as cfg
import utils

class CelebADataset(Dataset):
  """
  CelebA Dataset
  config.py options:
    transform: a transformation function for each image
    training: bool indicating whether to train or test
    train/test_dir: path to dataset folder
    train/test_samples: number of samples taken from each label
    train/test_classes: number of labels to take for GE2E loss
    image_dim: dimension to resize images into
  """
  def __init__(self):
    self.transform = utils.transform_fn
    self.in_chnl = cfg.in_chnl
    if cfg.training == True:
      self.path = cfg.train_dir
      self.n_samples = cfg.train_samples
    else:
      self.path = cfg.test_dir
      self.n_samples = cfg.test_samples
    
    self.labels = os.listdir(self.path)
    self.dim = cfg.img_dim
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    label_folder = path.join(self.path, label)
    images = os.listdir(label_folder)
    shuffle(images)
    images = [path.join(label_folder, x) for x in images]
    images = images[:self.n_samples]
    image_arrays = np.array([self.transform(cv2.imread(img)) for img in images])
    return torch.from_numpy(image_arrays).view(-1, self.in_chnl, self.dim, self.dim).float()

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
        fpath_new = cfg.train_dir
      elif partition == '1':
        fpath_new = cfg.val_dir
      else:
        fpath_new = cfg.test_dir

      fpath_old = path.join(image_dir, fname)
      fpath_new = path.join(fpath_new, label)

      if not os.path.exists(fpath_new): os.makedirs(fpath_new)
      fpath_new = path.join(fpath_new, fname)
      if not os.path.exists(fpath_new): os.rename(fpath_old, fpath_new)
  
  if rm_small_classes:
    remove_small_classes(cfg.train_dir)

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
    if len(os.listdir(path_to_folder)) < cfg.train_samples:
      rmtree(path_to_folder)
      count +=1

  return count

if __name__ == '__main__':
 image_dir = cfg.dataset_dir
 label_file = cfg.label_dir
 eval_file = cfg.partition_dir

 restucture_celeba(image_dir, label_file, eval_file, True)