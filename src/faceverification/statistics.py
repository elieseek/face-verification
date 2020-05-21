import os
from os import path
import torch
from torch.utils.data import Dataset, DataLoader

# locals
from .dataset import CelebADataset
from .evaluate import calc_embedding
from .config import config as cfg

def calc_pixel_stats():
  """
  Calculates the mean and SD per pixel per channel for the train dataset
  """
  dataset = CelebADataset()
  data_loader = DataLoader(dataset, batch_size=cfg.train_classes, 
                            num_workers=cfg.num_workers, drop_last=False,
                            pin_memory=True)

  sum_tensor = torch.zeros(cfg.in_chnl, cfg.img_dim, cfg.img_dim)
  n_imgs = 0
  for batch in data_loader:
    batch = batch.view(-1, batch.size(2), batch.size(3),batch.size(4)) # flatten into array of images
    n_imgs += batch.size(0)
    sum_tensor += batch.sum(0)
  mean_tensor = sum_tensor.div(n_imgs)

  sum_tensor = torch.zeros(cfg.in_chnl, cfg.img_dim, cfg.img_dim)
  for batch in data_loader:
    batch = batch.view(-1, batch.size(2), batch.size(3),batch.size(4)) # flatten into array of images
    sum_tensor += ((batch-mean_tensor)**2).sum(0)
  sd_tensor = (sum_tensor.div(n_imgs-1))**0.5

  return mean_tensor, sd_tensor
