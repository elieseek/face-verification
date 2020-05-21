import numpy

from . import faceverification
from faceverification.statistics import calc_pixel_stats

if __name__ =="__main__":
  mean, sd = calc_pixel_stats()
  mean, sd = mean.numpy(), sd.numpy()
  mean = np.moveaxis(mean, 0, -1) # flip shape to channels-last so compatible with CV2
  sd = np.moveaxis(sd, 0, -1)
  print("Mean and SD calculated. Beginning training.")
  train(mean, sd)