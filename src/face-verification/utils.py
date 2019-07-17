import cv2

from config import config

def transform_fn(image):
  dim = config.img_dim
  h, w, c = image.shape
  crop = int((h-w)/2)
  if crop > 0:
    return cv2.resize(image[crop:-crop, :], (dim, dim))
    #return image[crop:-crop, :]
  else:
    return cv2.resize(image[:, crop:-crop], (dim, dim))
    #return image[:, crop:-crop]