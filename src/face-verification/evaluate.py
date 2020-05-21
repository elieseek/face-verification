import torch
import numpy as np
import cv2
import os
from os import path
import pickle

# locals
from utils import transform_fn
from config import config as cfg

def calc_embedding(model, image,
                    mean=np.zeros((cfg.img_dim, cfg.img_dim, cfg.in_chnl)), 
                    sd=np.ones((cfg.img_dim, cfg.img_dim, cfg.in_chnl)),
                    channels_last=True
                  ):
  image = transform_fn(image, mean, sd)
  if channels_last==True:
    image = np.moveaxis(image, -1, 0)
  with torch.no_grad():
    embedding = model(torch.from_numpy(image).unsqueeze(0).float()).numpy()
    return np.squeeze(embedding)

def calc_all_embeddings(data_path, model, model_path,
                        mean=np.zeros((cfg.img_dim, cfg.img_dim, cfg.in_chnl)), 
                        sd=np.ones((cfg.img_dim, cfg.img_dim, cfg.in_chnl))
                        ):
  """
  returns dicitonary of embeddings for labels/images in train/test/val split
  """
  model.load_state_dict(torch.load(model_path))
  model.eval()
  all_embeds = dict()
  labels = os.listdir(data_path)
  for label in labels:
    all_embeds[label] = dict()
    label_path = path.join(data_path, label)
    images = os.listdir(label_path)
    for image in images:
      image_path = path.join(label_path, image)
      image_data = cv2.imread(image_path)
      all_embeds[label][image] = calc_embedding(model, image_data, mean, sd)

  return all_embeds

def cos_similarity(v_1, v_2):
  nv_1 = np.linalg.norm(v_1)
  nv_2 = np.linalg.norm(v_2)
  return np.true_divide(np.dot(v_1, v_2), nv_1*nv_2)

def calc_similarity_dict(embedding_dict, embedding):
  similarity_dict = dict()
  labels = embedding_dict.keys()
  for label in labels:
    similarity_dict[label] = dict()
    images = embedding_dict[label].keys()
    for image in images:
      similarity_dict[label][image] = cos_similarity(embedding, embedding_dict[label][image])

  return similarity_dict

def create_similarity_files(embedding_dict, similarity_dir):
  labels = embedding_dict.keys()
  for label in labels:
    label_dict = dict()
    images = embedding_dict[label].keys()
    for image in images:
      embedding = embedding_dict[label][image]
      label_dict[image] = calc_similarity_dict(embedding_dict, embedding)
    save_path = path.join(similarity_dir, label+'.pkl')
    with open(save_path, 'wb') as f:
      pickle.dump(label_dict, f)


# def calc_far(threshold)