import cv2
import torch
from torch.functional import F

from config import config

def transform_fn(image):
  dim = config.img_dim
  h, w, c = image.shape
  crop = int((h-w)/2)
  if crop > 0:
    return cv2.resize(image[crop:-crop, :], (dim, dim))

  else:
    return cv2.resize(image[:, crop:-crop], (dim, dim))

def compute_centroids(embeddings):
  """
  Args:
    embeddings of shape (n_classes, n_samples, n_embeddings, len_embeddings)
  Returns:
    centroids of shape (n_classes, n_embeddings, len_embeddings)
  """
  centroids = []
  for sample in embeddings:
    centroids.append(torch.mean(sample, 0, keepdim=True))
  return torch.stack(centroids)

def get_cos_sim_matrix(embeddings, centroids):
  """
  Args:
    embeddings of shape (n_classes, n_samples, len_embeddings)
    centroids of shape (n_embeddings, len_embeddings)
  Returns:
    cos_sim_matrix of shape (n_classes, n_samples, n_centroids)
  """
  embeddings_flat = embeddings.view(-1, embeddings.size(2))
  centroids_norm = centroids / centroids.norm(dim=1)[:,None]
  embeddings_norm = embeddings_flat / embeddings_flat.norm(dim=1)[:, None]
  cos_sim_matrix = torch.mm(embeddings_norm, centroids_norm.transpose(0,1))
  return cos_sim_matrix.view(-1, embeddings.size(1), centroids.size(0))

