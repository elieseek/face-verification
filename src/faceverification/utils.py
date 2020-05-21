import cv2
import numpy as np
import torch
from torch.functional import F

# locals
from config import config as cfg

def transform_fn(image, 
                  mean=np.zeros((cfg.img_dim, cfg.img_dim, cfg.in_chnl)), 
                  sd=np.ones((cfg.img_dim, cfg.img_dim, cfg.in_chnl))
                ):
  """
  Crop and z-score normalise images
  default mean and sd make no changes to image
  """
  dim = cfg.img_dim
  h, w, c = image.shape
  crop = int((h-w)/2)
  if crop > 0:
     image = cv2.resize(image[crop:-crop, :], (dim, dim))
  else:
    image = cv2.resize(image[:, crop:-crop], (dim, dim))

  return np.true_divide(image-mean,sd)

def compute_centroids(embeddings):
  """
  Args:
    embeddings of shape (n_classes, n_samples, n_embeddings)
  Returns:
    centroids of shape (n_classes, n_embeddings)
  """
  centroids = []
  for sample in embeddings:
    centroids.append(torch.mean(sample, 0, keepdim=False))

  return torch.stack(centroids)

def get_cos_sim_matrix(embeddings, centroids):
  """
  Args:
    embeddings of shape (n_classes, n_samples, n_embeddings)
    centroids of shape (n_classes, n_embeddings)
  Returns:
    cos_sim_matrix of shape (n_classes, n_samples, n_centroids)
  """
  embeddings_flat = embeddings.view(-1, embeddings.size(2))
  centroids_norm = centroids / centroids.norm(dim=1)[:,None]
  embeddings_norm = embeddings_flat / embeddings_flat.norm(dim=1)[:, None]
  cos_sim_matrix = torch.mm(embeddings_norm, centroids_norm.transpose(0,1))

  return cos_sim_matrix.view(-1, embeddings.size(1), centroids.size(0))

def calc_softmax_loss(similarity_matrix):
  """
  Args:
    Similarity matrix of shape (n_classes, n_samples, n_centroids)
  Returns:
    Softmax loss over all embeddings
  """
  total_loss = 0
  # For each group of embeddings calculate softmax loss
  for i in range(similarity_matrix.size(0)):
    class_col = torch.neg(similarity_matrix[i].transpose(0,1)[i])
    embedding_softmax = torch.logsumexp(similarity_matrix[i], dim=1)
    total_loss += torch.sum(class_col + embedding_softmax)
  
  return total_loss