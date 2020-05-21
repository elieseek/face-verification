import torch
import torch.nn as nn

# local
from . import dataset
from .config import config as cfg
from . import utils

class ConvEmbedder(nn.Module):
  def __init__(self):
    super(ConvEmbedder, self).__init__()
    in_chnl = cfg.in_chnl
    out_chnl = cfg.out_chnl
    hidden_size = cfg.hidden_size
    bias = cfg.bias
    embedding_dimension = cfg.embedding_dimension
    self.final_state_size = int(out_chnl*4*(cfg.img_dim/2**2)**2)
    self.conv_net = nn.Sequential(
      # Input: batch * in_chnl * 64 * 64
      nn.Conv2d(in_chnl, out_chnl, kernel_size=5, padding=2, bias=bias),
      nn.ReLU(),
      # State size: out_chnl * 64 * 64
      nn.Conv2d(out_chnl, out_chnl*2, kernel_size=5, padding=2, bias=bias),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(),
      # State size: (out_chnl*2) * 32 * 32
      nn.Conv2d(out_chnl*2, out_chnl*4, kernel_size=3, padding=1, bias=bias),
      nn.ReLU(),
      # State size: (out_chnl*4) * 32 * 32
      nn.Conv2d(out_chnl*4, out_chnl*8, kernel_size=3, padding=1,bias=bias),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(),
      # State size: (out_chnl*8) * 16 * 16
      nn.Conv2d(out_chnl*8, out_chnl*16, kernel_size=3, padding=1, bias=bias),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(),
      # State size: (out_chnl*16) * 8 * 8
    )

    self.fc_net = nn.Sequential(
      # Input is flattened final convnet state size
      nn.Linear(self.final_state_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, embedding_dimension)
    )

  def forward(self, x):
    output = self.conv_net(x)
    output = output.view(output.size()[0], -1)
    output = self.fc_net(output)
    output = nn.functional.normalize(output, dim=-1, p=2)
    return output

  def embed(self, x):
    return self.forward(x.unsqueeze(0))

class GE2ELoss(nn.Module):
  def __init__(self, device):
    super(GE2ELoss, self).__init__()
    self.device = device
    self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
    self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
    self.loss_fn = utils.calc_softmax_loss
  
  def forward(self, embeddings):
    torch.clamp(self.w, min=1e-6) # sets minimum on w
    centroids = utils.compute_centroids(embeddings)
    cos_sim = utils.get_cos_sim_matrix(embeddings, centroids)
    similarity_mat = self.w*cos_sim.to(self.device) + self.b
    return self.loss_fn(similarity_mat)

class CombinedModel(nn.Module):
  def __init__(self, model_a, model_b):
    super(CombinedModel, self).__init__()
    self.model_a = model_a
    self.model_b = model_b
    self.batch_size = cfg.train_classes
    self.n_samples = cfg.train_samples
  
  def forward(self, x):
    output = self.model_a(x)
    output = torch.reshape(output, (self.batch_size, self.n_samples, -1))
    output = self.model_b(output)
    return output

# Testing
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import cv2
  import time
  import numpy as np
  from torch.utils.data import DataLoader
  mean, sd = dataset.calc_pixel_stats()
  mean, sd = mean.numpy(), sd.numpy()
  mean = np.moveaxis(mean, 0, -1) # flip shape to channels-last so compatible with CV2
  sd = np.moveaxis(sd, 0, -1)
  device = torch.device(cfg.device)
  data = dataset.CelebADataset(False, mean, sd)
  loader = DataLoader(data, batch_size=cfg.train_classes, shuffle=True)
  net = ConvEmbedder().to(device, non_blocking=True)
  ge2e = GE2ELoss(device)
  net.load_state_dict(torch.load(cfg.resume_model_path))
  ge2e.load_state_dict(torch.load(cfg.resume_ge2e_path))
  prev_emb = torch.Tensor()
  i = 1
  for image_batch in loader:
    print(image_batch.shape)
    test = np.moveaxis(image_batch.numpy()[0].astype(float).astype(np.uint8), 1,-1)
    print(test.shape)
    cv2.imshow('name',(test[0]))
    cv2.waitKey(0)
    start = time.time()
    print(image_batch.shape)
    image_batch = image_batch.float().to(device, non_blocking=True)
    image_batch = torch.reshape(image_batch, (-1, image_batch.size(2), image_batch.size(3), image_batch.size(4)))
    print(image_batch.shape)
    print("n_params: {}".format(sum([p.numel() for p in net.parameters() if p.requires_grad])))
    embeds = net(image_batch)
    embeds = torch.reshape(embeds, (cfg.train_classes, cfg.train_samples, -1))
    loss = ge2e(embeds)  
    loss.backward()
    if i != 1:
      mix = torch.stack([prev_emb,embeds[0]])
      print(mix.shape)
      c = utils.compute_centroids(mix)
      matrix = utils.get_cos_sim_matrix(mix, c)
      print(matrix)
      print(loss)
    i += 1
    prev_emb = embeds[0]
    # print(embeds[:2])
    embeds2 = embeds.view(cfg.train_classes,cfg.train_samples,cfg.embedding_dimension)
    # print(ge2e.forward(embeds2))
    print("{:0.2f} seconds".format(time.time()-start))
