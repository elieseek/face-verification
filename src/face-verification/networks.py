import torch
import torch.nn as nn
import dataset

# Configs
from config import config

class FaceEmbedder(nn.Module):
  def __init__(self):
    super(FaceEmbedder, self).__init__()
    in_chnl = config.in_chnl
    out_chnl = config.out_chnl
    hidden_size = config.hidden_size
    bias = config.bias
    embedding_dimension = config.embedding_dimension
    self.final_state_size = int(out_chnl*16*(config.img_dim/2**5)**2)
    self.conv_net = nn.Sequential(
      # Input: in_chnl * 3 * 256 * 256
      nn.Conv2d(in_chnl, out_chnl, kernel_size=5, padding=2, bias=bias),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(),
      # State size: out_chnl * 128 * 128
      nn.Conv2d(out_chnl, out_chnl*2, kernel_size=5, padding=2, bias=bias),
      nn.MaxPool2d(2, stride=2),
      nn.ReLU(),
      # State size: (out_chnl*2) * 64 * 64
      nn.Conv2d(out_chnl*2, out_chnl*4, kernel_size=3, padding=1, bias=bias),
      nn.MaxPool2d(2, stride=2),
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
    return output

  def embed(self, x):
    return self.forward(x.unsqueeze(0))

class GE2ELoss(nn.Module):
  def __init__(self, device):
    super(GE2ELoss, self).__init__()
    self.device = device
    self.w = nn.Parameter(torch.Tenor(10.0).to(device), requires_grad=True)
    self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
  
  def forward(self, embeddings):
    torch.clamp(self.w, 1e-6) # sets minimum on w
    centroids = utils.compute_centroids(embeddings)
    cos_sim = utils.get_cos_sim(embeddings, centroids)
    similarity_mat = self.w*cos_sim.to(device) + self.b
    return utils.calc_loss(similarity_mat)

# Testing
if __name__ == '__main__':
  import time
  import utils
  from torch.utils.data import DataLoader
  device = torch.device(config.device)
  data = dataset.CelebADataset()
  loader = DataLoader(data, batch_size=config.train_classes)
  for image_batch in loader:
    start = time.time()
    image_batch = image_batch.to(device)
    image_batch = torch.reshape(image_batch, (10*64, image_batch.size(2), image_batch.size(3), image_batch.size(4)))
    net = FaceEmbedder().to(device)
    embeds = net.forward(image_batch)
    print(embeds.shape)
    embeds2 = embeds.view(64,10,256)
    centroids = utils.compute_centroids(embeds2).view(64, 256)
    print("centroids: {}".format(centroids.shape))
    print("similarity: {}".format(utils.get_cos_sim_matrix(embeds2, centroids).shape))
    print(utils.get_cos_sim_matrix(embeds2, centroids))
    print("{:0.2f} seconds".format(time.time()-start))
    break
  