import torch
import torch.nn as nn
import dataset

# local
from config import config as cfg
import utils

class ConvEmbedder(nn.Module):
  def __init__(self):
    super(ConvEmbedder, self).__init__()
    in_chnl = cfg.in_chnl
    out_chnl = cfg.out_chnl
    hidden_size = cfg.hidden_size
    bias = cfg.bias
    embedding_dimension = cfg.embedding_dimension
    self.final_state_size = int(out_chnl*16*(cfg.img_dim/2**3)**2)
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
    torch.clamp(self.w, 1e-6) # sets minimum on w
    centroids = utils.compute_centroids(embeddings)
    cos_sim = utils.get_cos_sim_matrix(embeddings, centroids)
    similarity_mat = self.w*cos_sim.to(self.device) + self.b
    return self.loss_fn(similarity_mat)

# Testing
if __name__ == '__main__':
  import time
  from torch.utils.data import DataLoader
  device = torch.device(cfg.device)
  data = dataset.CelebADataset()
  loader = DataLoader(data, batch_size=cfg.train_classes)
  for image_batch in loader:
    start = time.time()
    image_batch = image_batch.to(device, non_blocking=True)
    image_batch = torch.reshape(image_batch, (10*cfg.train_classes, image_batch.size(2), image_batch.size(3), image_batch.size(4)))
    net = ConvEmbedder().to(device, non_blocking=True)
    loss = GE2ELoss(device)
    print("n_params: {}".format(sum([p.numel() for p in net.parameters() if p.requires_grad])))
    embeds = net(image_batch)
    print(embeds.shape)
    embeds2 = embeds.view(cfg.train_classes,10,256)
    print(loss.forward(embeds.view(cfg.train_classes,10,256)))
    print("{:0.2f} seconds".format(time.time()-start))
    break