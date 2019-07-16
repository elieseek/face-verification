import torch
import torch.nn as nn

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

    self.conv_net = nn.Sequential(
      # Input: in_chnl * 3 * 218 * 178
      nn.Conv2d(in_chnl, out_chnl, kernel_size=5, bias=bias),
      nn.MaxPool2d(2, stride=2),
      # State size: out_chnl * 107 * 87
      nn.Conv2d(out_chnl, out_chnl*2, kernel_size=5, padding=1, bias=bias),
      nn.MaxPool2d(2, stride=2),
      # State size: (out_chnl*2) * 53 * 43
      nn.Conv2d(out_chnl*2, out_chnl*4, kernel_size=3, bias=bias),
      nn.MaxPool2d(2, stride=2),
      # State size: (out_chnl*2) * 26 * 22
      nn.Conv2d(out_chnl*4, out_chnl*8, kernel_size=3, padding=1,bias=bias),
      nn.MaxPool2d(2, stride=2),
      # State size: (out_chnl*2) * 13 * 11
      nn.Conv2d(out_chnl*8, out_chnl*16, kernel_size=3,bias=bias),
      nn.MaxPool2d(2, stride=2),
      # State size: (out_chnl*2) * 6 * 5
    )

    self.fc_net = nn.Sequential(
      # Input is flattened final convnet state size
      nn.Linear(out_chnl*16 * 6 * 5, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, embedding_dimension)
    )

    def forward(self, x):
      output = self.conv_net(x)
      output = output.view(output.size()[0], -1)
      output = self.fc_net(X)
      return output

    def embed(self, x):
      return forward(x)