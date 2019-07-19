import os

import torch
from torch.utils.data import DataLoader

# locals
from config import config as cfg
from dataset import CelebADataset
from networks import ConvEmbedder, GE2ELoss
import utils

def train():
  device = torch.device(cfg.device)

  dataset = CelebADataset()
  data_loader = DataLoader(dataset, batch_size=cfg.train_classes, shuffle=True, 
                            num_workers=cfg.num_workers, drop_last=True,
                            pin_memory=True)
  
  embedder_net = ConvEmbedder().to(device, non_blocking=True)
  ge2e_net = GE2ELoss(device)
  
  if cfg.resume_training == True:
    embedder_net.load_state_dict(torch.load(cfg.resume_model_path))
    ge2e_net.load_state_dict(torch.load(cfg.resume_ge2e_path))

  optimiser = torch.optim.Adam([
    {'params': embedder_net.parameters()},
    {'params': ge2e_net.parameters()}
  ], lr=cfg.learning_rate)

  loss_history = []
  no_improvement_count = 0
  embedder_net.train()
  print("GPU available: {}, Current device: {}".format(torch.cuda.is_available(), torch.cuda.get_device_name()))
  for epoch in range(cfg.n_epochs):
    print("Epoch {}/{}".format(epoch, cfg.n_epochs))
    total_loss = 0
    for i, image_batch in enumerate(data_loader):
      image_batch = image_batch.to(device, non_blocking=True)
      image_batch = torch.reshape(image_batch, (cfg.train_samples*cfg.train_classes, image_batch.size(2), image_batch.size(3), image_batch.size(4)))
      optimiser.zero_grad()

      embeddings = embedder_net(image_batch)
      embeddings = torch.reshape(image_batch, (cfg.train_classes, cfg.train_samples, -1))

      loss=ge2e_net(embeddings)
      loss.backward()
      optimiser.step()
      total_loss += loss
      if cfg.logging_rate != 0 and (i + 1) % cfg.logging_rate == 0:
        print("Epoch: {}, Batch: {}, Loss: {:.04f}, total loss: {:.04f}"
              .format(epoch, i, loss, total_loss)
              )

    loss_history.append(total_loss)
    if total_loss != min(loss_history):
      no_improvement_count += 1
    else: 
      no_improvement_count = 0

    # Stop training if total loss doesn't improve
    if no_improvement_count == cfg.early_stopping:
      print("Loss didn't improve for {} epochs, early stopping at epoch {}.".format(cfg.early_stopping, epoch))
      break

    
    print("Epoch: {}, total loss: {:.04f}".format(epoch, total_loss))
    if cfg.checkpoint_rate != 0 and (epoch + 1) % cfg.checkpoint_rate == 0:
      print("checkpoint saved!")
      torch.save(embedder_net.state_dict(), cfg.checkpoint_dir + 'model_chkpt_{}.pt'.format(epoch))
      torch.save(ge2e_net.state_dict(), cfg.checkpoint_dir + 'ge2e_chkpt_{}.pt'.format(epoch))

  print("Final loss: {:.04f}, Final total loss: {:.04f}".format(loss, total_loss))
  torch.save(embedder_net.state_dict(), cfg.model_dir + 'model_epoch_{}.pt'.format(epoch))

if __name__ == "__main__":
  train()