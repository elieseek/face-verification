# file temporarily used to configure options
# will be replaced by yaml later
import cv2

class Config:
  def __init__(self):
    self.training = True
    self.train_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\train'
    self.val_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\val'
    self.test_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\test'
    self.label_dir = 'C:\cs_projects\\face-verification\data\identity_CelebA.txt'
    self.model_dir = 'C:\cs_projects\\face-verification\model\\'
    self.checkpoint_dir = 'C:\cs_projects\\face-verification\model\checkpoints\\'

    self.train_samples = 10
    self.test_samples = 10
    self.train_classes = 64
    self.test_classes = 64

    self.resume_training = False
    self.resume_model_path = 'C:\cs_projects\\face-verification\model\model.pt'
    self.resume_ge2e_path = 'C:\cs_projects\\face-verification\model\ge2e.pt'
    self.checkpoint_rate = 5
    self.num_workers = 16
    self.learning_rate = 1e-3
    self.n_epochs = 100
    self.logging_rate = 0
    self.early_stopping = 10 # stop training if model doesn't improve, -1 for no early_stopping

    self.img_dim = 64

    # ConvEmbedder Hparams
    self.in_chnl = 3
    self.out_chnl = 64
    self.bias = False
    self.hidden_size = 256
    self.embedding_dimension = 256

    self.device = 'cuda'

config = Config()