# file temporarily used to configure options
# will be replaced by yaml later
import cv2

class Config:
  def __init__(self):
    self.dataset_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba'
    self.train_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\train'
    self.val_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\val'
    self.test_dir = 'C:\cs_projects\\face-verification\data\img_align_celeba\\test'
    self.label_dir = 'C:\cs_projects\\face-verification\data\identity_CelebA.txt'
    self.partition_dir = 'C:\cs_projects\\face-verification\data\list_eval_partition.txt'

    self.model_dir = 'C:\cs_projects\\face-verification\model\\'
    self.checkpoint_dir = 'C:\cs_projects\\face-verification\model\checkpoints\\'

    self.train_samples = 10
    self.test_samples = 10
    self.train_classes = 64
    self.test_classes = 64

    self.resume_training = True
    self.resume_model_path = 'C:\cs_projects\\face-verification\model\model.pt'
    self.resume_ge2e_path = 'C:\cs_projects\\face-verification\model\ge2e.pt'
    self.num_workers = 4
    self.learning_rate = 0.01
    self.n_epochs = 100
    self.logging_rate = 0
    self.early_stopping = 5 # stop training if model doesn't improve, -1 for no early_stopping
    self.checkpoint_rate = 0
    
    self.img_dim = 64

    # ConvEmbedder Hparams
    self.in_chnl = 3
    self.out_chnl = 8
    self.bias = True
    self.hidden_size = 128
    self.embedding_dimension = 64

    self.device = 'cuda'
    self.n_gpu = 1

config = Config()