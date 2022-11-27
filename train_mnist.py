import torch
from torch import nn, optim
from torch.utils import data
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from ct.model.cct import CCT_torch


class MNIST(pl.LightningDataModule):

  """
  TODO
    - image transoformation: augmentation
    - use AdamW
  """

  def __init__(self, batch_size):
    super().__init__()
    self.root = 'data'
    self.batch_size = batch_size

  def prepare_data(self):
    torchvision.datasets.MNIST(root=self.root, train=True, download=True)
    torchvision.datasets.MNIST(root=self.root, train=False, download=True)
  
  def setup(self, stage=None):
    transform = transforms.Compose([
      transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])

    target_transform = transforms.Lambda(lambda y: torch.zeros(
      10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    train_full = torchvision.datasets.MNIST(root=self.root, train=True, transform=transform, target_transform=target_transform)

    self.train = data.Subset(dataset=train_full, indices=range(55000))
    self.val = data.Subset(dataset=train_full, indices=range(55000, len(train_full)))
    self.test = torchvision.datasets.MNIST(root=self.root, train=False, transform=transform, target_transform=target_transform)

  def train_dataloader(self):
    return data.DataLoader(self.train, batch_size=self.batch_size)

  def val_dataloader(self):
    return data.DataLoader(self.val, batch_size=self.batch_size)

  def test_dataloader(self):
    return data.DataLoader(self.test, batch_size=self.batch_size)


class CCT(pl.LightningModule):
  def __init__(self):
    super().__init__()
    
    self.cct = CCT_torch(
      img_size=28, n_input_channels=1, num_layers=2, num_heads=2,
      embedding_dim=128, num_classes=10, mlp_ratio=1)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.cct(x)
    loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y)

    y = torch.argmax(y, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.mean((y == y_pred)*1.)
    
    self.log_dict({'acc': acc}, prog_bar=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.cct(x)
    loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y)
    
    y = torch.argmax(y, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.mean((y == y_pred)*1.)

    return loss, acc

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_pred = self.cct(x)
    loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y)
    return loss

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer


if __name__ == "__main__":
  mnist = MNIST(batch_size=8)
  mnist.setup()
  cct = CCT()
  trainer = pl.Trainer(max_epochs=1)
  trainer.fit(model=cct, datamodule=mnist)
