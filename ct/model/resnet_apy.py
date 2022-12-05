import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torchvision

from .cct import CCT_ResNet_torch
from .ct import ConceptTransformer

import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)


class ResNet_aPY_torch(nn.Module):
  dim = 768
  n_classes = 20
  n_concepts = 64

  """
  TODO:
    - Backend CT uses ResNet50 as the tokenizer
  """

  def __init__(self):
    super().__init__(resnet='50')
    
    if resnet == '50':
      resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
      self.linear = nn.Linear(2048, self.n_classes)
    elif resnet == '34':
      resnet = torchvision.models.resnet34(weights='IMAGENET1K_V1')
      self.linear = nn.Linear(512, self.n_classes)

    layers = list(resnet.children())[:-1] # Include Adaptive average maxpool (AdaptiveAvgPool2d)  
    self.feature_extractor = nn.Sequential(*layers)    

  def forward(self, images):
    """
    inputs:
      x - batch images
    """

    out = self.feature_extractor(images)
    out = torch.flatten(out, 1)
    out = self.linear(out)

    return out


class ResNet_aPY(pl.LightningModule):
  n_classes = 20 # Consider only pascal data for now

  def __init__(self, args):
    super().__init__()

    self.args = args
  
    self.model = ResNet_aPY_torch()

    class_counts = [183, 161, 254, 228, 296, 73, 528, 187, 446, 103, 113, 244, 153, 149, 2488, 225, 123, 121, 90, 150]
    class_counts = torch.tensor(class_counts)
    class_weight = 1 / class_counts
    self.class_weight = class_weight / class_weight.sum()

    self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes)
    self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes)
    self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes)

  def training_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class = self.model(image)

    # Loss
    loss = self.loss_fn(target_class, pred_class)

    # Accuracy
    self.train_accuracy(pred_class, target_class.int())
    self.log('train_acc', self.train_accuracy, prog_bar=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class = self.model(image)
    
    # Loss
    loss = self.loss_fn(target_class, pred_class)

    # Accuracy
    self.val_accuracy(pred_class, target_class.int())
    self.log('val_acc', self.val_accuracy, prog_bar=True)
    self.log('val_loss', loss)

  def test_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class = self.model(image)
    
    # Loss
    loss = self.loss_fn(target_class, pred_class)

    # Accuracy
    self.test_accuracy(pred_class, target_class.int())
    self.log('test_acc', self.test_accuracy, prog_bar=True)
    self.log('test_loss', loss)

  def loss_fn(self, target_class, pred_class, expl_coeff=0.0):
    cls_loss = F.cross_entropy(pred_class, target_class, weight=self.class_weight.cuda())    

    return cls_loss

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
    return optimizer