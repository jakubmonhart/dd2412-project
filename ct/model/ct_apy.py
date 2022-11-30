import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F

from .cct import CCT_torch
from .ct import ConceptTransformer

import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)


class CT_aPY_torch(nn.Module):
  dim = 128
  n_classes = 32
  n_concepts = 64

  """
  TODO:
    - Backend CT uses ResNet50 as the tokenizer
  """

  def __init__(self):
    super().__init__()
    
    self.cct = CCT_torch(
      img_size=256, n_input_channels=3, num_layers=2, num_heads=2,
      embedding_dim=self.dim, num_classes=self.n_classes, mlp_ratio=1, backbone=True)

    # Concept transformer
    self.concept_transformer = ConceptTransformer(n_concepts=self.n_concepts, dim=self.dim, n_classes=self.n_classes)

  def forward(self, images):
    """
    inputs:
      x - batch images
    """

    patches = self.cct(images)
    out, attn = self.concept_transformer(patches)

    return out, attn


def loss_fn(target_class, target_concept, pred_class, attn, expl_coeff=2.0):
  cls_loss = F.cross_entropy(pred_class, target_class)
  
  # We are using multiple-head attention -> we need to average over them?
  attn = attn.squeeze(2)
  attn = torch.mean(attn, dim=1)

  # TODO - should we normalize the attentions to sum to 1 as they do in the paper?
  
  expl_loss = nn.functional.mse_loss(target_concept, attn)
  
  loss = cls_loss + expl_coeff*expl_loss

  return loss, cls_loss.item(), expl_loss.item()


class CT_aPY(pl.LightningModule):
  def __init__(self):
    super().__init__()
    
    self.model = CT_aPY_torch()

    # self.accuracy = torchmetrics.Accuracy()
    self.accuracy_fn = torchmetrics.functional.accuracy

  def training_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)

    # Loss
    loss, cls_loss, expl_loss = loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    # self.accuracy(pred_class, target_class.int())
    acc = self.accuracy_fn(pred_class, target_class.int(), threshold=0.0)
    self.log('train_cls_loss', cls_loss)
    self.log('train_expl_loss', expl_loss)
    self.log('train_acc', acc, prog_bar=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    
    # Loss
    loss, cls_loss, expl_loss = loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    acc = self.accuracy_fn(pred_class, target_class.int(), threshold=0.0)
    self.log('val_cls_loss', cls_loss)
    self.log('val_expl_loss', expl_loss)
    self.log('val_acc', acc, prog_bar=True)
    self.log('val_loss', loss)


  def test_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    
    # Loss
    loss, cls_loss, expl_loss = loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    acc = self.accuracy_fn(pred_class, target_class.int(), threshold=0.0)
    self.log('test_cls_loss', cls_loss)
    self.log('test_expl_loss', expl_loss)
    self.log('test_acc', acc, prog_bar=True)
    self.log('test_loss', loss)

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=1e-3)
    return optimizer