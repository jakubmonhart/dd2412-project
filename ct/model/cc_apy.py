"""
Concept Classifier for aPY dataset
"""

import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torchvision

from .ct import ConceptTransformer

import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)


class CC_aPY_torch(nn.Module):
  n_classes = 20
  n_concepts = 64

  def __init__(self, resnet='50', dropout=0):
    super().__init__()
    
    # Backbone
    if resnet == '50':
      resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
      # self.linear = nn.Linear(2048, self.n_classes)
      dim = 2048
    elif resnet == '34':
      resnet = torchvision.models.resnet34(weights='IMAGENET1K_V1')
      # self.linear = nn.Linear(512, self.n_classes)
      dim = 512

    layers = list(resnet.children())[:-1] # Include Adaptive average maxpool (AdaptiveAvgPool2d)  
    self.feature_extractor = nn.Sequential(*layers)

    # Concept Classifier
    self.concept_classifier = nn.Sequential(
      nn.Dropout(p=dropout),
      nn.Linear(dim, dim),
      nn.Dropout(p=dropout),
      nn.ReLU(),
      nn.Linear(dim, self.n_concepts)
    )

    # Class Classifier
    self.class_classifier = nn.Sequential(
      nn.Dropout(p=dropout),
      nn.ReLU(),
      nn.Linear(self.n_concepts, dim),
      nn.Dropout(p=dropout),
      nn.ReLU(),
      nn.Linear(dim, self.n_classes)
    )

  def forward(self, x):
    """
    inputs:
      x - batch images
    """

    x = self.feature_extractor(x)
    x = torch.flatten(x, 1)
    pred_concept = self.concept_classifier(x)

    pred_class = self.class_classifier(pred_concept)
    
    pred_concept = pred_concept.softmax(dim=-1) # In Concept Transformer, attn output also goes through softmax
    return pred_class, pred_concept


class CC_aPY(pl.LightningModule):
  n_classes = 20 # TODO - considering only pascal data for now

  def __init__(self, args):
    super().__init__()

    self.args = args
    self.save_hyperparameters()
  
    self.model = CC_aPY_torch(resnet=args.resnet, dropout=args.dropout)

    class_counts = [183, 161, 254, 228, 296, 73, 528, 187, 446, 103, 113, 244, 153, 149, 2488, 225, 123, 121, 90, 150]
    class_counts = torch.tensor(class_counts)
    class_weight = 1 / class_counts
    self.class_weight = class_weight / class_weight.sum()

    self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, top_k=1)
    self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, top_k=1)
    self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, top_k=1)

  def training_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)

    # Loss
    loss, cls_loss, expl_loss = self.loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    self.train_accuracy(pred_class, target_class.int())
    self.log('train_cls_loss', cls_loss)
    self.log('train_expl_loss', expl_loss, prog_bar=True)
    self.log('train_acc', self.train_accuracy, prog_bar=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    
    # Loss
    loss, cls_loss, expl_loss = self.loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    self.val_accuracy(pred_class, target_class.int())
    self.log('val_acc', self.val_accuracy, prog_bar=True)
    self.log('val_cls_loss', cls_loss)
    self.log('val_expl_loss', expl_loss, prog_bar=True)
    self.log('val_loss', loss)

  def test_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    
    # Loss
    loss, cls_loss, expl_loss = self.loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    self.test_accuracy(pred_class, target_class.int())
    self.log('test_acc', self.test_accuracy, prog_bar=True)
    self.log('test_cls_loss', cls_loss)
    self.log('test_expl_loss', expl_loss, prog_bar=True)
    self.log('test_loss', loss)

  def predict_step(self, batch, batch_idx=None):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    pred_class = pred_class.argmax(dim=-1)
    attn = attn.squeeze(2)
    attn = torch.mean(attn, dim=1)

    # Unnormalize
    image = torchvision.transforms.functional.normalize(image, mean=(0, 0, 0), std=(1/0.229, 1/0.224, 1/0.225))
    image = torchvision.transforms.functional.normalize(image, mean=(-0.485, -0.456, -0.406), std=(1, 1, 1))

    return image, target_class, target_concept, pred_class, attn

  def loss_fn(self, target_class, target_concept, pred_class, pred_concept):
    if self.args.loss_weight:
      cls_loss = F.cross_entropy(pred_class, target_class, weight=self.class_weight.cuda())
    else:
      cls_loss = F.cross_entropy(pred_class, target_class, weight=None)

    # TODO - should we normalize the attentions to sum to 1 as they do in the paper?
    # attn went through softmax before, we need to account for that
    norm = target_concept.sum(-1, keepdims=True)    
    normalized_target_concept = (target_concept / norm)
    n_concepts = self.model.n_concepts
  
    expl_loss = n_concepts*nn.functional.mse_loss(pred_concept, normalized_target_concept)
  
    loss = cls_loss + self.args.expl_coeff*expl_loss

    return loss, cls_loss.item(), expl_loss.item()

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
    return optimizer