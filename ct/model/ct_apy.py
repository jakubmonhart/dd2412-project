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


class CT_aPY_torch(nn.Module):
  n_classes = 20
  n_concepts = 64
  image_size = 224

  """
  cct_n_heads: number of heads of attention inside Compact Transformer (cct)
  num_heads: number of heads of cross-attention inside Concept Transformer (ct)
  """

  def __init__(self, dim, cct_n_layers, cct_n_heads, cct_mlp_ratio, num_heads, resnet='50'):
    super().__init__()
    
    self.cct = CCT_ResNet_torch(
      img_size=self.image_size, n_input_channels=3, num_layers=cct_n_layers, num_heads=cct_n_heads,
      embedding_dim=dim, num_classes=self.n_classes, mlp_ratio=cct_mlp_ratio, resnet=resnet)

    # Concept transformer
    self.concept_transformer = ConceptTransformer(
      n_concepts=self.n_concepts, dim=dim, n_classes=self.n_classes, num_heads=num_heads)

  def forward(self, images):
    """
    inputs:
      x - batch images
    """

    patches = self.cct(images)
    out, attn = self.concept_transformer(patches)

    return out, attn


class CT_aPY(pl.LightningModule):
  n_classes = 20 # TODO - considering only pascal dataset for now

  def __init__(self, args):
    super().__init__()

    self.args = args
    self.save_hyperparameters()
  
    self.model = CT_aPY_torch(
      dim=args.dim, cct_n_layers=args.cct_n_layers, cct_n_heads=args.cct_n_heads,
      cct_mlp_ratio=args.cct_mlp_ratio, num_heads=args.num_heads, resnet=args.resnet)

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
    self.log('train_loss', loss)
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
    self.log('val_cls_loss', cls_loss)
    self.log('val_expl_loss', expl_loss)
    self.log('val_acc', self.val_accuracy, prog_bar=True)
    self.log('val_loss', loss)

  def test_step(self, batch, batch_idx):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    
    # Loss
    loss, cls_loss, expl_loss = self.loss_fn(target_class, target_concept, pred_class, attn)

    # Accuracy
    self.test_accuracy(pred_class, target_class.int())
    self.log('test_cls_loss', cls_loss)
    self.log('test_expl_loss', expl_loss)
    self.log('test_acc', self.test_accuracy, prog_bar=True)
    self.log('test_loss', loss)

  def predict_step(self, batch, batch_idx=None):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    pred_class = pred_class.argmax(dim=-1)
    attn = attn.squeeze(2)
    attn = torch.mean(attn, dim=1)

    return target_class, target_concept, pred_class, attn

  def my_predict_step(self, batch, batch_idx=None):
    image, (target_class, target_concept) = batch
    pred_class, attn = self.model(image)
    pred_class = pred_class.argmax(dim=-1)
    attn = attn.squeeze(2)
    attn = torch.mean(attn, dim=1)

    # Unnormalize
    image = torchvision.transforms.functional.normalize(image, mean=(0, 0, 0), std=(1/0.229, 1/0.224, 1/0.225))
    image = torchvision.transforms.functional.normalize(image, mean=(-0.485, -0.456, -0.406), std=(1, 1, 1))

    return image, target_class, target_concept, pred_class, attn

  def loss_fn(self, target_class, target_concept, pred_class, attn):
    if self.args.loss_weight:
      cls_loss = F.cross_entropy(pred_class, target_class, weight=self.class_weight.cuda())
    else:
      cls_loss = F.cross_entropy(pred_class, target_class, weight=None)

    # We are using multiple-head attention -> we need to average over them?
    attn = attn.squeeze(2)
    attn = torch.mean(attn, dim=1)

    # TODO - should we normalize the attentions to sum to 1 as they do in the paper?
    # attn went through softmax before, we need to account for that
    norm = target_concept.sum(-1, keepdims=True)    
    normalized_target_concept = (target_concept / norm)
    n_concepts = self.model.n_concepts
  
    expl_loss = n_concepts*nn.functional.mse_loss(attn, normalized_target_concept)
  
    loss = cls_loss + self.args.expl_coeff*expl_loss

    return loss, cls_loss.item(), expl_loss.item()

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)

    if self.args.scheduler == 'none':
      return [optimizer]
    
    elif self.args.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=self.args.epochs, verbose=True)
      return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
    
    elif self.args.scheduler == 'cosine_restart':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=self.args.warmup_epochs, verbose=True)
      return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
      