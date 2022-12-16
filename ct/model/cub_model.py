import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torchvision
from .ct import ConceptTransformer
from .cub_backbone import VIT_Backbone

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def global_attention_loss(attn, concepts):
    attn_s = attn.squeeze()
    concept_norm = (concepts.T / concepts.sum(axis=1).T).T
    n_concepts = concept_norm.shape[-1]
    return n_concepts * F.mse_loss(attn_s, concept_norm, reduction="mean")


def spatial_attention_loss(attn, concept):
    if attn is None:
        return 0.0
    numerical = ~torch.isnan(concept).any(-1).squeeze()
    if not torch.any(numerical):
        return 0.0
    concept_num = concept[numerical] / concept.sum(-1, keepdims=True)[numerical]
    attn_num = attn[numerical]
    n_concepts = concept_num.shape[-1]
    return n_concepts * F.mse_loss(concept_num, attn_num, reduction="mean")





class _CUB_CT(nn.Module):
    def __init__(self, n_global_concepts, n_spatial_concepts, dim, n_classes, num_heads) -> None:
        super().__init__()
        self.backbone = VIT_Backbone()
        self.backbone = self.backbone.to(device)
        self.global_ct = ConceptTransformer(
            n_concepts=n_global_concepts,
            dim=dim,
            n_classes=n_classes,
            att_pool=False,
            num_heads=num_heads,
            is_spatial=False)
        
        self.spatial_ct = ConceptTransformer(
            n_concepts=n_spatial_concepts,
            dim=dim,
            n_classes=n_classes,
            att_pool=False,
            num_heads=num_heads,
            is_spatial=True
        )
    
    def forward(self, image):
        embedding = self.backbone(image) 
        out, attn_global = self.global_ct(embedding[:, 0].unsqueeze(1))
        attn_global = attn_global.mean(1)
        out_spatial, attn_spatial = self.spatial_ct(embedding[:, 1:])
        attn_spatial = attn_spatial.mean(1)
        out = out + out_spatial
        return out, attn_global, attn_spatial


class CUB_CT(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = _CUB_CT(
            n_global_concepts=args.n_global_concepts,
            n_spatial_concepts=args.n_spatial_concepts,
            dim=args.dim,
            n_classes=args.n_classes,
            num_heads=args.num_heads
        )
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes, top_k=1)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes, top_k=1)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes, top_k=1)

    
    
    def training_step(self, batch, batch_idx):
        image = batch['image']
        target_class = batch['label']
        global_concepts = batch['global_attr']
        spatial_concepts = batch['spatial_attr']
        predictions, attn_global, attn_spatial = self.model(image)
        # Loss
        loss, cls_loss, explain_loss = self.loss_fn(
                target_class = target_class, 
                target_global_concept = global_concepts,
                target_spatial_concept = spatial_concepts,
                pred_class = predictions, 
                attn_global = attn_global,
                attn_spatial = attn_spatial)
        # Accuracy
        self.train_accuracy(predictions, target_class.int())
        self.log('train_loss', loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_expl_loss', explain_loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target_class = batch['label']
        global_concepts = batch['global_attr']
        spatial_concepts = batch['spatial_attr']

        predictions, attn_global, attn_spatial = self.model(image)
        
        # Loss
        loss, cls_loss, explain_loss = self.loss_fn(
                target_class = target_class, 
                target_global_concept = global_concepts,
                target_spatial_concept = spatial_concepts,
                pred_class = predictions, 
                attn_global = attn_global,
                attn_spatial = attn_spatial)
        # Accuracy
        self.val_accuracy(predictions, target_class.int())
        self.log('val_cls_loss', cls_loss)
        self.log('val_expl_loss', explain_loss)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        image = batch['image']
        target_class = batch['label']
        global_concepts = batch['global_attr']
        spatial_concepts = batch['spatial_attr']

        predictions, attn_global, attn_spatial = self.model(image)
        
        # Loss
        loss, cls_loss, explain_loss = self.loss_fn(
                target_class = target_class, 
                target_global_concept = global_concepts,
                target_spatial_concept = spatial_concepts,
                pred_class = predictions, 
                attn_global = attn_global,
                attn_spatial = attn_spatial)

        # Accuracy
        self.test_accuracy(predictions, target_class.int())
        self.log('test_cls_loss_' + self.test_mode, cls_loss)
        self.log('test_expl_loss_' + self.test_mode, explain_loss)
        self.log('test_acc_' + self.test_mode, self.test_accuracy, prog_bar=True)
        self.log('test_loss_' + self.test_mode, loss)

    def predict_step(self, batch, batch_idx=None):
        image = batch['image']
        target_class = batch['label']
        global_concepts = batch['global_attr']
        spatial_concepts = batch['spatial_attr']

        predictions, attn_global, attn_spatial = self.model(image)

        predictions = predictions.argmax(dim=-1)
        
        return target_class, global_concepts, spatial_concepts, predictions, attn_global, attn_spatial

    def loss_fn(self, 
                target_class, 
                target_global_concept,
                target_spatial_concept,
                pred_class, 
                attn_global,
                attn_spatial):

        cls_loss = F.cross_entropy(pred_class, target_class, weight=None)
        

        explain_loss = global_attention_loss(attn_global,  target_global_concept) + spatial_attention_loss(attn_spatial, target_spatial_concept)
    
        loss = cls_loss + self.args.expl_coeff * explain_loss

        return loss, cls_loss.item(), explain_loss.item()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.05)
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
        
        

            
