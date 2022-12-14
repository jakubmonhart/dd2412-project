import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torchvision
from .ct import ConceptTransformer
from .cub_backbone import VIT_Backbone


class _CUB_CT(nn.Module):
    def __init__(self, n_global_concepts, n_spatial_concepts, dim, n_classes, num_heads) -> None:
        super().__init__()
        self.backbone = VIT_Backbone()
        self.global_ct = ConceptTransformer(
            n_concepts=n_global_concepts,
            dim=dim,
            n_classes=n_classes,
            att_pool=True,
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
    
    def forward(self, batch):
        image = batch['image']
        clobal_concepts = batch['global_attr']
        spatial_concepts = batch['spatial_attr']
        embedding = self.backbone(image)
        print(embedding.shape)