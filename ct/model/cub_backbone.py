import torch
from torchvision.models import vit_l_16,  ViT_L_16_Weights
import torch.nn as nn


class VIT_Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vit = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
        self._process_input = vit._process_input
        self.class_token = vit.class_token
        self.encoder = vit.encoder
    def forward(self, x):
        # forward method got from torchvision implementation
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x
        return x



