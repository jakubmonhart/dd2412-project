"""
Concept Transformer is implemented here.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .attention import CrossAttention

class ConceptTransformer(nn.Module):
  """
  Only non-spatial for now (enough for MNIST dataset).
  """

  def __init__(self, n_concepts, dim, n_classes, num_heads=2):
    """
    num_heads is not specified in the text, from the code, it looks like 2 is used for MNIST experiment
    """
    super().__init__()

    self.concepts = nn.Parameter(torch.zeros(1, n_concepts, dim), requires_grad=True) # First dimension is 1 to make it broadcastable over batches
    # Initialization of the parameter is not described in the text, only in the code in Appendix C (TODO - should we try other options)
    nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(dim))
    self.cross_attention = CrossAttention(dim=dim, out_dim=n_classes, num_heads=num_heads)

    self.token_attention_pool = nn.Linear(dim, 1)


  def forward(self, x):
  
    # Pool over patches (taken from the Appendix C of the paper) - TODO - why not use simple mean here?
    token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
    x_pooled = torch.matmul(token_attn, x)

    out, attn = self.cross_attention(x_pooled, self.concepts) # out.shape = [B, P, n_classes] - P=1 for mnist experiment
    out = out.squeeze(1) # Squeeze over patches (there is only one)
    # attn = attn.mean(1) # Average attention over heads

    return out, attn

